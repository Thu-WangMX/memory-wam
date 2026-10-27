from copy import deepcopy
import types
import numpy as np
import pytest
import torch
from test_robomme_past_action import make_model as old_model, batch as old_batch
from test_robomme_past_flow import video_cache
from fastwam.memory.past_action import DreamDojoLabels
from fastwam.memory.past_future_action import MODE, SCHEMA, sampling_contract, past_future_layout, training_targets
from fastwam.memory.robomme_clock import decision_frames
from fastwam.models.wan22.fastwam_helios_past_future import FastWAMHeliosPastFuture
from fastwam.memory.helios_history import memory_variant_profile, validate_memory_variant


def model(future=False):
    m=old_model('sparse')
    m.helios_config['prediction_horizon']=32
    m.video_expert.mtm_pred_size=2
    if future:
        label=deepcopy(m.helios_config['past_action']['label_contract'])
        label.update(schema=SCHEMA, **sampling_contract())
        m.__class__=FastWAMHeliosPastFuture
        common=dict(hidden_dim=32,ffn_dim=64,text_dim=16,freq_dim=16,eps=1e-6,
                    num_heads=2,attn_head_dim=24,num_layers=2,action_dim=8)
        m._initialize_past_action(dict(enabled=True,learning_mode='flow_matching',
                                     target_mode=MODE,label_contract=label),common)
    return m


def batch(m):
    x=old_batch(m)
    x.update(input_latents=torch.randn(2,4,3,8,8), action=torch.randn(2,32,8),
             image_is_pad=torch.zeros(2,9,dtype=torch.bool), action_is_pad=torch.zeros(2,32,dtype=torch.bool))
    if m.query_slots==18:x['past_action_loss_valid']=x['past_action_transition_valid'].clone()
    return x


def pre(m,x):
    b=len(x['action'])
    vp=m._video_pre(x['anchor_latents'],x['history_latents'],x['input_latents'][:,:,1:],
                    torch.full((b,),500.),x['video_context'],x['video_context_mask'])
    ap=m.action_expert.pre_dit(x['action'],torch.full((b,),700.),x['context'],x['context_mask'])
    ap['freqs']=m._action_freqs(32,vp)
    return vp,ap


@pytest.mark.parametrize('boundary',[0,1,17,35,49,1100])
def test_queries_do_not_use_future_episode_length_and_labels_never_cross_boundary(boundary):
    current=boundary+16
    x=past_future_layout(decision_frames(current+1,boundary),boundary)
    assert x['past_action_visual_slots'][-2:].tolist()==[99,100]
    assert x['past_action_relative_time'][-2:].tolist()==[1.,2.]
    assert x['past_action_transition_frames'][-2:].tolist()==[
        list(range(current+2,current+17,2)),list(range(current+18,current+33,2))]
    labels=DreamDojoLabels.__new__(DreamDojoLabels)
    n=current+6;labels.offsets={0:(0,n)}
    labels._latents=np.arange(n,dtype=np.float32)[:,None].repeat(64,1)
    labels.valid=np.ones(n,dtype=bool);labels.valid[[0,boundary]]=False
    labels.phase=np.where(np.arange(n)<boundary,1,3);labels.phase[boundary]=2
    labels.mean=np.zeros(64,dtype=np.float32);labels.std=np.ones(64,dtype=np.float32)
    before=deepcopy(x);y,loss_valid=training_targets(labels,0,x)
    assert loss_valid[-2:].sum()==2
    assert x['past_action_transition_valid'][-2:].all()
    assert y[-2,:2,0].tolist()==[current+2,current+4]
    assert torch.count_nonzero(y[~loss_valid])==0
    for k in x:torch.testing.assert_close(x[k],before[k],atol=0,rtol=0)


@pytest.mark.parametrize('future',[False,True])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_training_32_actions_two_video_latents_backward(monkeypatch,future,checkpoint):
    monkeypatch.setenv('FASTWAM_CHECKPOINT_WHOLE_LAYER',str(int(checkpoint)))
    m=model(future).train();m.mot.mot_checkpoint_mixed_attn=checkpoint
    x=batch(m);m.build_inputs=types.MethodType(lambda self,sample,tiled=False:sample,m)
    loss,metrics=m.training_loss(x);loss.backward()
    for p in [m.video_expert.patch_long.weight,m.past_action_expert.noisy_latent_embedding.weight,
              m.action_expert.head.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    if future:
        assert metrics['loss_future_action']>0
        assert m.past_action_expert.direction_embedding.weight.grad.abs().sum()>0
    x['action']=x['action'][:,:16]
    with pytest.raises(ValueError,match='lengths'):m.training_loss(x)


def test_future_cannot_leak_to_past_and_full_forward_matches_cache_multi_step():
    m=model(True).eval();x={k:v[:1] for k,v in batch(m).items()};vp,_=pre(m,x)
    cache=video_cache(m,vp);h=vp['meta']['mtm']['L_hist'];valid=x['past_action_transition_valid'][...,None]
    action=x['action'].clone();past=torch.randn(1,18,8,64,generator=torch.Generator().manual_seed(121)).masked_fill(~valid,0)
    steps,deltas=m.infer_action_scheduler.build_inference_schedule(3,m.device,torch.float32)
    freqs=m._action_freqs(32,vp)
    for t,delta in zip(steps,deltas):
        ap=m.action_expert.pre_dit(action,t[None],x['context'],x['context_mask']);ap['freqs']=freqs
        pp,qvalid=m._flow_pre(x,vp,x['video_context'],x['video_context_mask'],past,t[None])
        joint=m._joint_past_forward(vp,ap,pp,qvalid)
        changed=past.clone();changed[:,16:]+=999
        cp,_=m._flow_pre(x,vp,x['video_context'],x['video_context_mask'],changed,t[None])
        alt=m._joint_past_forward(vp,ap,cp,qvalid)
        torch.testing.assert_close(joint['past_action'][:,:16],alt['past_action'][:,:16],atol=0,rtol=0)
        assert not torch.allclose(joint['action'],alt['action'])
        combined,key_valid,p_cached=m.mot.prefill_past_action(pp,cache,qvalid)
        torch.testing.assert_close(joint['past_action'],p_cached,atol=4e-6,rtol=4e-5)
        mask=torch.cat((key_valid[0],torch.ones(32,dtype=torch.bool)))[None].expand(32,-1)
        a_cached=m._predict_action_noise_with_cache(action,t[None],x['context'],x['context_mask'],freqs,combined,mask,h+18)
        velocity=m.action_expert.post_dit(joint['action'],ap)
        torch.testing.assert_close(velocity,a_cached,atol=4e-6,rtol=4e-5)
        action=m.infer_action_scheduler.step(velocity,delta,action)
        past=m.infer_action_scheduler.step(m._past_predictions(joint['past_action']).masked_fill(~valid,0),delta,past).masked_fill(~valid,0)
    got,extra=m._denoise_action_with_condition(x['action'].clone(),steps,deltas,x['context'],x['context_mask'],freqs,
        cache,torch.ones(32,h+32,dtype=torch.bool),h,{},vp=vp,context=x['video_context'],context_mask=x['video_context_mask'],
        observation_frame_indices=decision_frames(82,49),execution_start=49,generator=torch.Generator().manual_seed(121),rand_device='cpu')
    torch.testing.assert_close(got,action,atol=6e-6,rtol=6e-5)
    torch.testing.assert_close(extra['past_action_prediction'],past[0],atol=6e-6,rtol=6e-5)


def test_separate_future_loss_padding_and_no_teacher_at_sigma_one():
    m=model(True).eval();x=batch(m);x['past_action_loss_valid'][:,16:]=False
    vp,ap=pre(m,x);inputs=dict(x,action_flow_timestep=torch.ones(2)*1000)
    torch.manual_seed(10);a=m._training_expert_forward(vp,ap,inputs,x)
    y=deepcopy(x);y['past_action_target'].add_(10000)
    torch.manual_seed(10);b=m._training_expert_forward(vp,ap,inputs,y)
    torch.testing.assert_close(a['action'],b['action'],atol=0,rtol=0)
    loss,metrics=m._extra_training_loss(a,x)
    assert metrics['loss_future_action']==0 and torch.isfinite(loss)
    out=dict(a,past_action_flow_target=a['past_action_flow_target'].clone())
    out['past_action_flow_target'][:,16:]+=999
    changed,_=m._extra_training_loss(out,x)
    torch.testing.assert_close(loss,changed,atol=0,rtol=0)


def test_contracts_reject_cross_variant_resume_and_light_profile(tmp_path):
    for future in [False,True]:
        m=model(future);p=tmp_path/f'{future}.pt';m.save_checkpoint(p,step=1)
        assert model(future).load_checkpoint(p)==1
        for wrong in [old_model('sparse'),model(not future)]:
            with pytest.raises(ValueError,match='contract'):wrong.load_checkpoint(p)
        contract=m._helios_inference_contract()
        assert contract['action_horizon']==32 and contract['replan_stride']==16 and contract['future_video_latents']==2
    for variant,kernel in [('long99_less_temporal',[2,8,8]),('long99_less_spatial',[4,4,4])]:
        cfg=memory_variant_profile(variant);assert cfg['patch_long']==kernel
        validate_memory_variant(variant,cfg['history_sizes'],kernel,cfg['patch_mid'],2)
        with pytest.raises(ValueError):validate_memory_variant(variant,cfg['history_sizes'],[4,8,8],cfg['patch_mid'],2)


def test_action_rope_spans_two_video_intervals():
    m=model();vp,_=pre(m,batch(m));base=vp['meta']['mtm']['current_frame_index']
    got=m._action_freqs(32,vp)
    for i in range(2):
        expected=m._build_video_aligned_action_freqs(action_seq_len=16,temporal_base=base+i,
            grid_h=4,grid_w=4,device=m.device)
        torch.testing.assert_close(got[16*i:16*(i+1)],expected,atol=0,rtol=0)


def test_policy_predicts32_returns16_and_keeps_observation_clock(tmp_path,monkeypatch):
    import json
    from test_robomme_helios import RecordingModel,RecordingStream,obs
    from experiments.robomme.policy import AnchorHeliosPolicy
    from fastwam.datasets.robodojo import text_filename
    class Model(RecordingModel):
        helios_config=dict(RecordingModel.helios_config,prediction_horizon=32)
        def infer_action(self,**kw):
            self.call=kw
            return {'action':torch.arange(32,dtype=torch.float32)[:,None].expand(32,8)}
    (tmp_path/'stats.json').write_text(json.dumps({k:{'default':{'global_mean':[0.]*8,'global_std':[1.]*8}} for k in ('state','action')}))
    torch.save(dict(context=torch.ones(128,4096),mask=torch.ones(128,dtype=torch.bool)),tmp_path/text_filename('test'))
    monkeypatch.setattr('experiments.robomme.policy.CausalVAEStream',RecordingStream)
    m=Model();p=AnchorHeliosPolicy(m,tmp_path/'stats.json',tmp_path)
    a=p.infer(obs(0,49,True));assert a['actions'].shape==(16,8)
    np.testing.assert_array_equal(a['actions'][:,0],np.arange(16))
    assert m.call['action_horizon']==32
    p.infer(obs(49,65));assert p.observation_frame_indices[-1]==64
    assert [x.shape[2] for x in p.execution_encoder.inputs]==[1,4]


def test_deployment_accepts_saved_future_dataset_and_horizon(tmp_path,monkeypatch):
    import json
    from omegaconf import OmegaConf
    from fastwam.memory.robomme_clock import CONTRACT
    from experiments.robomme.policy import AnchorHeliosPolicy
    cfg=OmegaConf.create(dict(model=dict(helios_config=dict(prediction_horizon=32,
        observation_contract=CONTRACT,past_action=dict(enabled=True,target_mode=MODE,learning_mode='flow_matching'))),
        data=dict(train=dict(_target_='fastwam.datasets.robomme_past_future.RoboMMEPastFutureActionDataset',
                            prepared_root=str(tmp_path))),
        robomme_provenance=dict(stats_sha256='ok',assets_sha256='ok',vae_sha256='ok')))
    OmegaConf.save(cfg,tmp_path/'config.yaml')
    (tmp_path/'provenance.json').write_text(json.dumps(dict(text_sha256={})))
    monkeypatch.setattr('experiments.robomme.policy.sha256',lambda p:'ok')
    class ReachedFactory(Exception):pass
    def instantiate(config,**kw):
        assert config.helios_config.prediction_horizon==32
        assert config.helios_config.past_action.target_mode==MODE
        raise ReachedFactory()
    monkeypatch.setattr('experiments.robomme.policy.instantiate',instantiate)
    with pytest.raises(ReachedFactory):AnchorHeliosPolicy.from_checkpoint(tmp_path/'config.yaml','unused')


@pytest.mark.parametrize('variant,expected',[('long99_less_temporal',822),('long99_less_spatial',1110)])
def test_light_memory_real_spatial_grid_and_future_gradient(variant,expected):
    from fastwam.models.wan22.helios_video import MTMVideoDiT
    m=model(True);profile=memory_variant_profile(variant)
    video=MTMVideoDiT(hidden_dim=32,ffn_dim=64,text_dim=16,freq_dim=16,eps=1e-6,
        num_heads=2,attn_head_dim=24,num_layers=2,in_dim=4,out_dim=4,patch_size=(1,2,2),
        has_image_input=False,seperated_timestep=True,multi_term_memory=True,
        mtm_history_sizes=(96,2,1),mtm_anchor_size=2,mtm_pred_size=2,
        mtm_patch_kernel_long=tuple(profile['patch_long']))
    m.mot.mixtures['video']=video;m.helios_config.update(memory_variant=variant,patch_long=profile['patch_long'])
    x={k:v[:1] for k,v in batch(m).items()}
    x.update(anchor_latents=torch.randn(1,2,4,1,24,20),history_latents=torch.randn(1,99,4,1,24,20),
             input_latents=torch.randn(1,4,3,24,20))
    vp,ap=pre(m,x)
    assert vp['meta']['mtm']['L_hist']==expected and vp['meta']['mtm']['L_pred']==240
    out=m._training_expert_forward(vp,ap,dict(x,action_flow_timestep=torch.tensor([700.])),x)
    loss,_=m._extra_training_loss(out,x);loss.backward()
    assert video.patch_long.weight.grad.abs().sum()>0
