from copy import deepcopy
from pathlib import Path
import types
import pytest
import torch
from hydra import compose, initialize_config_dir

from fastwam.memory.helios_history import memory_variant_profile, validate_memory_variant
from fastwam.memory.robomme_dense_clock import dense_prefix_frames
from fastwam.models.wan22.fastwam_helios_asv import FastWAMHeliosPastFutureASV
from test_robomme_query_density import make_model, batch


def model(asv=True, temporal=False):
    m = make_model()
    m.helios_config.update(history_padding='compact', anchor_mode='demo_execution_first')
    if temporal:
        m.helios_config.update(memory_variant='long75_less_temporal', patch_long=[2,8,8])
        v = m.video_expert
        v.mtm_patch_kernel_long = (2,8,8)
        v._long_temporal_ratio = 2
        v.patch_long = torch.nn.Conv3d(4,32,(2,8,8),stride=(2,8,8))
        v._init_mtm_patches_from_patch_embedding()
    if asv:
        m.__class__ = FastWAMHeliosPastFutureASV
        m._configure_asv()
    return m


def inputs(m, current=67, boundary=35, b=2):
    x = {k:v[:b] for k,v in batch(m,current=current,boundary=boundary).items()}
    count = min(75,len(dense_prefix_frames(current,boundary)))
    x['history_valid'] = (torch.arange(75)[None] >= 75-count).expand(b,-1).clone()
    x['anchor_valid'] = torch.tensor([[boundary>0,True]]).expand(b,-1).clone()
    return x


def prepare(m,x,future=None,tv=None):
    future = x['input_latents'][:,:,1:] if future is None else future
    tv = torch.full((future.shape[0],),500.) if tv is None else tv
    vp=m._video_pre(x['anchor_latents'],x['history_latents'],future,tv,
                    x['video_context'],x['video_context_mask'],history_valid=x['history_valid'])
    return m._apply_anchor_validity(vp,x['anchor_valid'],32)


@pytest.mark.parametrize('temporal',[False,True])
@pytest.mark.parametrize('current,boundary',[(0,0),(67,35),(1635,35)])
def test_training_gradient_padding_and_backward_compatibility(temporal,current,boundary,monkeypatch):
    monkeypatch.setenv('FASTWAM_CHECKPOINT_WHOLE_LAYER','1')
    m=model(asv=not temporal,temporal=temporal).train()
    x=inputs(m,current,boundary)
    # A short sample and a long sample coexist in the same minibatch.
    if current>100:
        x['history_valid'][1,:-1]=False
        x['anchor_valid'][1,0]=False
    m.build_inputs=types.MethodType(lambda self,sample,tiled=False:sample,m)
    loss,metrics=m.training_loss(x);loss.backward()
    assert torch.isfinite(loss) and metrics['loss_future_action']>0
    assert m.action_expert.head.weight.grad.abs().sum()>0
    assert m.past_action_expert.head.weight.grad.abs().sum()>0
    if current>100:assert m.video_expert.patch_long.weight.grad.abs().sum()>0


def test_asv_mask_and_action_gradient_to_future_video():
    m=model().eval();x=inputs(m)
    vp=prepare(m,x)
    ap=m.action_expert.pre_dit(x['action'],torch.full((2,),700.),x['context'],x['context_mask'])
    ap['freqs']=m._action_freqs(32,vp)
    noise=torch.randn_like(x['past_action_target'])
    pp,valid=m._flow_pre(x,vp,x['video_context'],x['video_context_mask'],noise,torch.full((2,),700.))
    mask=m._three_expert_mask(vp,32,valid)
    h,f=vp['meta']['mtm']['L_hist'],vp['meta']['mtm']['L_pred'];n=h+f+32
    assert mask[:,h+f:n,h:h+f].all()
    assert not mask[:,:h,h:].any()
    assert not mask[:,n:,h:h+f+32].any()
    assert not mask[:,h:h+f,h+f:].any()
    assert not mask[:,n:n+32,n+32:].any()
    vp['tokens'].retain_grad()
    a=m._joint_past_forward(vp,ap,pp,valid)
    a['action'].square().mean().backward()
    assert vp['tokens'].grad[:,h:].abs().sum()>0
    changed=dict(vp,tokens=torch.cat((vp['tokens'][:,:h],vp['tokens'][:,h:]+10),1))
    b=m._joint_past_forward(changed,ap,pp,valid)
    assert not torch.allclose(a['action'],b['action'])
    torch.testing.assert_close(a['past_action'],b['past_action'])
    torch.testing.assert_close(a['video'][:,:h],b['video'][:,:h])


@pytest.mark.parametrize('current,boundary',[(0,0),(67,35),(1635,35)])
def test_cached_solver_matches_dense_joint_at_every_step(current,boundary):
    m=model().eval();x=inputs(m,current,boundary,b=1)
    frames=dense_prefix_frames(current,boundary)
    with torch.no_grad():
        vp=prepare(m,x)
        kv,validh,_=m._prefill_inference_condition(vp,x['video_context'],x['video_context_mask'],frames,boundary)
        h=vp['meta']['mtm']['L_hist']
        gen=torch.Generator().manual_seed(42)
        past=torch.randn((1,34,8,64),generator=gen)
        valid=x['past_action_transition_valid'];past.masked_fill_(~valid[...,None],0)
        future=torch.randn(vp['_asv_future_shape'],generator=gen)
        action=x['action'].clone()
        steps,deltas=m.infer_action_scheduler.build_inference_schedule(3,m.device,torch.float32)
        vs,vd=m.infer_video_scheduler.build_inference_schedule(3,m.device,torch.float32)
        freqs=m._action_freqs(32,vp)
        for ta,da,tv,dv in zip(steps,deltas,vs,vd):
            cur=prepare(m,x,future,tv[None])
            ap=m.action_expert.pre_dit(action,ta[None],x['context'],x['context_mask']);ap['freqs']=freqs
            pp,qvalid=m._flow_pre(x,cur,x['video_context'],x['video_context_mask'],past,ta[None])
            joint=m._joint_past_forward(cur,ap,pp,qvalid)
            cached=m._joint_with_history_cache(cur,ap,pp,qvalid,kv)
            for key in ['action','past_action','video']:
                expected=joint[key][:,h:] if key=='video' else joint[key]
                torch.testing.assert_close(cached[key],expected,atol=1e-5,rtol=1e-4)
            va=m.action_expert.post_dit(joint['action'],ap)
            vl=m._past_predictions(joint['past_action']).masked_fill(~valid[...,None],0)
            vf=m.video_expert.post_dit(joint['video'],cur)
            action=m.infer_action_scheduler.step(va,da,action)
            past=m.infer_action_scheduler.step(vl,da,past).masked_fill(~valid[...,None],0)
            future=m.infer_video_scheduler.step(vf,dv,future)
        got,extra=m._denoise_action_with_condition(x['action'].clone(),steps,deltas,x['context'],x['context_mask'],
            freqs,kv,None,h,{},vp=vp,context=x['video_context'],context_mask=x['video_context_mask'],
            observation_frame_indices=frames,execution_start=boundary,
            generator=torch.Generator().manual_seed(42),rand_device='cpu')
        torch.testing.assert_close(got,action,atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(extra['future_video_prediction'],future,atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(extra['past_action_prediction'],past[0],atol=1e-5,rtol=1e-4)


def test_strict_checkpoint_migration_and_reload(tmp_path):
    old=model(asv=False);old.helios_config['data_fingerprint']='source'
    source=tmp_path/'old.pt';old.save_checkpoint(source,step=17500)
    new=model();new.helios_config['data_fingerprint']='new_asv'
    new.initialize_asv_weights(source,'source')
    for k,v in old.mot.state_dict().items():torch.testing.assert_close(v,new.mot.state_dict()[k],atol=0,rtol=0)
    with pytest.raises(ValueError,match='contract'):new.load_checkpoint(source)
    with pytest.raises(ValueError,match='contract'):new.initialize_asv_weights(source,'wrong')
    new.save_checkpoint(tmp_path/'asv.pt',step=2500)
    assert new.load_checkpoint(tmp_path/'asv.pt')==2500
    with pytest.raises(ValueError,match='contract'):old.load_checkpoint(tmp_path/'asv.pt')


def test_288_config_tokens_and_constant_initialization():
    p=memory_variant_profile('long75_less_temporal')
    validate_memory_variant('long75_less_temporal',p['history_sizes'],p['patch_long'],p['patch_mid'],2)
    m=model(asv=False,temporal=True)
    assert m.video_expert.patch_long(torch.ones(1,4,72,16,32)).shape==(1,32,36,2,4)
    torch.testing.assert_close(m.video_expert.patch_long(torch.ones(1,4,2,8,8)).flatten(),
                               m.video_expert.patch_embedding(torch.ones(1,4,1,2,2)).flatten())
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
        a=compose(config_name='train',overrides=['task=robomme_helios_push75_asv'])
        b=compose(config_name='train',overrides=['task=robomme_helios_push75_288'])
    assert a.max_steps==20000 and int(a.max_steps*a.warmup_ratio)==300 and a.learning_rate==2e-5
    assert b.max_steps==50000 and int(b.max_steps*b.warmup_ratio)==2500 and b.learning_rate==1e-4
    assert a.min_learning_rate==b.min_learning_rate==1e-6
    assert list(b.model.helios_config.patch_long)==[2,8,8]
    assert list(b.data.train.history_sizes)==[72,2,1]


def test_asv_learning_rate_warmup_and_absolute_floor():
    from fastwam.trainer import Wan22Trainer
    dummy=types.SimpleNamespace(learning_rate=2e-5,min_learning_rate=1e-6)
    parameter=torch.nn.Parameter(torch.zeros(1))
    dummy.optimizer=torch.optim.AdamW([parameter],lr=dummy.learning_rate)
    schedule=Wan22Trainer._build_scheduler(dummy,'cosine',20000,300)
    assert dummy.optimizer.param_groups[0]['lr']==pytest.approx(2e-5/300)
    for step in range(1,20001):
        dummy.optimizer.step();schedule.step()
        if step==300:assert dummy.optimizer.param_groups[0]['lr']==pytest.approx(2e-5)
    assert dummy.optimizer.param_groups[0]['lr']==pytest.approx(1e-6)
