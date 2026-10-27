import types
import pytest
import torch
from torch.utils.data import default_collate
from fastwam.models.wan22.helios_video import MTMVideoDiT
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.fastwam_helios import FastWAMHelios
from fastwam.memory.helios_history import episode_anchors, select_history, validate_memory_variant, memory_variant_profile
from fastwam.datasets.lerobot.helios_dataset import collate_full_history


def make_model(variant):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    profile = memory_variant_profile(variant)
    sizes = profile['history_sizes']
    patches = (profile['patch_long'], profile['patch_mid'])
    anchors = profile['anchor_frames']
    common = dict(hidden_dim=32, ffn_dim=64, text_dim=16, freq_dim=16,
                  eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=2)
    v = MTMVideoDiT(**common, in_dim=4, out_dim=4, patch_size=(1,2,2),
        has_image_input=False, seperated_timestep=True, multi_term_memory=True,
        mtm_history_sizes=sizes, mtm_anchor_size=anchors, mtm_pred_size=1,
        mtm_full_history=variant=='full_history_dense',
        mtm_history_position_offset=profile['history_position_offset'],
        mtm_patch_kernel_long=patches[0], mtm_patch_kernel_mid=patches[1])
    a = ActionDiT(**common,action_dim=14)
    m = FastWAMHelios(v,a,MoT({'video':v,'action':a},mot_checkpoint_mixed_attn=False),
        torch.nn.Identity(),text_dim=16,proprio_dim=14,action_train_shift=1.,action_infer_shift=1.)
    m.helios_config=dict(history_sizes=sizes,anchor_frames=anchors,patch_long=patches[0],
                        patch_mid=patches[1],condition_augmentation=False,memory_variant=variant)
    m.freeze_unused_history_branches()
    m.vae.temporal_downsample_factor=4
    return m


def sample(m, n, hw=8):
    prefix=torch.randn(n,4,1,hw,hw)
    anchors,av=episode_anchors(prefix,m.helios_config['anchor_frames'])
    history,hv=select_history(prefix,m.helios_config['history_sizes'],m.helios_config['memory_variant'])
    ctx=torch.randn(3,16)
    mask=torch.ones(3,dtype=torch.bool)
    return dict(anchor_latents=anchors,anchor_valid=av,history_latents=history,history_valid=hv,
        input_latents=torch.randn(4,2,hw,hw),action=torch.randn(16,14),
        context=ctx,context_mask=mask,video_context=ctx,video_context_mask=mask,
        image_is_pad=torch.zeros(5,dtype=torch.bool),action_is_pad=torch.zeros(16,dtype=torch.bool))


def joint(m,x):
    b=len(x['action'])
    vp=m._video_pre(x['anchor_latents'],x['history_latents'],x['input_latents'][:,:,1:],
        torch.full((b,),500.),x['context'],x['context_mask'])
    vp=m._apply_history_batch_padding(vp,x,16)
    ap=m.action_expert.pre_dit(x['action'],torch.full((b,),700.),x['context'],x['context_mask'])
    out=m.mot(embeds_all={'video':vp['tokens'],'action':ap['tokens']},
        attention_mask=m._joint_mask(vp,16),freqs_all={'video':vp['freqs'],'action':m._action_freqs(16,vp)},
        context_all={'video':{'context':vp['context'],'mask':vp['context_mask']},
                     'action':{'context':ap['context'],'mask':ap['context_mask']}},
        t_mod_all={'video':vp['t_mod'],'action':ap['t_mod']})
    return m.action_expert.post_dit(out['action'],ap),m.video_expert.post_dit(out['video'],vp),vp


@pytest.mark.parametrize('variant,frames,count',[
    ('anchor_only',45,360),('recent19_dense',45,2520),('full_history_dense',45,5640),
    ('multiscale',45,426),('wo_long',45,390),('uniform_mid',45,630),
    ('reverse_scale',45,426),('wo_anchor',45,186),
    ('long_less_temporal',45,462),('long_less_spatial',45,510)])
def test_token_counts_and_temporal_positions(variant,frames,count):
    m=make_model(variant)
    s=sample(m,frames,hw=24)
    # Production latent spatial shape 24x20.
    for key in ['anchor_latents','history_latents','input_latents']:s[key]=s[key][...,:20]
    x=default_collate([s])
    p=m._video_pre(x['anchor_latents'],x['history_latents'],x['input_latents'][:,:,1:],
        torch.zeros(1),x['context'],x['context_mask'])
    assert p['meta']['mtm']['L_hist']==count
    assert p['meta']['mtm']['current_frame_index']==(46 if variant=='full_history_dense' else 20)


@pytest.mark.parametrize('chunk',[0,23])
def test_batch4_padding_rope_and_joint_prefill_parity(chunk,monkeypatch):
    monkeypatch.setenv('FASTWAM_QKV_CHUNK_SIZE',str(chunk))
    m=make_model('full_history_dense').eval()
    samples=[sample(m,n) for n in [1,19,23,31]]
    batch=collate_full_history(samples)
    assert batch['history_batch_pad'].sum(1).tolist()==[12,12,8,0]
    with torch.no_grad():
        actions,videos,vp=joint(m,batch)
        for i,s in enumerate(samples):
            one=default_collate([s])
            ar,vr,pr=joint(m,one)
            torch.testing.assert_close(actions[i:i+1],ar,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(videos[i:i+1],vr,atol=2e-6,rtol=2e-5)
            h=pr['meta']['mtm']['L_hist']
            kv=m.mot.prefill_video_cache(video_tokens=pr['tokens'][:,:h],video_freqs=pr['freqs'][:h],
                video_t_mod=pr['t_mod'][:,:h],video_context_payload={'context':pr['context'],'mask':pr['context_mask'][:,:h]},
                video_attention_mask=torch.ones(h,h,dtype=torch.bool))
            cached=m._predict_action_noise_with_cache(one['action'],torch.full((1,),700.),
                one['context'],one['context_mask'],m._action_freqs(16,pr),kv,
                torch.ones(16,h+16,dtype=torch.bool),h)
            torch.testing.assert_close(ar,cached,atol=2e-6,rtol=2e-5)
        dirty=dict(batch)
        dirty['history_latents']=batch['history_latents'].clone()
        dirty['history_latents'][batch['history_batch_pad']]=1000
        da,dv,_=joint(m,dirty)
        torch.testing.assert_close(actions,da,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(videos,dv,atol=2e-6,rtol=2e-5)
        future=dict(batch);future['input_latents']=batch['input_latents']*100
        fa,_,_=joint(m,future)
        torch.testing.assert_close(actions,fa,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('variant',['anchor_only','recent19_dense','full_history_dense',
    'wo_long','uniform_mid','reverse_scale','wo_anchor','long_less_temporal','long_less_spatial'])
def test_loss_backward_online_selection_and_checkpoint(variant,tmp_path):
    m=make_model(variant)
    samples=[sample(m,n) for n in [1,20,23,27]]
    batch=collate_full_history(samples) if variant=='full_history_dense' else default_collate(samples)
    m.build_inputs=types.MethodType(lambda self,x,tiled=False:x,m)
    m.helios_config['condition_augmentation']=True
    loss,metrics=m.training_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.video_expert.patch_embedding.weight.grad.abs().sum()>0
    if m.helios_config['history_sizes'][0]:
        assert m.video_expert.patch_long.weight.grad.abs().sum()>0
    else:
        assert m.video_expert.patch_long.weight.grad is None
    path=tmp_path/'weights.pt'
    m.save_checkpoint(path,step=71)
    assert m.load_checkpoint(path)==71
    prefix=torch.randn(1,4,27,8,8)
    m._encode_input_image_latents_tensor=types.MethodType(lambda self,image,tiled=False:prefix,m)
    x=default_collate([samples[-1]])
    result=m.infer_action(prompt=None,input_image=torch.zeros(1),action_horizon=16,
        proprio=torch.zeros(14),context=x['context'],context_mask=x['context_mask'],
        full_kv_frame_index=26,num_inference_steps=2,seed=5)
    expected={'anchor_only':48,'recent19_dense':336,'full_history_dense':464,
        'wo_long':52,'uniform_mid':84,'reverse_scale':56,'wo_anchor':24,'long_less_temporal':60,'long_less_spatial':68}[variant]
    assert result['full_kv_video_tokens']==expected and torch.isfinite(result['action']).all()
    m.helios_config['memory_variant']='wrong'
    with pytest.raises(ValueError,match='contract'):m.load_checkpoint(path)


def test_full_history_preserves_old_frames_and_validation():
    h=torch.arange(45.).reshape(45,1,1,1,1)
    full,_=select_history(h,variant='full_history_dense')
    recent,_=select_history(h,variant='recent19_dense')
    assert torch.equal(full,h) and torch.equal(recent,h[-19:])
    with pytest.raises(ValueError):validate_memory_variant('recent19_dense',[16,2,1],[4,8,8],[2,4,4])


def test_whole_layer_checkpoint_forward_and_gradients(monkeypatch):
    m=make_model('full_history_dense').train()
    batch=collate_full_history([sample(m,n) for n in [1,19,23,27]])
    monkeypatch.setenv('FASTWAM_QKV_CHUNK_SIZE','23')
    monkeypatch.setenv('FASTWAM_FFN_CHUNK_SIZE','23')
    monkeypatch.setenv('FASTWAM_CHECKPOINT_WHOLE_LAYER','0')
    a,v,_=joint(m,batch)
    (a.square().mean()+v.square().mean()).backward()
    grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    monkeypatch.setenv('FASTWAM_CHECKPOINT_WHOLE_LAYER','1')
    ac,vc,_=joint(m,batch)
    torch.testing.assert_close(ac,a)
    torch.testing.assert_close(vc,v)
    (ac.square().mean()+vc.square().mean()).backward()
    for n,p in m.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=2e-6,rtol=2e-4)


LAYOUT_VARIANTS = ['wo_long','uniform_mid','reverse_scale','wo_anchor','long_less_temporal','long_less_spatial']

@pytest.mark.parametrize('variant',LAYOUT_VARIANTS)
@pytest.mark.parametrize('frames',[1,2,3,18,19,27])
def test_new_layout_online_history_matches_training(variant,frames):
    m=make_model(variant).eval()
    prefix=torch.randn(frames,4,1,8,8)
    train_history,_=select_history(prefix,m.helios_config['history_sizes'],variant)
    train_anchors,_=episode_anchors(prefix,m.helios_config['anchor_frames'])
    observed=prefix.squeeze(2).permute(1,0,2,3).unsqueeze(0)
    m._encode_input_image_latents_tensor=types.MethodType(lambda self,image,tiled=False:observed,m)
    ctx=torch.randn(1,3,16);mask=torch.ones(1,3,dtype=torch.bool)
    captured=[];original=m._video_pre
    def capture(self,anchors,history,*args,**kwargs):
        captured.append((anchors.clone(),history.clone()))
        return original(anchors,history,*args,**kwargs)
    m._video_pre=types.MethodType(capture,m)
    result=m.infer_action(prompt=None,input_image=torch.zeros(1),action_horizon=16,
        proprio=torch.zeros(14),context=ctx,context_mask=mask,
        full_kv_frame_index=frames-1,num_inference_steps=2,seed=5)
    assert torch.isfinite(result['action']).all()
    assert captured
    torch.testing.assert_close(captured[0][0],train_anchors.unsqueeze(0))
    torch.testing.assert_close(captured[0][1],train_history.unsqueeze(0))

@pytest.mark.parametrize('variant',LAYOUT_VARIANTS)
def test_new_layout_mask_and_cached_action_parity(variant):
    m=make_model(variant).eval();batch=default_collate([sample(m,27)])
    with torch.no_grad():
        action,_,pre=joint(m,batch)
        h=pre['meta']['mtm']['L_hist'];p=pre['meta']['mtm']['L_pred']
        mask=m._joint_mask(pre,16)
        assert mask[:h,:h].all() and not mask[:h,h:].any()
        assert mask[h:h+p,:h+p].all() and not mask[h:h+p,h+p:].any()
        assert mask[h+p:,:h].all() and not mask[h+p:,h:h+p].any()
        assert mask[h+p:,h+p:].all()
        kv=m.mot.prefill_video_cache(video_tokens=pre['tokens'][:,:h],video_freqs=pre['freqs'][:h],
            video_t_mod=pre['t_mod'][:,:h],video_context_payload={'context':pre['context'],'mask':pre['context_mask'][:,:h]},
            video_attention_mask=torch.ones(h,h,dtype=torch.bool))
        cached=m._predict_action_noise_with_cache(batch['action'],torch.full((1,),700.),
            batch['context'],batch['context_mask'],m._action_freqs(16,pre),kv,
            torch.ones(16,h+16,dtype=torch.bool),h)
        torch.testing.assert_close(action,cached,atol=2e-6,rtol=2e-5)
        changed=dict(batch);changed['input_latents']=batch['input_latents']*100
        alt,_,_=joint(m,changed)
        torch.testing.assert_close(action,alt,atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize('variant,removed',[('wo_long',36),('wo_anchor',240)])
def test_removing_memory_does_not_shift_remaining_rope(variant,removed):
    base=make_model('multiscale');other=make_model(variant)
    def get(m):
        s=sample(m,27,hw=24)
        for key in ['anchor_latents','history_latents','input_latents']:s[key]=s[key][...,:20]
        return joint(m,default_collate([s]))[2]
    with torch.no_grad():a=get(base);b=get(other)
    freqs=a['freqs']
    expected=torch.cat([freqs[:240],freqs[240+removed:]]) if variant=='wo_long' else freqs[removed:]
    torch.testing.assert_close(b['freqs'],expected)
    torch.testing.assert_close(base._action_freqs(16,a),other._action_freqs(16,b))

@pytest.mark.parametrize('variant',LAYOUT_VARIANTS)
def test_new_layout_contract_rejects_wrong_anchors_or_patches(variant):
    p=memory_variant_profile(variant)
    validate_memory_variant(variant,p['history_sizes'],p['patch_long'],p['patch_mid'],p['anchor_frames'])
    with pytest.raises(ValueError):
        validate_memory_variant(variant,p['history_sizes'],p['patch_long'],p['patch_mid'],2-p['anchor_frames'])
    with pytest.raises(ValueError):
        validate_memory_variant(variant,p['history_sizes'],[1,2,2],p['patch_mid'],p['anchor_frames'])


@pytest.mark.parametrize('variant',LAYOUT_VARIANTS)
def test_new_checkpoint_restores_a_fresh_model(variant,tmp_path):
    original=make_model(variant).eval()
    path=tmp_path/'layout.pt';original.save_checkpoint(path,step=123)
    saved=torch.load(path,map_location='cpu',weights_only=False)['helios_contract']
    fresh=make_model(saved['memory_variant']).eval()
    assert fresh.helios_config==saved
    assert fresh.load_checkpoint(path)==123
    batch=default_collate([sample(original,27)])
    with torch.no_grad():
        expected_action,expected_video,_=joint(original,batch)
        actual_action,actual_video,_=joint(fresh,batch)
    torch.testing.assert_close(actual_action,expected_action)
    torch.testing.assert_close(actual_video,expected_video)


def test_removed_long_and_fixed_anchors_do_not_leak_back_in():
    prefix=torch.arange(27.).reshape(27,1,1,1,1)
    for variant in ('wo_long','wo_anchor'):
        profile=memory_variant_profile(variant)
        before,_=select_history(prefix,profile['history_sizes'],variant)
        edited=prefix.clone()
        if variant=='wo_long':edited[-19:-3]=999
        else:edited[:2]=999
        after,_=select_history(edited,profile['history_sizes'],variant)
        assert torch.equal(before,after)


def test_reverse_2_16_uses_same_window_and_token_budget():
    m=make_model('reverse_scale')
    prefix=torch.arange(27.).reshape(27,1,1,1,1)
    selected,_=select_history(prefix,m.helios_config['history_sizes'],'reverse_scale')
    long,mid,current=m._segments(selected.unsqueeze(0))
    assert long.flatten().tolist()==[8.,9.]
    assert mid.flatten().tolist()==list(map(float,range(10,26)))
    assert current.item()==26
    assert long.shape[2]%m.video_expert.mtm_patch_kernel_long[0]==0
    assert mid.shape[2]%m.video_expert.mtm_patch_kernel_mid[0]==0
    with pytest.raises(ValueError,match='fixed memory profile'):
        validate_memory_variant('reverse_scale',[16,2,1],[2,4,4],[4,8,8],2)
    s=sample(m,27,hw=24)
    for key in ['anchor_latents','history_latents','input_latents']:s[key]=s[key][...,:20]
    with torch.no_grad():
        pre=joint(m,default_collate([s]))[2]
    meta=pre['meta']['mtm']
    assert (meta['L_anchor'],meta['L_long'],meta['L_mid'],meta['L_curr'])==(240,30,36,120)
    assert meta['current_frame_index']==20


@pytest.mark.parametrize('variant,long_count',[
    ('long_less_temporal',72),('long_less_spatial',120)])
def test_less_long_compression_preserves_window_and_other_rope(variant,long_count):
    base=make_model('multiscale');other=make_model(variant)
    prefix=torch.arange(27.).reshape(27,1,1,1,1)
    for m in (base,other):
        selected,_=select_history(prefix,m.helios_config['history_sizes'],m.helios_config['memory_variant'])
        long,mid,current=m._segments(selected.unsqueeze(0))
        assert long.flatten().tolist()==list(map(float,range(8,24)))
        assert mid.flatten().tolist()==[24.,25.] and current.item()==26
    def pre(m):
        s=sample(m,27,hw=24)
        for key in ['anchor_latents','history_latents','input_latents']:s[key]=s[key][...,:20]
        return joint(m,default_collate([s]))[2]
    with torch.no_grad():a=pre(base);b=pre(other)
    meta=b['meta']['mtm']
    assert (meta['L_anchor'],meta['L_long'],meta['L_mid'],meta['L_curr'])==(240,long_count,30,120)
    torch.testing.assert_close(a['freqs'][:240],b['freqs'][:240])
    torch.testing.assert_close(a['freqs'][240+36:],b['freqs'][240+long_count:])
    torch.testing.assert_close(base._action_freqs(16,a),other._action_freqs(16,b))
