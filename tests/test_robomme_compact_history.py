"""Fixed virtual slots, patch-boundary validity, and compact joint/cache parity."""
from copy import deepcopy
import math
import types

import pytest
import torch

from fastwam.memory.helios_padding import (
    apply_history_padding, history_padding_contract, validate_history_padding,
    zero_invalid_history,
)
from fastwam.memory.robomme_clock import PHASE_ANCHOR_MODE, decision_frames
from test_robomme_horizon32 import model, batch


def _fake_pre(sizes, kernels, valid, height=24, width=20, future=2, anchors=2):
    b = valid.shape[0]
    lengths = [math.ceil(n/k[0])*math.ceil(height/k[1])*math.ceil(width/k[2])
               for n, k in zip(sizes, kernels)]
    a = anchors * (height//2) * (width//2)
    p = future * (height//2) * (width//2)
    h = a + sum(lengths)
    s = h+p
    return dict(tokens=torch.arange(b*s*2.).reshape(b, s, 2),
                freqs=torch.arange(s*2.).reshape(s, 1, 2).to(torch.complex128),
                t_mod=torch.randn(b, s, 6, 2), t=torch.randn(b, p, 2),
                context=torch.randn(b, 3, 2), context_mask=torch.ones(b, s, 3, dtype=torch.bool),
                meta=dict(grid_size=(future, height//2, width//2),
                          mtm=dict(L_anchor=a, L_long=lengths[0], L_mid=lengths[1],
                                   L_curr=lengths[2], L_short=lengths[2], L_hist=h,
                                   L_pred=p, current_frame_index=sum(sizes)+1)))


@pytest.mark.parametrize('size', [19, 75, 99])
@pytest.mark.parametrize('long_kernel', [(4,8,8), (2,8,8), (4,4,4), (1,2,2)])
@pytest.mark.parametrize('observed', [1,2,3,4,6,7,74,75,76,99])
def test_patch_any_valid_right_alignment_and_rope(size, long_kernel, observed):
    sizes, kernels = (size-3, 2, 1), (long_kernel, (2,4,4), (1,2,2))
    observed = min(size, observed)
    valid = torch.arange(size)[None] >= size-observed
    before = _fake_pre(sizes, kernels, valid)
    pre = apply_history_padding(deepcopy(before), valid, sizes=sizes, kernels=kernels,
                                height=24, width=20, action_length=32, mode='compact')
    keep = pre['meta']['history_keep_indices']
    old, now = before['meta']['mtm'], pre['meta']['mtm']
    assert now['L_anchor'] == old['L_anchor'] == 240
    assert now['L_curr'] == now['L_short'] == 120
    assert now['L_mid'] == (30 if observed >= 2 else 0)
    long_frames = max(0, observed-3)
    assert now['L_long'] == math.ceil(long_frames/long_kernel[0]) * math.ceil(24/long_kernel[1]) * math.ceil(20/long_kernel[2])
    assert now['L_hist'] == sum(now[k] for k in ('L_anchor','L_long','L_mid','L_curr'))
    assert now['current_frame_index'] == size+1
    video_keep = torch.cat((keep, torch.arange(old['L_hist'], old['L_hist']+old['L_pred'])))
    torch.testing.assert_close(pre['freqs'], before['freqs'][video_keep], atol=0, rtol=0)
    torch.testing.assert_close(pre['t_mod'], before['t_mod'][:, video_keep], atol=0, rtol=0)
    for name in ('t', 'context'):
        torch.testing.assert_close(pre[name], before[name], atol=0, rtol=0)
    assert pre['meta']['key_valid'].all()  # Single-example compact has no batch pad.
    if observed == size:
        torch.testing.assert_close(pre['tokens'], before['tokens'], atol=0, rtol=0)


def test_batch_padding_and_nan_raw_history_are_isolated():
    sizes, kernels = (72,2,1), ((4,8,8),(2,4,4),(1,2,2))
    valid = torch.arange(75)[None] >= torch.tensor([74,68])[:, None]
    history = torch.randn(2,75,4,1,8,8)
    history[~valid] = float('nan')
    sanitized = zero_invalid_history(history, valid)
    assert torch.isfinite(sanitized).all() and sanitized[~valid].count_nonzero() == 0
    torch.testing.assert_close(sanitized[valid], history[valid], atol=0, rtol=0)
    pre = apply_history_padding(_fake_pre(sizes,kernels,valid), valid,
            sizes=sizes,kernels=kernels,height=24,width=20,action_length=32,mode='compact')
    hvalid = pre['meta']['history_key_valid']
    assert hvalid.shape == (2,399) and hvalid[1].all()
    assert hvalid[0].sum() == 360 and not hvalid[0,240:279].any()


def test_all_fixed_profiles_include_empty_segments_and_reverse_scales():
    from fastwam.memory.helios_history import MEMORY_PROFILES
    for variant, (sizes, long, mid, anchors, _) in MEMORY_PROFILES.items():
        if variant == 'full_history_dense':
            continue
        kernels = (long,mid,(1,2,2))
        valid = torch.arange(sum(sizes))[None] >= torch.tensor([sum(sizes)-1,0])[:,None]
        before = _fake_pre(sizes,kernels,valid,anchors=anchors)
        pre = apply_history_padding(deepcopy(before),valid,sizes=sizes,kernels=kernels,
                                   height=24,width=20,action_length=32,mode='compact')
        # One full prefix keeps the shared batch layout, while the startup row
        # masks every long/mid key regardless of the compression order.
        torch.testing.assert_close(pre['tokens'],before['tokens'],atol=0,rtol=0)
        hvalid = pre['meta']['history_key_valid']
        assert hvalid[1].all()
        assert hvalid[0].sum() == anchors*120+120, variant


def _pre(m, x, mode):
    m.helios_config.update(history_padding=mode, anchor_mode=PHASE_ANCHOR_MODE)
    b = x['action'].shape[0]
    vp = m._video_pre(x['anchor_latents'],x['history_latents'],x['input_latents'][:,:,1:],
                     torch.full((b,),500.), x['video_context'],x['video_context_mask'],
                     history_valid=x['history_valid'])
    return m._apply_anchor_validity(vp,x['anchor_valid'],32)


def _joint(m,x,vp):
    b = len(x['action'])
    t = torch.full((b,),700.)
    ap = m.action_expert.pre_dit(x['action'],t,x['context'],x['context_mask'])
    ap['freqs'] = m._action_freqs(32,vp)
    pp, valid = m._flow_pre(x,vp,x['video_context'],x['video_context_mask'],x['past_action_target'],t)
    return m._joint_past_forward(vp,ap,pp,valid), ap, pp, valid


def _sample(m):
    x = batch(m)
    x['history_valid'] = torch.arange(99)[None] >= torch.tensor([92,98])[:,None]
    x['anchor_valid'][1,0] = False
    return x


@pytest.mark.parametrize('future',[False,True])
def test_masked_and_compact_joint_outputs_and_real_gradients(future):
    m = model(future).eval()
    x = _sample(m)
    mask = _pre(m,x,'mask')
    out,_,_,_ = _joint(m,x,mask)
    compact = _pre(m,x,'compact')
    small,_,_,_ = _joint(m,x,compact)
    for name in ('action','past_action'):
        torch.testing.assert_close(small[name],out[name],atol=5e-6,rtol=5e-5)
    old_h, new_h = mask['meta']['mtm']['L_hist'], compact['meta']['mtm']['L_hist']
    torch.testing.assert_close(small['video'][:,new_h:],out['video'][:,old_h:],atol=5e-6,rtol=5e-5)
    # Every branch has the same forbidden history keys, including P and Q.
    attn = m._three_expert_mask(compact,32,x['past_action_valid'])
    hvalid = compact['meta']['history_key_valid']
    assert not attn[:,:,:new_h].masked_select((~hvalid)[:,None].expand(-1,attn.shape[1],-1)).any()
    runs = []
    for mode in ('mask','compact'):
        clone = deepcopy(m).train()
        clone.helios_config['history_padding'] = mode
        clone.build_inputs = types.MethodType(lambda self,sample,tiled=False:sample,clone)
        torch.manual_seed(998)
        loss,_ = clone.training_loss(x)
        loss.backward()
        runs.append((loss.detach(), {name:p.grad for name,p in clone.named_parameters() if p.requires_grad}))
    torch.testing.assert_close(runs[0][0],runs[1][0],atol=5e-6,rtol=5e-5)
    for name, grad in runs[0][1].items():
        other = runs[1][1][name]
        if grad is None:
            assert other is None
        else:
            assert torch.isfinite(grad).all() and torch.isfinite(other).all()
            torch.testing.assert_close(grad,other,atol=8e-6,rtol=2e-4,msg=lambda msg:f'{name}: {msg}')


@pytest.mark.parametrize('future',[False,True])
def test_compact_single_vs_mixed_batch_and_prefill(future):
    m = model(future).eval()
    x = _sample(m)
    x['history_valid'][1,-80:] = True
    full = _pre(m,x,'compact')
    out,_,_,_ = _joint(m,x,full)
    one = {k:v[:1] for k,v in x.items()}
    vp = _pre(m,one,'compact')
    solo,ap,pp,valid = _joint(m,one,vp)
    assert vp['meta']['mtm']['L_hist'] < full['meta']['mtm']['L_hist']
    for name in ('action','past_action'):
        torch.testing.assert_close(solo[name],out[name][:1],atol=5e-6,rtol=5e-5)
    kv,hvalid,_ = m._prefill_inference_condition(vp,one['video_context'],one['video_context_mask'],
                                               decision_frames(82,49),49)
    combined,key_valid,cached = m.mot.prefill_past_action(pp,kv,valid)
    torch.testing.assert_close(cached,solo['past_action'],atol=5e-6,rtol=5e-5)
    amask = torch.cat((key_valid[0],torch.ones(32,dtype=torch.bool)))[None].expand(32,-1)
    velocity = m._predict_action_noise_with_cache(one['action'],torch.tensor([700.]),
                   one['context'],one['context_mask'],ap['freqs'],combined,amask,
                   kv[0]['k'].shape[1]+m.query_slots)
    torch.testing.assert_close(velocity,m.action_expert.post_dit(solo['action'],ap),atol=5e-6,rtol=5e-5)


@pytest.mark.parametrize('mode',['mask','compact'])
def test_invalid_raw_padding_cannot_change_even_mixed_patches(mode):
    m = model(True).eval()
    x = _sample(m)
    a = _pre(m,x,mode)
    x['history_latents'][~x['history_valid']] = float('nan')
    b = _pre(m,x,mode)
    for name in ('tokens','freqs','t_mod'):
        torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)


@pytest.mark.parametrize('future',[False,True])
def test_actual_online_inference_matches_masked_fixed_grid(future):
    from fastwam.models.wan22.helios_video import MTMVideoDiT
    from fastwam.memory.robomme_clock import CONTRACT
    m = model(future).eval()
    m.helios_config.update(anchor_mode=PHASE_ANCHOR_MODE,observation_contract=CONTRACT)
    m.mot.mixtures['video'] = MTMVideoDiT(
        hidden_dim=32,ffn_dim=64,text_dim=16,freq_dim=16,eps=1e-6,
        num_heads=2,attn_head_dim=24,num_layers=2,in_dim=48,out_dim=48,
        patch_size=(1,2,2),has_image_input=False,seperated_timestep=True,
        multi_term_memory=True,mtm_history_sizes=(96,2,1),mtm_anchor_size=2,mtm_pred_size=2)
    frames = [0]  # No demo, no observed transition, only true execution anchor/current.
    kwargs = dict(prompt=None,input_image=None,action_horizon=32,
                  observation_latents=torch.randn(1,48,1,24,20),
                  observation_frame_indices=frames,execution_start=0,proprio=torch.zeros(8),
                  context=torch.randn(3,16),context_mask=torch.ones(3,dtype=torch.bool),
                  full_kv_frame_index=0,num_inference_steps=2,seed=42)
    m.helios_config['history_padding'] = 'mask'
    masked = m.infer_action(**kwargs)
    m.helios_config['history_padding'] = 'compact'
    compact = m.infer_action(**kwargs)
    assert masked['full_kv_video_tokens'] == 606
    assert compact['full_kv_video_tokens'] == 360
    for key in ('action','past_action_prediction'):
        torch.testing.assert_close(compact[key],masked[key],atol=6e-6,rtol=6e-5)
    assert torch.isfinite(compact['action']).all()


def test_legacy_contract_unchanged_and_incompatible_checkpoint_rejected(tmp_path):
    assert history_padding_contract('legacy_zero') == {}
    m = model()
    old = m._helios_inference_contract()
    path = tmp_path/'legacy.pt'
    m.save_checkpoint(path)
    assert m.load_checkpoint(path) is None
    assert 'history_padding' not in old
    m.helios_config['history_padding'] = 'compact'
    assert m._helios_inference_contract()['history_padding'] == 'compact'
    with pytest.raises(ValueError,match='contract'):
        m.load_checkpoint(path)
    with pytest.raises(ValueError,match='full_history_dense'):
        validate_history_padding('compact','full_history_dense')
    with pytest.raises(ValueError,match='Unknown'):
        validate_history_padding('bad')
    with pytest.raises(ValueError,match='current'):
        zero_invalid_history(torch.zeros(1,75,4,1,8,8),torch.zeros(1,75,dtype=torch.bool))
