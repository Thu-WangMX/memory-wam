"""Ordered label coverage, flow corruption, and joint/cached solver parity."""
from copy import deepcopy
import types

import numpy as np
import pytest
import torch

from fastwam.memory.past_action import DreamDojoLabels, past_action_interval_layout
from fastwam.memory.robomme_clock import decision_frames
from test_robomme_past_action import make_model, batch, forward


def video_cache(m, vp):
    h = vp['meta']['mtm']['L_hist']
    return m.mot.prefill_video_cache(
        video_tokens=vp['tokens'][:, :h], video_freqs=vp['freqs'][:h],
        video_t_mod=vp['t_mod'][:, :h],
        video_context_payload=dict(context=vp['context'], mask=vp['context_mask'][:, :h]),
        video_attention_mask=torch.ones(h, h, dtype=torch.bool))


@pytest.mark.parametrize('boundary', [0, 1, 2, 16, 17, 18, 33, 35, 49, 1100])
def test_every_observed_transition_once_in_order(boundary):
    for current in [boundary, boundary+16, boundary+32, boundary+1600]:
        frames = decision_frames(current+1, boundary)
        x = past_action_interval_layout(frames, boundary)
        present = x['past_action_transition_frames'] >= 0
        f = x['past_action_transition_frames'][x['past_action_transition_valid']].tolist()
        start = frames[-100] if len(frames) > 99 else 0
        assert f == [i for i in range(start+1, current+1) if i != boundary]
        assert x['past_action_valid'].equal(x['past_action_transition_valid'].any(-1))
        assert (x['past_action_interval_length'] == present.sum(-1)).all()
        assert (x['past_action_transition_frames'][present] <= current).all()


def test_ordered_labels_short_interval_boundary_and_mask():
    x = past_action_interval_layout(decision_frames(52, 35), 35)
    assert x['past_action_frame_indices'][-6:].tolist() == [0, 16, 32, 34, 35, 51]
    assert x['past_action_interval_length'][-6:].tolist() == [0, 16, 16, 2, 1, 16]
    assert x['past_action_transition_frames'][-3, :4].tolist() == [33, 34, -1, -1]
    labels = DreamDojoLabels.__new__(DreamDojoLabels)
    labels.offsets = {0: (0, 52)}
    labels._latents = np.arange(52, dtype=np.float32)[:, None].repeat(64, 1)
    labels.valid = np.ones(52, dtype=bool)
    labels.valid[[0, 35]] = False
    labels.phase = np.where(np.arange(52) < 35, 1, 3)
    labels.phase[35] = 2
    labels.mean, labels.std = np.zeros(64), np.ones(64)*2
    y = labels.targets(0, x)
    assert y.shape == (99, 16, 64)
    torch.testing.assert_close(y[-3, :2, 0], torch.tensor([16.5, 17.], dtype=y.dtype))
    assert torch.count_nonzero(y[~x['past_action_transition_valid']]) == 0
    assert y[-1, :, 0].tolist() == [i/2 for i in range(36, 52)]


def test_interval_loss_supervises_interior_not_padding():
    m = make_model('block16')
    x = batch(m)
    out, _, _ = forward(m, x)
    base, _ = m._extra_training_loss(out, x)
    changed = deepcopy(x)
    changed['past_action_target'][~x['past_action_transition_valid']] += 9999
    masked, _ = m._extra_training_loss(out, changed)
    torch.testing.assert_close(base, masked, atol=0, rtol=0)
    changed['past_action_target'][0, -1, 5] += 9999
    middle, _ = m._extra_training_loss(out, changed)
    assert middle > base


@pytest.mark.parametrize('mode', ['flow', 'sparse'])
def test_flow_corruption_direction_padding_and_time(monkeypatch, mode):
    m = make_model(mode).eval()
    x = batch(m)
    calls = []
    original = m._flow_pre
    def capture(self, sample, vp, context, context_mask, noisy, t):
        calls.append((noisy.detach().clone(), t.clone()))
        return original(sample, vp, context, context_mask, noisy, t)
    m._flow_pre = types.MethodType(capture, m)
    monkeypatch.setattr(torch, 'randn_like', lambda x: torch.ones_like(x)*2)
    out, _, _ = forward(m, x)
    valid = x['past_action_transition_valid'][..., None]
    clean = x['past_action_target'].masked_fill(~valid, 0)
    noise = torch.ones_like(clean).mul(2).masked_fill(~valid, 0)
    torch.testing.assert_close(calls[0][0], clean*.3+noise*.7)
    torch.testing.assert_close(out['past_action_flow_target'], noise-clean)
    assert (calls[0][1] == 700).all()
    # With sigma=1 the input is pure noise, independently of the label.
    inputs = dict(x, action_flow_timestep=torch.full((2,), 1000.))
    _, vp, ap = forward(m, x)
    a = m._training_expert_forward(vp, ap, inputs, x)
    y = deepcopy(x)
    y['past_action_target'].add_(1000)
    b = m._training_expert_forward(vp, ap, inputs, y)
    torch.testing.assert_close(a['action'], b['action'], atol=0, rtol=0)
    torch.testing.assert_close(a['past_action'], b['past_action'], atol=0, rtol=0)


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('mode', ['flow', 'sparse'])
def test_flow_backward_isolation_and_empty_loss(monkeypatch, checkpoint, mode):
    monkeypatch.setenv('FASTWAM_CHECKPOINT_WHOLE_LAYER', str(int(checkpoint)))
    m = make_model(mode).train()
    m.mot.mot_checkpoint_mixed_attn = checkpoint
    x = batch(m)
    m.build_inputs = types.MethodType(lambda self, sample, tiled=False: sample, m)
    loss, metrics = m.training_loss(x)
    loss.backward()
    for p in (m.past_action_expert.query, m.past_action_expert.head.weight,
              m.past_action_expert.noisy_latent_embedding.weight,
              m.past_action_expert.interval_length_embedding.weight,
              m.past_action_expert.time_projection[1].weight,
              m.video_expert.patch_long.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert 'past_action_velocity_mse_demo' in metrics
    m.eval()
    torch.manual_seed(111)
    a, _, _ = forward(m, x)
    y = deepcopy(x)
    y['input_latents'][:, :, 1:].add_(999)
    y['action'].add_(999)
    y['context'].add_(999)
    torch.manual_seed(111)
    b, _, _ = forward(m, y)
    torch.testing.assert_close(a['past_action'], b['past_action'])
    y['past_action_transition_valid'].zero_()
    y['past_action_valid'].zero_()
    if 'past_action_visual_slots' in y:
        y['past_action_visual_slots'].fill_(-1)
    out, _, _ = forward(m, y)
    loss, _ = m._extra_training_loss(out, y)
    assert loss.item() == 0 and torch.isfinite(loss)


@pytest.mark.parametrize('mode', ['flow', 'sparse'])
def test_joint_noisy_forward_matches_step_cache_and_solver(mode):
    m = make_model(mode).eval()
    x = {k: v[:1] for k, v in batch(m).items()}
    with torch.no_grad():
        vp = m._video_pre(x['anchor_latents'], x['history_latents'], x['input_latents'][:, :, 1:],
                          torch.tensor([500.]), x['video_context'], x['video_context_mask'])
        cache = video_cache(m, vp)
        h = vp['meta']['mtm']['L_hist']
        action = x['action'].clone()
        gen = torch.Generator().manual_seed(121)
        past = torch.randn((1, m.query_slots, m.transition_slots, 64), generator=gen)
        valid = x['past_action_transition_valid'][..., None]
        past.masked_fill_(~valid, 0)
        steps, deltas = m.infer_action_scheduler.build_inference_schedule(3, m.device, torch.float32)
        freqs = m._action_freqs(16, vp)
        for t, delta in zip(steps, deltas):
            ap = m.action_expert.pre_dit(action, t[None], x['context'], x['context_mask'])
            ap['freqs'] = freqs
            pp, qvalid = m._flow_pre(x, vp, x['video_context'], x['video_context_mask'], past, t[None])
            joint = m._joint_past_forward(vp, ap, pp, qvalid)
            combined, key_valid, p_cached = m.mot.prefill_past_action(pp, cache, qvalid)
            torch.testing.assert_close(joint['past_action'], p_cached, atol=3e-6, rtol=3e-5)
            mask = torch.cat((key_valid[0], torch.ones(16, dtype=torch.bool)))[None].expand(16, -1)
            a_cached = m._predict_action_noise_with_cache(
                action, t[None], x['context'], x['context_mask'], freqs, combined, mask, h+m.query_slots)
            pa = m.action_expert.post_dit(joint['action'], ap)
            torch.testing.assert_close(pa, a_cached, atol=3e-6, rtol=3e-5)
            pv = m._past_predictions(joint['past_action']).masked_fill(~valid, 0)
            action = m.infer_action_scheduler.step(pa, delta, action)
            past = m.infer_action_scheduler.step(pv, delta, past).masked_fill(~valid, 0)
        got, extra = m._denoise_action_with_condition(
            x['action'].clone(), steps, deltas, x['context'], x['context_mask'], freqs, cache,
            torch.ones(16, h+16, dtype=torch.bool), h, {}, vp=vp,
            context=x['video_context'], context_mask=x['video_context_mask'],
            observation_frame_indices=decision_frames(82, 49), execution_start=49,
            generator=torch.Generator().manual_seed(121), rand_device='cpu')
        torch.testing.assert_close(got, action, atol=5e-6, rtol=5e-5)
        torch.testing.assert_close(extra['past_action_prediction'], past[0], atol=5e-6, rtol=5e-5)
        assert extra['past_action_flow_steps'] == 3
        assert torch.count_nonzero(extra['past_action_prediction'][~valid[0, ..., 0]]) == 0


@pytest.mark.parametrize('mode', ['flow', 'sparse'])
def test_flow_checkpoint_roundtrip_and_no_deterministic_resume(tmp_path, mode):
    m = make_model(mode)
    path = tmp_path/'flow.pt'
    m.save_checkpoint(path, step=12)
    reference = m.past_action_expert.noisy_latent_embedding.weight.clone()
    with torch.no_grad():
        m.past_action_expert.noisy_latent_embedding.weight.zero_()
    assert m.load_checkpoint(path) == 12
    torch.testing.assert_close(reference, m.past_action_expert.noisy_latent_embedding.weight)
    with pytest.raises(ValueError, match='contract'):
        make_model('block16').load_checkpoint(path)
    with pytest.raises(ValueError, match='contract'):
        make_model('flow' if mode == 'sparse' else 'sparse').load_checkpoint(path)
