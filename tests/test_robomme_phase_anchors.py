"""Phase anchor identity, padding isolation, cached/joint parity and old contracts."""
from copy import deepcopy
from pathlib import Path
import json
import types

import numpy as np
import pytest
import torch

from fastwam.memory.helios_history import episode_anchors
from fastwam.memory.robomme_clock import (
    CONTRACT, LEGACY_ANCHOR_MODE, PHASE_ANCHOR_MODE, decision_frames,
    inference_contract, select_robomme_anchors,
)
from test_robomme_horizon32 import model, batch, pre


@pytest.mark.parametrize('boundary', [0, 1, 17, 35, 49, 1100])
def test_both_anchors_stay_fixed_beyond_window(boundary):
    first = None
    for steps in [0, 1, 3, 120]:
        frames = decision_frames(boundary + steps*16 + 1, boundary)
        history = torch.tensor(frames, dtype=torch.float32).view(-1, 1, 1, 1, 1) + 1
        anchors, valid = select_robomme_anchors(history, frames, boundary, mode=PHASE_ANCHOR_MODE)
        assert valid.tolist() == [boundary > 0, True]
        assert anchors.flatten().tolist() == [1. if boundary else 0., boundary+1.]
        if first is not None:
            torch.testing.assert_close(anchors, first, atol=0, rtol=0)
        first = anchors
        legacy = select_robomme_anchors(history)
        for actual, expected in zip(legacy, episode_anchors(history, 2)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_demo_only_does_not_use_future_execution_and_requires_full_clock():
    history = torch.arange(3.).view(-1, 1, 1, 1, 1) + 1
    anchors, valid = select_robomme_anchors(history, [0, 16, 32], 49, mode=PHASE_ANCHOR_MODE)
    assert valid.tolist() == [True, False]
    assert anchors.flatten().tolist() == [1., 0.]
    for frames, boundary in [([16, 32, 49], 49), ([0, 16, 65], 49), ([0, 16, 16], 16)]:
        with pytest.raises(ValueError, match='clock'):
            select_robomme_anchors(history, frames, boundary, mode=PHASE_ANCHOR_MODE)


@pytest.mark.parametrize('future', [False, True])
def test_invalid_demo_is_invisible_and_joint_matches_cached_flow(future):
    m = model(future).eval()
    m.helios_config['anchor_mode'] = PHASE_ANCHOR_MODE
    x = {k: v[:1] for k, v in batch(m).items()}
    x['anchor_valid'][:, 0] = False
    vp, _ = pre(m, x)
    vp = m._apply_anchor_validity(vp, x['anchor_valid'], 32)
    h = vp['meta']['mtm']['L_hist']
    altered = deepcopy(x)
    altered['anchor_latents'][:, 0].add_(1234)
    changed, _ = pre(m, altered)
    m._apply_anchor_validity(changed, x['anchor_valid'], 32)
    cache, hvalid, _ = m._prefill_inference_condition(
        vp, x['video_context'], x['video_context_mask'], decision_frames(82, 49), 49)
    assert not hvalid[:, :16].any() and hvalid[:, 16:].all()
    valid = x['past_action_transition_valid'][..., None]
    action = x['action'].clone()
    past = torch.randn_like(x['past_action_target']).masked_fill(~valid, 0)
    steps, deltas = m.infer_action_scheduler.build_inference_schedule(3, m.device, torch.float32)
    freqs = m._action_freqs(32, vp)
    for t, delta in zip(steps, deltas):
        ap = m.action_expert.pre_dit(action, t[None], x['context'], x['context_mask'])
        ap['freqs'] = freqs
        pp, qvalid = m._flow_pre(x, vp, x['video_context'], x['video_context_mask'], past, t[None])
        cp, _ = m._flow_pre(x, changed, x['video_context'], x['video_context_mask'], past, t[None])
        joint = m._joint_past_forward(vp, ap, pp, qvalid)
        alt = m._joint_past_forward(changed, ap, cp, qvalid)
        # Invalid anchor queries may change; no valid key or other expert can read them.
        for name in ['action', 'past_action']:
            torch.testing.assert_close(joint[name], alt[name], atol=0, rtol=0)
        torch.testing.assert_close(joint['video'][:, 16:], alt['video'][:, 16:], atol=0, rtol=0)
        mask = m._three_expert_mask(vp, 32, qvalid)
        assert not mask[:, :, :16].any()
        combined, key_valid, cached = m.mot.prefill_past_action(pp, cache, qvalid)
        assert not key_valid[:, :16].any()
        torch.testing.assert_close(joint['past_action'], cached, atol=4e-6, rtol=4e-5)
        amask = torch.cat((key_valid[0], torch.ones(32, dtype=torch.bool)))[None].expand(32, -1)
        cached_a = m._predict_action_noise_with_cache(
            action, t[None], x['context'], x['context_mask'], freqs, combined, amask, h+m.query_slots)
        velocity = m.action_expert.post_dit(joint['action'], ap)
        torch.testing.assert_close(velocity, cached_a, atol=4e-6, rtol=4e-5)
        action = m.infer_action_scheduler.step(velocity, delta, action)
        past = m.infer_action_scheduler.step(
            m._past_predictions(joint['past_action']).masked_fill(~valid, 0), delta, past).masked_fill(~valid, 0)


@pytest.mark.parametrize('future', [False, True])
def test_real_training_backward_with_absent_demo(future):
    m = model(future).train()
    m.helios_config['anchor_mode'] = PHASE_ANCHOR_MODE
    m.build_inputs = types.MethodType(lambda self, sample, tiled=False: sample, m)
    x = batch(m)
    x['anchor_valid'][1, 0] = False
    x['anchor_latents'][1, 0] = 0
    loss, _ = m.training_loss(x)
    loss.backward()
    for p in [m.video_expert.patch_embedding.weight, m.video_expert.patch_long.weight,
              m.past_action_expert.noisy_latent_embedding.weight, m.action_expert.head.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize('future', [False, True])
@pytest.mark.parametrize('boundary', [0, 49])
def test_actual_infer_action_accepts_32_and_uses_same_anchors(future, boundary):
    from fastwam.models.wan22.helios_video import MTMVideoDiT
    m = model(future).eval()
    m.helios_config.update(anchor_mode=PHASE_ANCHOR_MODE, observation_contract=CONTRACT)
    m.mot.mixtures['video'] = MTMVideoDiT(
        hidden_dim=32, ffn_dim=64, text_dim=16, freq_dim=16, eps=1e-6,
        num_heads=2, attn_head_dim=24, num_layers=2, in_dim=48, out_dim=48,
        patch_size=(1, 2, 2), has_image_input=False, seperated_timestep=True,
        multi_term_memory=True, mtm_history_sizes=(96, 2, 1), mtm_anchor_size=2, mtm_pred_size=2)
    frames = decision_frames(boundary+33, boundary)
    observations = torch.randn(1, 48, len(frames), 24, 20)
    expected, valid = select_robomme_anchors(
        observations[0].permute(1, 0, 2, 3).unsqueeze(2), frames, boundary, mode=PHASE_ANCHOR_MODE)
    original = m._video_pre
    seen = []
    def capture(self, anchors, *args, **kwargs):
        seen.append(anchors.detach().clone())
        return original(anchors, *args, **kwargs)
    m._video_pre = types.MethodType(capture, m)
    result = m.infer_action(
        prompt=None, input_image=None, action_horizon=32, observation_latents=observations,
        observation_frame_indices=frames, execution_start=boundary, proprio=torch.zeros(8),
        context=torch.randn(3, 16), context_mask=torch.ones(3, dtype=torch.bool),
        full_kv_frame_index=len(frames)-1, num_inference_steps=2, seed=42)
    torch.testing.assert_close(seen[0][0], expected, atol=0, rtol=0)
    assert result['action'].shape == (32, 8) and torch.isfinite(result['action']).all()
    assert result['past_action_query_tokens'] == (18 if future else 16)


def test_checkpoint_anchor_semantics_cannot_silently_change(tmp_path):
    old, new = model(), model()
    for m in [old, new]:
        m.helios_config['observation_contract'] = CONTRACT
    new.helios_config['anchor_mode'] = PHASE_ANCHOR_MODE
    path = tmp_path/'old.pt'
    old.save_checkpoint(path, step=17)
    assert old.load_checkpoint(path) == 17
    with pytest.raises(ValueError, match='contract'):
        new.load_checkpoint(path)
    new.save_checkpoint(path, step=18)
    assert new.load_checkpoint(path) == 18
    with pytest.raises(ValueError, match='contract'):
        old.load_checkpoint(path)
    assert 'anchor_mode' not in inference_contract()
    assert inference_contract()['anchor_source_indices'] == 'first_two_global_decision_latents'
    assert new._helios_inference_contract()['anchor_slot_roles'] == ['demo', 'execution']


def test_all_new_task_defaults_and_legacy_override():
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[1]
    tasks = sorted((root/'configs/task').glob('robomme_helios*.yaml'))
    with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
        for path in tasks:
            cfg = compose(config_name='train', overrides=[f'task={path.stem}'])
            assert cfg.model.helios_config.anchor_mode == PHASE_ANCHOR_MODE
            assert cfg.data.train.anchor_mode == PHASE_ANCHOR_MODE
            if cfg.model.helios_config.get('past_action', {}).get('enabled'):
                assert cfg.model.helios_config.past_action.loss_weight == .1
            if 'future_loss_weight' in cfg.model.helios_config.get('past_action', {}):
                assert cfg.model.helios_config.past_action.future_loss_weight == .1
        cfg = compose(config_name='train', overrides=[
            'task=robomme_helios_pastfuture32_flow', 'model.helios_config.anchor_mode=global_first2'])
        assert cfg.data.train.anchor_mode == LEGACY_ANCHOR_MODE


def test_online_policy_passes_phase_clock_and_reset_clears_anchors(tmp_path, monkeypatch):
    from test_robomme_helios import RecordingModel, RecordingStream, obs
    from experiments.robomme.policy import AnchorHeliosPolicy
    from fastwam.datasets.robodojo import text_filename
    class RecordingPhaseModel(RecordingModel):
        helios_config = dict(RecordingModel.helios_config, anchor_mode=PHASE_ANCHOR_MODE)
    (tmp_path/'stats.json').write_text(json.dumps({
        k: {'default': {'global_mean': [0.]*8, 'global_std': [1.]*8}} for k in ('state', 'action')}))
    torch.save(dict(context=torch.ones(128, 4096), mask=torch.ones(128, dtype=torch.bool)),
               tmp_path/text_filename('test'))
    monkeypatch.setattr('experiments.robomme.policy.CausalVAEStream', RecordingStream)
    m = RecordingPhaseModel()
    policy = AnchorHeliosPolicy(m, tmp_path/'stats.json', tmp_path)
    first = None
    for start, end, initial in [(0, 49, True), (49, 65, False)]:
        policy.infer(obs(start, end, initial))
        call = m.call
        anchors, valid = select_robomme_anchors(
            call['observation_latents'][0].permute(1, 0, 2, 3).unsqueeze(2),
            call['observation_frame_indices'], call['execution_start'], mode=PHASE_ANCHOR_MODE)
        assert valid.all()
        if first is not None:
            torch.testing.assert_close(anchors, first, atol=0, rtol=0)
        first = anchors
    policy.reset()
    policy.infer(obs(0, 1, True))
    call = m.call
    anchors, valid = select_robomme_anchors(
        call['observation_latents'][0].permute(1, 0, 2, 3).unsqueeze(2),
        call['observation_frame_indices'], call['execution_start'], mode=PHASE_ANCHOR_MODE)
    assert valid.tolist() == [False, True] and anchors[0].count_nonzero() == 0
