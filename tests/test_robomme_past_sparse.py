"""Sparse selection must preserve causal label meaning and original time positions."""
import numpy as np
import pytest
import torch

from fastwam.memory.past_action import (
    DreamDojoLabels, past_action_interval_layout, past_action_sparse_layout,
    SPARSE_MODE, SPARSE_SCHEMA)
from fastwam.memory.robomme_clock import decision_frames
from test_robomme_past_action import make_model, batch, forward


@pytest.mark.parametrize('boundary', [0, 1, 2, 16, 17, 18, 33, 35, 49, 1100])
def test_sparse_clock_causal_phase_safe_uniform_and_no_duplicates(boundary):
    for current in [boundary, boundary+16, boundary+32, boundary+400, boundary+1800]:
        clock = decision_frames(current+1, boundary)
        dense = past_action_interval_layout(clock, boundary)
        x = past_action_sparse_layout(clock, boundary)
        valid = x['past_action_valid']
        candidates = torch.where(dense['past_action_valid'])[0].tolist()
        slots = x['past_action_visual_slots'][valid].tolist()
        assert len(slots) == min(16, len(candidates))
        assert slots == sorted(set(slots))
        if slots:
            assert slots[0] == candidates[0] and slots[-1] == candidates[-1]
            if len(candidates) > 16:
                gaps = np.diff([candidates.index(s) for s in slots])
                assert gaps.max()-gaps.min() <= 1
        assert x['past_action_transition_frames'].shape == (16, 8)
        assert (x['past_action_visual_slots'][~valid] == -1).all()
        assert (x['past_action_transition_frames'][~x['past_action_transition_valid']] == -1).all()
        for q in torch.where(valid)[0].tolist():
            slot = int(x['past_action_visual_slots'][q])
            frames = x['past_action_transition_frames'][q, x['past_action_transition_valid'][q]]
            original = dense['past_action_transition_frames'][slot, dense['past_action_transition_valid'][slot]]
            assert frames.tolist() == original.tolist()[::-2][::-1]
            assert len(frames) == (len(original)+1)//2
            assert frames[-1] == original[-1]
            assert (frames <= current).all() and (frames != boundary).all()
            assert ((frames < boundary).all() or (frames > boundary).all())
            assert x['past_action_interval_length'][q] == dense['past_action_interval_length'][slot]
            assert x['past_action_relative_time'][q] == dense['past_action_relative_time'][slot]
            assert x['past_action_frame_indices'][q] == dense['past_action_frame_indices'][slot]


def test_halfrate_targets_read_existing_adjacent_labels_and_odd_tail():
    x = past_action_sparse_layout(decision_frames(52, 35), 35)
    labels = DreamDojoLabels.__new__(DreamDojoLabels)
    labels.offsets = {0: (0, 52)}
    labels._latents = np.arange(52, dtype=np.float32)[:, None].repeat(64, 1)
    labels.valid = np.ones(52, dtype=bool)
    labels.valid[[0, 35]] = False
    labels.phase = np.where(np.arange(52) < 35, 1, 3)
    labels.phase[35] = 2
    labels.mean, labels.std = np.zeros(64), np.ones(64)*2
    y = labels.targets(0, x)
    assert y.shape == (16, 8, 64)
    assert y[-1, :, 0].tolist() == [i/2 for i in range(37, 52, 2)]
    assert x['past_action_transition_offsets'][-1].tolist() == list(range(2, 17, 2))
    assert x['past_action_transition_frames'][-2].tolist() == [34]+[-1]*7
    assert torch.count_nonzero(y[~x['past_action_transition_valid']]) == 0
    odd = past_action_sparse_layout(decision_frames(37, 36), 36)
    assert odd['past_action_transition_frames'][-1, :3].tolist() == [33, 35, -1]
    labels.manifest, labels.manifest_sha256 = {'fingerprint': 'test'}, 'test'
    contract = labels.contract(SPARSE_MODE)
    assert contract['schema'] == SPARSE_SCHEMA and contract['query_target_shape'] == [8, 64]
    assert contract['transition_frame_stride'] == 1
    assert contract['sampled_transition_stride'] == 2


def test_sparse_rope_matches_original_slots_for_different_batch_histories():
    sparse, dense = make_model('sparse'), make_model('flow')
    xs = batch(sparse)
    layout = sparse.query_layout(decision_frames(1601, 321), 321)
    xs.update({k: torch.stack([v, v]) for k, v in layout.items()})
    # A second history produces a different distribution of selected positions.
    other = sparse.query_layout(decision_frames(402, 49), 49)
    for k, v in other.items():
        xs[k][1] = v
    sf, df = sparse._past_freqs(xs), dense._past_freqs({})
    assert sf.shape[:3] == (2, 16, 1)
    for b in range(2):
        valid = xs['past_action_valid'][b]
        torch.testing.assert_close(sf[b, valid], df[xs['past_action_visual_slots'][b, valid]])
    # Exercise batched RoPE through the actual three-expert forward.
    out, _, _ = forward(sparse, xs)
    assert out['past_action'].shape[:2] == (2, 16)
    assert torch.isfinite(out['action']).all()


def test_invalid_sparse_noise_cannot_change_actions_or_valid_predictions():
    m = make_model('sparse').eval()
    x = batch(m)
    _, vp, ap = forward(m, x)
    noise = torch.randn(2, 16, 8, 64)
    a, valid = m._flow_pre(x, vp, x['video_context'], x['video_context_mask'], noise, torch.ones(2)*700)
    reference = m._joint_past_forward(vp, ap, a, valid)
    noise[~x['past_action_transition_valid']] = 100000
    b, valid = m._flow_pre(x, vp, x['video_context'], x['video_context_mask'], noise, torch.ones(2)*700)
    changed = m._joint_past_forward(vp, ap, b, valid)
    torch.testing.assert_close(reference['action'], changed['action'], atol=0, rtol=0)
    torch.testing.assert_close(reference['past_action'], changed['past_action'], atol=0, rtol=0)
