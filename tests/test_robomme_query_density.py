"""Configurable visual windows and past-query density preserve causal contracts."""
from copy import deepcopy
import types

import numpy as np
import pytest
import torch

from fastwam.memory.past_action import (
    DreamDojoLabels, SPARSE_MODE, SPARSE_SCHEMA, sparse_sampling_contract,
    sparse_target_schema, past_action_interval_layout, past_action_sparse_layout,
    validate_query_counts,
)
from fastwam.memory.past_future_action import (
    MODE, SCHEMA, sampling_contract, target_schema, past_future_layout,
)
from fastwam.memory.robomme_dense_clock import dense_prefix_frames
from fastwam.models.wan22.fastwam_helios_past_future import FastWAMHeliosPastFuture
from test_robomme_past_action import make_model as legacy_model, batch as legacy_batch
from test_robomme_horizon32 import pre
from test_robomme_past_flow import video_cache


def make_model(source=75, past=32, future=True):
    m = legacy_model('sparse')
    m.helios_config.update(
        history_sizes=[source-3, 2, 1], prediction_horizon=32,
        memory_variant='multiscale_long75' if source == 75 else 'multiscale_long99')
    m.video_expert.mtm_history_sizes = (source-3, 2, 1)
    m.video_expert.mtm_pred_size = 2
    label = deepcopy(m.helios_config['past_action']['label_contract'])
    contract = sampling_contract(source, past) if future else sparse_sampling_contract(source, past)
    schema = target_schema(source, past) if future else sparse_target_schema(source, past)
    label.update(schema=schema, **contract)
    if future:
        m.__class__ = FastWAMHeliosPastFuture
    common = dict(hidden_dim=32, ffn_dim=64, text_dim=16, freq_dim=16, eps=1e-6,
                  num_heads=2, attn_head_dim=24, num_layers=2, action_dim=8)
    m._initialize_past_action(dict(enabled=True, learning_mode='flow_matching',
        target_mode=MODE if future else SPARSE_MODE, label_contract=label,
        source_history_slots=source, past_query_slots=past), common)
    return m


def batch(m, current=1635, boundary=35):
    x = legacy_batch(m)
    x.update(history_latents=torch.randn(2, m.source_history_slots, 4, 1, 8, 8),
             history_valid=torch.ones(2, m.source_history_slots, dtype=torch.bool),
             input_latents=torch.randn(2, 4, 3, 8, 8), action=torch.randn(2, 32, 8),
             image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
             action_is_pad=torch.zeros(2, 32, dtype=torch.bool))
    layout = m.query_layout(dense_prefix_frames(current, boundary), boundary)
    x.update({key: torch.stack((value, value)) for key, value in layout.items()})
    if m.future_query_slots:
        x['past_action_loss_valid'] = x['past_action_transition_valid'].clone()
    return x


def test_legacy_default_contract_and_queries_remain_exact():
    expected = dict(query_slots=16, source_history_slots=99,
        query_selection='uniform_valid_interval_rank_rint_linspace_inclusive',
        query_padding='left_pad_invalid_no_repetition',
        transition_selection='endpoint_aligned_stride2_chronological',
        sampled_transition_stride=2, preserve_original_visual_slot=True)
    assert sparse_sampling_contract() == sparse_sampling_contract(99, 16) == expected
    assert sparse_target_schema() == SPARSE_SCHEMA
    assert target_schema() == SCHEMA
    assert sampling_contract() == sampling_contract(99, 16)
    assert sampling_contract()['query_slots'] == 18
    for boundary, current in [(0, 0), (35, 36), (1100, 2901)]:
        frames = dense_prefix_frames(current, boundary)
        for fn, kwargs in [(past_action_sparse_layout, dict(query_slots=16)),
                           (past_future_layout, dict(past_query_slots=16))]:
            a, b = fn(frames, boundary), fn(frames, boundary, 99, **kwargs)
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


@pytest.mark.parametrize('source,past', [(75, 16), (75, 32), (99, 16), (99, 32)])
@pytest.mark.parametrize('boundary', [0, 35, 1100])
@pytest.mark.parametrize('offset', range(16))
def test_density_uniform_selection_preserves_source_clock_and_all_offsets(source, past, boundary, offset):
    current = boundary + 1600 + offset
    frames = dense_prefix_frames(current, boundary)
    dense = past_action_interval_layout(frames, boundary, source)
    layout = past_future_layout(frames, boundary, source, past_query_slots=past)
    assert layout['past_action_transition_frames'].shape == (past+2, 8)
    selected = layout['past_action_visual_slots'][:past]
    candidates = torch.where(dense['past_action_valid'])[0]
    ranks = np.rint(np.linspace(0, len(candidates)-1, past)).astype(np.int64)
    assert selected.tolist() == candidates[ranks].tolist()
    assert len(set(selected.tolist())) == past
    for query, slot in enumerate(selected.tolist()):
        original = dense['past_action_transition_frames'][slot, dense['past_action_transition_valid'][slot]]
        valid = layout['past_action_transition_valid'][query]
        target = layout['past_action_transition_frames'][query, valid]
        assert target.tolist() == original.tolist()[::-2][::-1]
        assert (target <= current).all() and (target != boundary).all()
    assert layout['past_action_visual_slots'][-2:].tolist() == [source, source+1]
    assert layout['past_action_transition_frames'][-2:].tolist() == [
        list(range(current+2, current+17, 2)), list(range(current+18, current+33, 2))]
    assert layout['past_action_valid'][-2:].all()


@pytest.mark.parametrize('current', [35, 36, 39, 50, 51, 68])
def test_short_histories_left_pad_queries_without_repetition(current):
    frames = dense_prefix_frames(current, 35)
    dense = past_action_interval_layout(frames, 35, 75)
    layout = past_action_sparse_layout(frames, 35, 75, query_slots=32)
    valid = layout['past_action_valid']
    candidates = torch.where(dense['past_action_valid'])[0]
    assert valid.sum() == len(candidates)
    assert layout['past_action_visual_slots'][valid].tolist() == candidates.tolist()
    assert not valid[:32-len(candidates)].any()
    assert (layout['past_action_visual_slots'][~valid] == -1).all()
    assert not layout['past_action_transition_valid'][~valid].any()


@pytest.mark.parametrize('source,past', [(0, 16), (75, 0), (75, 76), (75, True), (75., 32), (75, 32.0)])
def test_invalid_query_counts_rejected(source, past):
    with pytest.raises(ValueError):
        validate_query_counts(source, past)


@pytest.mark.parametrize('future', [False, True])
def test_dataset_binds_same_configured_layout_and_contract_as_online(monkeypatch, future):
    from fastwam.datasets.robomme import RoboMMEHeliosDataset
    from fastwam.datasets.robomme_past_action import RoboMMEPastActionSparse16x8FlowDataset
    from fastwam.datasets.robomme_past_future import RoboMMEPastFutureActionDataset
    labels = DreamDojoLabels.__new__(DreamDojoLabels)
    labels.manifest, labels.manifest_sha256 = {'fingerprint': 'test'}, 'test'
    labels.mean, labels.std = np.zeros(64), np.ones(64)
    def base_init(self, **kwargs):
        self.history_sizes, self.anchor_frames = (72, 2, 1), 2
        self.prediction_horizon, self.root, self.rows = 32, '/unused', []
    monkeypatch.setattr(RoboMMEHeliosDataset, '__init__', base_init)
    monkeypatch.setattr('fastwam.datasets.robomme_past_action.DreamDojoLabels', lambda *args: labels)
    cls = RoboMMEPastFutureActionDataset if future else RoboMMEPastActionSparse16x8FlowDataset
    dataset = cls(dreamdojo_labels_root='/unused', past_query_slots=32)
    m = make_model(future=future)
    frames = dense_prefix_frames(1638, 35)
    offline, online = dataset.query_layout(frames, 35), m.query_layout(frames, 35)
    for key in offline:
        torch.testing.assert_close(offline[key], online[key], atol=0, rtol=0)
    contract = dataset.dreamdojo_contract()
    assert contract['source_history_slots'] == 75 and contract['past_query_slots'] == 32
    assert contract['query_slots'] == (34 if future else 32)
    assert contract['schema'] == m.target_schema


def test_new_model_rope_preserves_original_visual_slots_and_future_endpoints():
    m = make_model()
    x = batch(m)
    freqs = m._past_freqs(x)
    assert freqs.shape[:3] == (2, 34, 1)
    base = m.video_expert.freqs
    positions = x['past_action_visual_slots'].float() + 2
    expected = positions[..., None] * torch.angle(base[0][1]).float()
    torch.testing.assert_close(freqs[..., 0, :base[0].shape[-1]],
                              torch.polar(torch.ones_like(expected), expected).to(freqs.dtype))
    assert positions[0, -2:].tolist() == [77, 78]
    wrong = dict(x, past_action_visual_slots=x['past_action_visual_slots'].clone())
    wrong['past_action_visual_slots'][:, -2:] = torch.tensor([99, 100])
    with pytest.raises(ValueError, match='positions'):
        m._past_freqs(wrong)


def test_34_token_attention_and_joint_cached_solver_match():
    m = make_model().eval()
    x = {k: v[:1] for k, v in batch(m).items()}
    vp, _ = pre(m, x)
    cache = video_cache(m, vp)
    h = vp['meta']['mtm']['L_hist']
    valid = x['past_action_transition_valid'][..., None]
    action = x['action'].clone()
    past = torch.randn(1, 34, 8, 64, generator=torch.Generator().manual_seed(121)).masked_fill(~valid, 0)
    steps, deltas = m.infer_action_scheduler.build_inference_schedule(2, m.device, torch.float32)
    freqs = m._action_freqs(32, vp)
    for t, delta in zip(steps, deltas):
        ap = m.action_expert.pre_dit(action, t[None], x['context'], x['context_mask'])
        ap['freqs'] = freqs
        pp, qvalid = m._flow_pre(x, vp, x['video_context'], x['video_context_mask'], past, t[None])
        assert not pp['query_attention_mask'][:32, 32:].any()
        assert pp['query_attention_mask'][32:, :].all()
        joint = m._joint_past_forward(vp, ap, pp, qvalid)
        changed = past.clone(); changed[:, 32:] += 999
        cp, _ = m._flow_pre(x, vp, x['video_context'], x['video_context_mask'], changed, t[None])
        alternate = m._joint_past_forward(vp, ap, cp, qvalid)
        torch.testing.assert_close(joint['past_action'][:, :32], alternate['past_action'][:, :32], atol=0, rtol=0)
        assert not torch.allclose(joint['action'], alternate['action'])
        combined, key_valid, p_cached = m.mot.prefill_past_action(pp, cache, qvalid)
        torch.testing.assert_close(joint['past_action'], p_cached, atol=4e-6, rtol=4e-5)
        mask = torch.cat((key_valid[0], torch.ones(32, dtype=torch.bool)))[None].expand(32, -1)
        a_cached = m._predict_action_noise_with_cache(action, t[None], x['context'],
            x['context_mask'], freqs, combined, mask, h+34)
        velocity = m.action_expert.post_dit(joint['action'], ap)
        torch.testing.assert_close(velocity, a_cached, atol=4e-6, rtol=4e-5)
        action = m.infer_action_scheduler.step(velocity, delta, action)
        past = m.infer_action_scheduler.step(m._past_predictions(joint['past_action']).masked_fill(~valid, 0),
                                            delta, past).masked_fill(~valid, 0)
    got, extra = m._denoise_action_with_condition(x['action'].clone(), steps, deltas,
        x['context'], x['context_mask'], freqs, cache, torch.ones(32, h+32, dtype=torch.bool), h, {},
        vp=vp, context=x['video_context'], context_mask=x['video_context_mask'],
        observation_frame_indices=dense_prefix_frames(1635, 35), execution_start=35,
        generator=torch.Generator().manual_seed(121), rand_device='cpu')
    torch.testing.assert_close(got, action, atol=6e-6, rtol=6e-5)
    torch.testing.assert_close(extra['past_action_prediction'], past[0], atol=6e-6, rtol=6e-5)
    assert extra['past_action_query_tokens'] == 34


@pytest.mark.parametrize('source,past', [(99, 16), (75, 32)])
def test_loss_is_normalized_per_branch_not_scaled_by_more_query_tokens(source, past):
    m = make_model(source, past)
    m._past_predictions = types.MethodType(lambda self, tokens: tokens, m)
    prediction = torch.ones(1, past+2, 8, 64, requires_grad=True)
    prediction.data[:, past:] = 2
    out = dict(past_action=prediction, past_action_flow_target=torch.zeros_like(prediction),
               past_action_flow_timestep=torch.tensor([700.]))
    sample = dict(past_action_loss_valid=torch.ones(1, past+2, 8, dtype=torch.bool))
    loss, metrics = m._extra_training_loss(out, sample)
    weight = float(m.train_action_scheduler.training_weight(torch.tensor([700.])))
    assert metrics['loss_past_action_unweighted'] == pytest.approx(weight)
    assert metrics['loss_future_action_unweighted'] == pytest.approx(4*weight)
    assert float(loss) == pytest.approx(.5*weight)
    assert metrics['past_action_valid_transitions'] == past*8
    sample['past_action_loss_valid'][:, past:] = False
    _, metrics = m._extra_training_loss(out, sample)
    assert metrics['loss_future_action_unweighted'] == 0


def test_sampling_contract_saved_and_cross_density_checkpoint_rejected(tmp_path):
    old, new = make_model(99, 16), make_model(75, 32)
    assert old.helios_config['past_action']['query_slots'] == 18
    assert new.helios_config['past_action']['query_slots'] == 34
    assert new.helios_config['past_action']['past_query_slots'] == 32
    assert new.helios_config['past_action']['source_history_slots'] == 75
    for model, wrong, name in [(old, new, 'old'), (new, old, 'new')]:
        path = tmp_path / (name+'.pt')
        model.save_checkpoint(path, step=7)
        assert model.load_checkpoint(path) == 7
        with pytest.raises(ValueError, match='contract'):
            wrong.load_checkpoint(path)


def test_config_source_mismatch_rejected_before_building_expert():
    m = make_model()
    with pytest.raises(ValueError, match='source_history_slots'):
        m._configure_query_layout(dict(source_history_slots=99, past_query_slots=32))


def test_75_source_34_queries_compact_history_preserves_positions_and_cache():
    from test_robomme_compact_history import _pre, _joint
    m = make_model().eval()
    x = {key: value[:1] for key, value in batch(m, current=67, boundary=35).items()}
    frames = dense_prefix_frames(67, 35)
    x['history_valid'][:] = False
    x['history_valid'][:, -len(frames):] = True
    fixed = _pre(m, x, 'mask')
    reference, _, _, _ = _joint(m, x, fixed)
    compact = _pre(m, x, 'compact')
    out, ap, pp, valid = _joint(m, x, compact)
    assert compact['meta']['mtm']['current_frame_index'] == 76
    assert compact['meta']['mtm']['L_hist'] < fixed['meta']['mtm']['L_hist']
    assert (x['past_action_visual_slots'][0, -2:]+2).tolist() == [77, 78]
    for branch in ('action', 'past_action'):
        torch.testing.assert_close(out[branch], reference[branch], atol=5e-6, rtol=5e-5)
    kv, _, _ = m._prefill_inference_condition(compact, x['video_context'],
        x['video_context_mask'], frames, 35)
    combined, key_valid, cached = m.mot.prefill_past_action(pp, kv, valid)
    torch.testing.assert_close(cached, out['past_action'], atol=5e-6, rtol=5e-5)
    mask = torch.cat((key_valid[0], torch.ones(32, dtype=torch.bool)))[None].expand(32, -1)
    velocity = m._predict_action_noise_with_cache(x['action'], torch.tensor([700.]),
        x['context'], x['context_mask'], ap['freqs'], combined, mask,
        kv[0]['k'].shape[1]+m.query_slots)
    torch.testing.assert_close(velocity, m.action_expert.post_dit(out['action'], ap),
                              atol=5e-6, rtol=5e-5)
