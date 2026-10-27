"""Causal information flow and train/prefill parity of the real three-expert MoT."""
from copy import deepcopy
import types

import numpy as np
import pytest
import torch

from fastwam.memory.past_action import (
    TARGET_SCHEMA, INTERVAL_SCHEMA, SPARSE_SCHEMA, SPARSE_MODE, sparse_sampling_contract,
    DreamDojoLabels, past_action_layout, past_action_interval_layout)
from fastwam.memory.robomme_clock import decision_frames
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.helios_video import MTMVideoDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.fastwam_helios_past_action import FastWAMHeliosPastAction
from fastwam.models.wan22.fastwam_helios_past_action_block16 import FastWAMHeliosPastActionBlock16
from fastwam.models.wan22.fastwam_helios_past_action_flow import FastWAMHeliosPastActionFlow
from fastwam.models.wan22.fastwam_helios_past_action_sparse import FastWAMHeliosPastActionSparse16x8Flow


def make_model(mode='endpoint'):
    torch.set_num_threads(2)
    torch.manual_seed(13)
    common = dict(hidden_dim=32, ffn_dim=64, text_dim=16, freq_dim=16, eps=1e-6,
                  num_heads=2, attn_head_dim=24, num_layers=2)
    video = MTMVideoDiT(**common, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
                       has_image_input=False, seperated_timestep=True,
                       multi_term_memory=True, mtm_history_sizes=(96, 2, 1),
                       mtm_anchor_size=2, mtm_pred_size=1)
    action = ActionDiT(**common, action_dim=8)
    mot = MoT(dict(video=video, action=action), mot_checkpoint_mixed_attn=False)
    cls = dict(endpoint=FastWAMHeliosPastAction, block16=FastWAMHeliosPastActionBlock16,
               flow=FastWAMHeliosPastActionFlow,
               sparse=FastWAMHeliosPastActionSparse16x8Flow)[mode]
    m = cls(video, action, mot, torch.nn.Identity(), text_dim=16, proprio_dim=8)
    m.helios_config = dict(history_sizes=[96, 2, 1], anchor_frames=2,
                          patch_long=[4, 8, 8], patch_mid=[2, 4, 4],
                          memory_variant='multiscale_long99', condition_augmentation=False)
    m.vae.temporal_downsample_factor = 4
    label = dict(schema=TARGET_SCHEMA, latent_dim=64, teacher_input=False,
                 transition_frame_stride=1, boundary_supervision=False,
                 normalization_mean=[0.] * 64, normalization_std=[1.] * 64,
                 label_fingerprint='test', label_manifest_sha256='test')
    config = dict(enabled=True, label_contract=label)
    if mode != 'endpoint':
        label.update(schema=INTERVAL_SCHEMA, query_target_shape=[16, 64],
                     interval_alignment='previous_decision_exclusive_to_current_decision_inclusive',
                     interval_padding='chronological_left_aligned_invalid_tail',
                     window_predecessor='taken_before_cropping_history',
                     loss_reduction='mean_valid_transitions_per_example_then_mean_nonempty_examples')
    if mode == 'sparse':
        config['target_mode'] = SPARSE_MODE
        label.update(schema=SPARSE_SCHEMA, query_target_shape=[8, 64], **sparse_sampling_contract())
    if mode in ('flow', 'sparse'):
        config['learning_mode'] = 'flow_matching'
        label.update(teacher_input=True, teacher_input_policy='masked_flow_corrupted_training_only',
                     inference_teacher_input=False)
    m._initialize_past_action(config, dict(**common, action_dim=8))
    return m


@pytest.fixture(params=['endpoint', 'block16'])
def model(request):
    return make_model(request.param)


def batch(model):
    b = 2
    context, mask = torch.randn(b, 3, 16), torch.ones(b, 3, dtype=torch.bool)
    layouts = [model.query_layout(decision_frames(82, 49), 49), model.query_layout([0], 0)]
    sample = dict(anchor_latents=torch.randn(b, 2, 4, 1, 8, 8),
                  anchor_valid=torch.ones(b, 2, dtype=torch.bool),
                  history_latents=torch.randn(b, 99, 4, 1, 8, 8),
                  history_valid=torch.ones(b, 99, dtype=torch.bool),
                  input_latents=torch.randn(b, 4, 2, 8, 8), action=torch.randn(b, 16, 8),
                  video_context=context, video_context_mask=mask,
                  context=context.clone(), context_mask=mask.clone(),
                  image_is_pad=torch.zeros(b, 5, dtype=torch.bool),
                  action_is_pad=torch.zeros(b, 16, dtype=torch.bool),
                  past_action_target=torch.randn(
                      (b, model.query_slots, 64) if model.prediction_width == 64
                      else (b, model.query_slots, model.transition_slots, 64)))
    sample.update(torch.utils.data.default_collate(layouts))
    return sample


def forward(m, x):
    vp = m._video_pre(x['anchor_latents'], x['history_latents'], x['input_latents'][:, :, 1:],
                      torch.full((2,), 500.), x['video_context'], x['video_context_mask'])
    ap = m.action_expert.pre_dit(x['action'], torch.full((2,), 700.),
                                x['context'], x['context_mask'])
    ap['freqs'] = m._action_freqs(16, vp)
    return m._training_expert_forward(vp, ap, dict(x, action_flow_timestep=torch.full((2,), 700.)), x), vp, ap


def test_layout_phase_boundary_short_demo_and_rollover():
    x = past_action_layout(decision_frames(82, 49), 49)
    assert x['past_action_frame_indices'][-7:].tolist() == [0, 16, 32, 48, 49, 65, 81]
    assert x['past_action_phase'][-7:].tolist() == [1, 1, 1, 1, 2, 3, 3]
    assert x['past_action_valid'][-7:].tolist() == [False, True, True, True, False, True, True]
    assert x['past_action_relative_time'][-3:].tolist() == [-2., -1., 0.]
    assert not past_action_layout([0], 0)['past_action_valid'].any()
    frames = decision_frames(2100, 42)
    end = past_action_layout(frames, 42)
    assert end['past_action_frame_indices'].tolist() == frames[-99:]
    assert end['past_action_valid'].all()
    with pytest.raises(ValueError, match='clock'):
        past_action_layout([0, 16, 33], 0)


def test_teacher_future_and_action_cannot_leak_into_past(model):
    model.eval()
    x = batch(model)
    with torch.no_grad():
        reference, vp, _ = forward(model, x)
        y = deepcopy(x)
        y['past_action_target'].normal_(100, 100)
        teacher, _, _ = forward(model, y)
        for key in reference:
            torch.testing.assert_close(reference[key], teacher[key], atol=0, rtol=0)
        y['input_latents'][:, :, 1:].normal_(100, 100)
        future, _, _ = forward(model, y)
        torch.testing.assert_close(reference['action'], future['action'])
        torch.testing.assert_close(reference['past_action'], future['past_action'])
        y['action'].normal_(100, 100)
        y['context'].normal_(100, 100)  # Includes action-only proprio context in production.
        action, _, _ = forward(model, y)
        torch.testing.assert_close(reference['past_action'], action['past_action'])
        mask = model._three_expert_mask(vp, 16, x['past_action_valid'])
        h, p = vp['meta']['mtm']['L_hist'], vp['meta']['mtm']['L_pred']
        n = h + p + 16
        assert not mask[:, :h+p, n:].any()
        assert not mask[:, n:, h:n].any()
        assert not mask[1, :, n:].any()  # No transition at execution frame zero.


def test_cache_parity_and_invalid_queries_cannot_change_action(model):
    model.eval()
    x = batch(model)
    with torch.no_grad():
        reference, vp, ap = forward(model, x)
        pp, valid = model._past_pre(x, vp, x['video_context'], x['video_context_mask'])
        h = vp['meta']['mtm']['L_hist']
        video_kv = model.mot.prefill_video_cache(
            video_tokens=vp['tokens'][:, :h], video_freqs=vp['freqs'][:h],
            video_t_mod=vp['t_mod'][:, :h],
            video_context_payload=dict(context=vp['context'], mask=vp['context_mask'][:, :h]),
            video_attention_mask=torch.ones(h, h, dtype=torch.bool))
        kv, key_valid, tokens = model.mot.prefill_past_action(pp, video_kv, valid)
        torch.testing.assert_close(reference['past_action'], tokens, atol=2e-6, rtol=2e-5)
        mask = torch.cat((key_valid, torch.ones(2, 16, dtype=torch.bool)), 1)[:, None].expand(-1, 16, -1)
        # Deployment serves one episode at a time; the cache API takes a 2D mask.
        pred = torch.cat([model.mot.forward_action_with_video_cache(
            ap['tokens'][i:i+1], ap['freqs'], ap['t_mod'][i:i+1],
            dict(context=ap['context'][i:i+1], mask=ap['context_mask'][i:i+1]),
            [{k: v[i:i+1] for k, v in layer.items()} for layer in kv], mask[i], h+99)
            for i in range(2)])
        torch.testing.assert_close(reference['action'], pred, atol=2e-6, rtol=2e-5)
        y = deepcopy(x)
        y['past_action_phase'][~valid] = 3
        y['past_action_relative_time'][~valid] = -123.
        changed, _, _ = forward(model, y)
        torch.testing.assert_close(reference['action'], changed['action'])
        torch.testing.assert_close(reference['past_action'][valid], changed['past_action'][valid])


@pytest.mark.parametrize('checkpoint', [False, True])
def test_gradients_reach_independent_expert_and_empty_loss_is_finite(model, monkeypatch, checkpoint):
    monkeypatch.setenv('FASTWAM_CHECKPOINT_WHOLE_LAYER', '1' if checkpoint else '0')
    model.mot.mot_checkpoint_mixed_attn = checkpoint
    model.train()
    x = batch(model)
    out, _, _ = forward(model, x)
    out['action'].square().mean().backward()
    assert model.past_action_expert.query.grad.abs().sum() > 0
    assert model.past_action_expert.blocks[0].self_attn.v.weight.grad.abs().sum() > 0
    assert model.past_action_expert.head.weight.grad is None  # Read hidden features, not teacher/head.
    model.zero_grad(set_to_none=True)
    model.build_inputs = types.MethodType(lambda self, sample, tiled=False: sample, model)
    loss, metrics = model.training_loss(x)
    assert metrics['past_action_valid_queries'] == 2.5
    loss.backward()
    for p in (model.past_action_expert.query, model.past_action_expert.head.weight,
              model.past_action_expert.relative_time_embedding[0].weight,
              model.video_expert.patch_long.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert (model.action_expert.blocks[0].self_attn.q.weight.data_ptr()
            != model.past_action_expert.blocks[0].self_attn.q.weight.data_ptr())
    out, _, _ = forward(model, x)
    x['past_action_valid'].zero_()
    if 'past_action_transition_valid' in x:
        x['past_action_transition_valid'].zero_()
    empty, _ = model._extra_training_loss(out, x)
    assert empty.item() == 0 and torch.isfinite(empty)


def test_checkpoint_restores_third_expert_and_rejects_contract_change(model, tmp_path):
    path = tmp_path/'past.pt'
    expected = model.past_action_expert.query.detach().clone()
    model.save_checkpoint(path, step=123)
    with torch.no_grad():
        model.past_action_expert.query.zero_()
    assert model.load_checkpoint(path) == 123
    torch.testing.assert_close(model.past_action_expert.query, expected)
    model.helios_config['past_action']['action_reads_past'] = False
    with pytest.raises(ValueError, match='contract'):
        model.load_checkpoint(path)


def test_auxiliary_only_does_not_condition_actions(model):
    model.eval()
    model.helios_config['past_action']['action_reads_past'] = False
    x = batch(model)
    with torch.no_grad():
        a, _, _ = forward(model, x)
        model.past_action_expert.query.normal_(10, 100)
        b, _, _ = forward(model, x)
    torch.testing.assert_close(a['action'], b['action'])
    assert not torch.allclose(a['past_action'], b['past_action'])
