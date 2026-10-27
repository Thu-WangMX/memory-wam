"""Real pretrained backward, label alignment, and demo/execution online replay.

This performs a small optimizer smoke step, not a benchmark training run.
Large gradients are checked then released to fit on one otherwise idle GPU.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.robomme.train import validate_recipe
from experiments.robomme.policy import AnchorHeliosPolicy
from fastwam.memory.past_action import past_action_layout
from fastwam.memory.robomme_clock import decision_frames, select_robomme_anchors
from scripts.smoke_robomme_helios import replay_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--memory-gib', type=float, default=65)
    parser.add_argument('--task', default='robomme_helios_past_action', choices=[
        'robomme_helios_past_action', 'robomme_helios_past_action_block16',
        'robomme_helios_past_action_flow', 'robomme_helios_past_action_sparse16x8_flow',
        'robomme_helios_past32_flow', 'robomme_helios_pastfuture32_flow',
        'robomme_helios_pastfuture32_light_flow'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(
        args.memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    root, out = Path(__file__).resolve().parents[1], Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=[
            f'task={args.task}', f'output_dir={out}', 'wandb.enabled=false'])
    report = dict(data=validate_recipe(cfg), trained_checkpoint=False)
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), out/'config.yaml')
    shutil.copyfile(cfg.data.train.pretrained_norm_stats, out/'dataset_stats.json')
    ds = instantiate(cfg.data.train)
    target_shape = ((99, 64) if ds.target_mode == 'endpoint' else
                    (16, 8, 64) if ds.target_mode == 'uniform16_half8' else
                    (18, 8, 64) if ds.target_mode == 'past16_future2_half8' else (99, 16, 64))
    validity_key = ('past_action_valid' if ds.target_mode == 'endpoint'
                    else 'past_action_transition_valid')
    # Audit target alignment at first and last execution decisions in all tasks.
    checked = []
    for task, episode_ids in ds.task_episodes.items():
        ep = episode_ids[0]
        for index in (ds.episode_samples[ep][0], ds.episode_samples[ep][-1]):
            sample = ds[index]
            assert sample['past_action_target'].shape == target_shape
            assert torch.count_nonzero(sample['past_action_target'][~sample[validity_key]]) == 0
            assert sample['past_action_frame_indices'][:16 if ds.target_mode == 'past16_future2_half8' else 99].max() <= sample['frame_index']
            checked.append(dict(task=task, episode=ep, frame=sample['frame_index'],
                                valid_queries=int(sample['past_action_valid'].sum()),
                                valid_transitions=int(sample[validity_key].sum())))
    report['label_audit'] = checked
    print('LABEL_ALIGNMENT_OK', len(checked), flush=True)
    model = instantiate(cfg.model)
    model.eval().requires_grad_(False)
    model.dit.train().requires_grad_(True)
    model.proprio_encoder.train().requires_grad_(True)
    model.freeze_unused_history_branches()
    row = max(ds.episodes, key=lambda r: len(decision_frames(r['length'], r['exec_start_idx'])))
    sample = ds[ds.episode_samples[row['episode_index']][-1]]
    batch = torch.utils.data.default_collate([sample])
    kept = {n: p for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in (
                'past_action.query', 'past_action.head.', 'past_action.phase_embedding.',
                'past_action.noisy_latent_embedding.', 'past_action.interval_length_embedding.',
                'past_action.direction_embedding.',
                'patch_long.', 'patch_mid.', 'proprio_encoder.'))}
    # Match the FP32 master-weight update used by mixed-precision ZeRO training.
    masters = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in kept.items()}
    optimizer = torch.optim.AdamW(masters.values(), lr=1e-4)
    gradients, hooks = {}, []
    def check(name):
        def hook(parameter):
            grad = parameter.grad
            if not torch.isfinite(grad).all():
                raise ValueError(f'Nonfinite gradient: {name}')
            gradients[name] = float(grad.float().norm())
            if name not in kept:
                parameter.grad = None
        return hook
    for name, param in model.named_parameters():
        if param.requires_grad:
            hooks.append(param.register_post_accumulate_grad_hook(check(name)))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    loss, metrics = model.training_loss(batch)
    # Convert only at this diagnostic output boundary.
    metrics = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
               for key, value in metrics.items()}
    assert torch.isfinite(loss)
    print('FORWARD_OK '+json.dumps(metrics), flush=True)
    loss.backward()
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()
    for suffix in ('past_action.query', 'past_action.head.weight',
                   'past_action.blocks.0.self_attn.v.weight', 'patch_long.weight'):
        values = [v for k, v in gradients.items() if k.endswith(suffix)]
        assert len(values) == 1 and values[0] > 0, (suffix, values)
    if ds.learning_mode == 'flow_matching':
        for suffix in ('past_action.noisy_latent_embedding.weight',
                       'past_action.interval_length_embedding.weight',
                       'past_action.time_projection.1.weight'):
            assert next(v for k, v in gradients.items() if k.endswith(suffix)) > 0
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and 'mixtures.past_action.' in n and n not in gradients]
    assert not missing, missing
    before = model.past_action_expert.query.detach().clone()
    for name, master in masters.items():
        master.grad = kept[name].grad.float()
    optimizer.step()
    with torch.no_grad():
        for name, master in masters.items():
            kept[name].copy_(master)
    assert not torch.equal(before, model.past_action_expert.query)
    optimizer.zero_grad(set_to_none=True)
    report['backward'] = dict(
        metrics=metrics, seconds=time.monotonic()-started, gradient_tensors=len(gradients),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        past_action_parameters=sum(p.numel() for p in model.past_action_expert.parameters()),
        optimizer_scope='FP32-master AdamW for query/head/phase/interval/noisy-input + memory projections + proprio; other gradients checked then released')
    (out/'backward.json').write_text(json.dumps(report['backward'], indent=2)+'\n')
    print('FULL_BACKWARD_OK '+json.dumps(report['backward']), flush=True)
    del loss, batch, optimizer, before, masters, master
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    torch.cuda.empty_cache()
    cache = ds.observed_prefix(row, sample['frame_index'])
    observations = cache['latents']
    frames = cache['frame_indices'][:len(observations)].tolist()
    prefix = observations.squeeze(2).permute(1, 0, 2, 3).unsqueeze(0)
    prediction = model.infer_action(
        prompt=None, input_image=None, observation_latents=prefix, action_horizon=ds.prediction_horizon,
        observation_frame_indices=frames, execution_start=row['exec_start_idx'],
        proprio=sample['proprio'][0], context=sample['context'], context_mask=sample['context_mask'],
        full_kv_frame_index=len(observations)-1, num_inference_steps=2, seed=42)
    torch.testing.assert_close(prediction['past_action_valid'], sample['past_action_valid'])
    assert prediction['past_action_prediction'].shape == target_shape
    assert torch.isfinite(prediction['past_action_prediction']).all()
    assert torch.count_nonzero(prediction['past_action_prediction'][~sample[validity_key]]) == 0
    assert torch.isfinite(prediction['action']).all()
    report['long_prefix'] = dict(episode=row['episode_index'], observations=len(observations),
                               visual_tokens=prediction['full_kv_video_tokens'],
                               query_tokens=prediction['past_action_query_tokens'],
                               target_shape=list(target_shape),
                               flow_steps=prediction.get('past_action_flow_steps'))
    print('LONG_PREFIX_OK '+json.dumps(report['long_prefix']), flush=True)
    policy = AnchorHeliosPolicy(model, out/'dataset_stats.json', ds.prepared_root/'text_cache', 2, 42)
    cases = [min((r for r in ds.rows if r['exec_start_idx'] > 0 and
                  r['length']-r['exec_start_idx'] > 32), key=lambda r: r['exec_start_idx']),
             next(r for r in ds.rows if r['exec_start_idx'] == 0 and r['length'] > 32)]
    report['replay'] = []
    for row in cases:
        policy.reset()
        cache = ds.load_latents(row)
        for decision in range(2):
            t = row['exec_start_idx'] + 16*decision
            inputs = replay_inputs(ds, row, 0 if decision == 0 else t-15, t+1, decision == 0)
            result = policy.infer(inputs)
            j = cache['frame_indices'].tolist().index(t)
            expected = cache['latents'][:j+1].squeeze(2).permute(1, 0, 2, 3).unsqueeze(0)
            torch.testing.assert_close(policy.last_prefix, expected, atol=.04, rtol=.01)
            assert policy.observation_frame_indices == cache['frame_indices'][:j+1].tolist()
            # Dense sample indices are raw-frame positions, not decision ranks.
            offline_index = next(i for i in ds.episode_samples[row['episode_index']]
                                 if ds.samples[i][1] == t)
            offline = ds[offline_index]
            online_anchors, online_valid = select_robomme_anchors(
                policy.last_prefix[0].permute(1, 0, 2, 3).unsqueeze(2),
                policy.observation_frame_indices, policy.execution_start,
                size=ds.anchor_frames, mode=ds.anchor_mode)
            torch.testing.assert_close(online_valid, offline['anchor_valid'], atol=0, rtol=0)
            torch.testing.assert_close(online_anchors, offline['anchor_latents'], atol=.04, rtol=.01)
            online = ds.query_layout(policy.observation_frame_indices, policy.execution_start)
            for name, tensor in online.items():
                torch.testing.assert_close(tensor, offline[name], atol=0, rtol=0)
            report['replay'].append(dict(episode=row['episode_index'], frame=t,
                                        action_shape=list(result['actions'].shape),
                                        query_clock_exact=True, phase_vae_matches_cache=True,
                                        anchor_mode=ds.anchor_mode, phase_anchors_match_cache=True))
            print('ONLINE_REPLAY_OK '+json.dumps(report['replay'][-1]), flush=True)
    (out/'model_smoke.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PAST_ACTION_MODEL_SMOKE_OK', flush=True)


if __name__ == '__main__':
    main()
