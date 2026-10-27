"""Replay real RGB through online policy and compare every cached VAE prefix.

The real VAE and AnchorHeliosPolicy.infer run unchanged. The action-network call
returns normalized zeros; this validates visual/clock/execution-length parity,
not action-network predictions or benchmark success.
"""
import argparse
import json
import os
from pathlib import Path
import time
import types

import numpy as np
import torch

from experiments.robomme.policy import AnchorHeliosPolicy
from fastwam.datasets.robodojo import read_jsonl, sha256
from fastwam.memory.robomme_clock import CONTRACT, PHASE_ANCHOR_MODE, decision_frames
from fastwam.memory.robomme_views import TWO_VIEW_HORIZONTAL, validate_view_metadata
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.helpers.loader import _load_registered_model
from scripts.prepare_robomme_dense_latents import (
    atomic_json, parse_indices, read_execution_rgb, resolve_vae,
)


class VAEOnlyModel:
    def __init__(self, vae, device):
        self.vae, self.device, self.torch_dtype = vae, device, torch.bfloat16
        self.helios_config = dict(observation_contract=CONTRACT,
                                 view_layout=TWO_VIEW_HORIZONTAL,
                                 anchor_mode=PHASE_ANCHOR_MODE, prediction_horizon=32)
        self._encode_video_latents = types.MethodType(FastWAM._encode_video_latents, self)
        self.calls = []

    def infer_action(self, **kwargs):
        self.calls.append(dict(frame_indices=list(kwargs['observation_frame_indices']),
                               execution_start=kwargs['execution_start'],
                               prediction_horizon=kwargs['action_horizon']))
        return {'action': torch.zeros(kwargs['action_horizon'], 8)}


def observations(front, wrist, begin, end, first, instruction):
    count = end-begin
    return dict(front_rgb_list=front[begin:end], wrist_rgb_list=wrist[begin:end],
                joint_state_list=[np.zeros(7, dtype=np.float32)]*count,
                gripper_state_list=[np.zeros(2, dtype=np.float32)]*count,
                is_first_step=first, task_goal=instruction)


def compare(actual, expected):
    if actual.shape != expected.shape:
        return dict(bitwise_equal=False, actual_shape=list(actual.shape),
                    expected_shape=list(expected.shape))
    delta = (actual.float()-expected.float()).abs()
    return dict(bitwise_equal=torch.equal(actual, expected),
                max_error=delta.max().item() if delta.numel() else 0.,
                mean_error=delta.mean().item() if delta.numel() else 0.)


@torch.no_grad()
def run(args, report):
    dataset, base, prepared = map(Path, (args.dataset_root, args.base_cache, args.prepared_root))
    metadata = json.loads((base/'config.json').read_text())
    validate_view_metadata(metadata, TWO_VIEW_HORIZONTAL)
    if sha256(prepared/'provenance.json') != metadata['source_prepared_provenance_sha256']:
        raise ValueError('Use the original audited prepared assets for this unfinished-cache smoke')
    info = json.loads((dataset/'meta/info.json').read_text())
    rows = read_jsonl(dataset/'meta/episodes.jsonl')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.memory_gib*2**30/total, 1.), device)
    torch.cuda.reset_peak_memory_stats(device)
    vae_path = resolve_vae(args.vae)
    if sha256(vae_path) != metadata['vae_model_sha256']:
        raise ValueError('VAE checkpoint differs from the base cache')
    vae = _load_registered_model(str(vae_path), 'wan_video_vae', torch.bfloat16,
                                 str(device)).eval().requires_grad_(False)
    report.update(vae_sha256=sha256(vae_path), device=str(device),
                  gpu_name=torch.cuda.get_device_name(device), memory_cap_gib=args.memory_gib)
    selected = set(parse_indices(args.episodes, len(rows)))
    for row in rows:
        episode = row['episode_index']
        if episode not in selected:
            continue
        tick = time.monotonic()
        path = base/f'episodes/episode_{episode:06d}.pt'
        sidecar = json.loads(path.with_suffix('.json').read_text())
        if sha256(path) != sidecar['sha256']:
            raise ValueError('Base smoke payload checksum changed')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        frames = decision_frames(row['length'], row['exec_start_idx'])
        if (payload['frame_indices'].tolist() != frames
                or payload.get('view_contract') != metadata['view_contract']):
            raise ValueError('Base smoke payload view/clock differs')
        expected_full = payload['latents'][:, :, 0].permute(1, 0, 2, 3)[None]
        front, wrist = read_execution_rgb(dataset, info, dict(row, exec_start_idx=0))
        model = VAEOnlyModel(vae, device)
        policy = AnchorHeliosPolicy(model, prepared/'dataset_stats.json', prepared/'text_cache',
                                   num_inference_steps=1)
        instruction = 'online view parity only: action network is stubbed'
        policy._contexts[instruction] = (torch.zeros(1, 1), torch.ones(1, dtype=torch.bool))
        boundary = row['exec_start_idx']
        demo_count = sum(frame < boundary for frame in frames)
        execution = [frame for frame in frames if frame >= boundary]
        episode_report = dict(episode_index=episode, semantic_task=row['semantic_task'],
                              execution_start=boundary, demo_latents=demo_count,
                              payload_sha256=sidecar['sha256'], prefix_checks=[])
        report['episodes'].append(episode_report)
        for index, current in enumerate(execution):
            begin = 0 if index == 0 else execution[index-1]+1
            output = policy.infer(observations(front, wrist, begin, current+1,
                                               index == 0, instruction))
            stop = frames.index(current)+1
            result = compare(policy.last_prefix, expected_full[:, :, :stop])
            result.update(current_frame=current, prefix_latents=stop,
                          clock_equal=policy.observation_frame_indices == frames[:stop],
                          executed_action_shape=list(output['actions'].shape),
                          execution=compare(policy.last_prefix[:, :, demo_count:],
                                            expected_full[:, :, demo_count:stop]))
            if demo_count:
                result['demo'] = compare(policy.demo_latents, expected_full[:, :, :demo_count])
            episode_report['prefix_checks'].append(result)
            print('ONLINE_VIEW_PARITY '+json.dumps(dict(episode=episode, **result)), flush=True)
            if (not result['bitwise_equal'] or not result['clock_equal']
                    or output['actions'].shape != (16, 8)
                    or model.calls[-1]['prediction_horizon'] != 32):
                raise AssertionError(f'Online/cache prefix mismatch: episode={episode}, frame={current}')
        episode_report.update(seconds=round(time.monotonic()-tick, 3),
                              all_prefixes_bitwise_equal=True, execution_stride=16,
                              prediction_horizon_stub=32)
        del policy, model, front, wrist, payload, expected_full
        torch.cuda.empty_cache()
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                  status='passed', all_prefixes_bitwise_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default=os.getenv('ROBOMME_DATA_ROOT'))
    parser.add_argument('--prepared-root', default=os.getenv('ROBOMME_HELIOS_PREPARED'))
    parser.add_argument('--base-cache', required=True)
    parser.add_argument('--episodes', default='0,1500')
    parser.add_argument('--vae')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-gib', type=float, default=6.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not args.dataset_root or not args.prepared_root or args.memory_gib <= 0:
        parser.error('Dataset/prepared paths and a positive memory cap are required')
    torch.set_num_threads(4)
    report = dict(status='running', episodes=[], view_layout=TWO_VIEW_HORIZONTAL,
                  action_network_verified=False, benchmark_success_evaluated=False,
                  scope='real_RGB_real_VAE_online_policy_clock_and_execute16_with_stub_action_head')
    try:
        run(args, report)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(args.output, report)
    print('ONLINE_VIEW_PARITY_PASSED '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
