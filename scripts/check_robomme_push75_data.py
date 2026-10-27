"""Real two-episode push75 smoke; NEVER produces a production training manifest.

This harness copies already encoded RGB base payloads into a separate test root,
extracts all 16 execution offsets there, and exercises the strict cache reader.
The production dataset is first constructed against the original audited 1600
episodes. Test-only cache rebinding then permits checking a two-episode subset
without weakening the production constructor or claiming a complete corpus.
"""
import argparse
from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from fastwam.datasets.robomme_dense_cache import DenseRoboMMECache
from fastwam.datasets.robomme_past_future import RoboMMEPastFutureActionDataset
from fastwam.memory.robomme_clock import PHASE_ANCHOR_MODE
from fastwam.memory.robomme_dense_clock import DENSE_SCHEMA, dense_prefix_frames, execution_offset
from fastwam.memory.robomme_views import TWO_VIEW_HORIZONTAL, mosaic, validate_view_metadata
from scripts.prepare_robomme_dense_latents import (
    atomic_json, extract, initialize, inputs, read_execution_rgb, relative_file,
    safe_output, sha256, validate_entry,
)


def build_partial_base(args, rows, metadata, work):
    """Independent copy: the original smoke base receives no manifest or writes."""
    source = Path(args.smoke_base).resolve()
    base = work/'base'
    base.mkdir(parents=True, exist_ok=True)
    manifest = dict(metadata=dict(metadata, complete=False, episode_count=len(rows),
                    partial_smoke_only=True, expected_production_episode_count=1600),
                    episodes={}, payload_sha256={}, partial_smoke_only=True)
    for row in rows:
        ep = row['episode_index']
        name = f'episodes/episode_{ep:06d}.pt'
        original, copied = source/name, base/name
        digest = sha256(original)
        copied.parent.mkdir(parents=True, exist_ok=True)
        if copied.exists():
            if sha256(copied) != digest:
                raise ValueError('Existing smoke copy differs; choose another test root')
        else:
            shutil.copyfile(original, copied)
        manifest['episodes'][str(ep)] = name
        manifest['payload_sha256'][str(ep)] = digest
    atomic_json(base/'manifest.json', manifest)
    return base, manifest


def extract_partial_offsets(args, rows, info, base, source, old_metadata, work):
    metadata = dict(old_metadata,
        source_manifest_sha256=sha256(base/'manifest.json'),
        base_cache_root=str(base), episode_count=len(rows), partial_smoke_only=True,
        expected_production_episode_count=1600)
    view = validate_view_metadata(source['metadata'])
    metadata.update(view_layout=TWO_VIEW_HORIZONTAL, view_layout_version=view['view_layout_version'],
                    view_contract=view, latent_shape=view['latent_shape'], mosaic=view['mosaic'])
    output = work/'dense'
    initialize(output, metadata)
    # Explicit test-only helper invocation. Official inputs()/--finalize still
    # require all 1600 episodes and are not bypassed by any production entry.
    extract_args = SimpleNamespace(episodes='all', offsets='all', num_shards=1,
                                  shard_index=0, vae=args.vae, device=args.device,
                                  memory_gib=args.memory_gib)
    # parse_indices in extract expects corpus-sized IDs. Keep the official rows
    # list but select only the two requested IDs for this isolated test run.
    extract_args.episodes = ','.join(str(row['episode_index']) for row in rows)
    full_rows = args.full_rows
    extract(extract_args, output, Path(args.dataset_root), base, info, full_rows, source, metadata)
    episodes = {}
    for row in rows:
        ep = row['episode_index']
        digest = source['payload_sha256'][str(ep)]
        entries = {}
        for offset in range(16):
            entry = json.loads((output/relative_file(ep, offset, 'json')).read_text())
            validate_entry(entry, output, base, source, row, offset, metadata, digest)
            entries[str(offset)] = entry
        episodes[str(ep)] = dict(offsets=entries)
    # complete refers ONLY to this explicitly marked two-episode test inventory.
    # Its base metadata.complete=False and episode_count=2 forbid production use.
    manifest = dict(schema=DENSE_SCHEMA, complete=True, metadata=metadata,
                    episodes=episodes, partial_smoke_only=True)
    atomic_json(output/'manifest.json', manifest)
    cache = DenseRoboMMECache(output, args.dataset_root, base, source, rows)
    return cache, view


def bind_test_dataset(args, rows, base, manifest, cache, view):
    # This executes the unchanged strict production constructor on the real
    # original data, prepared assets, and complete DreamDojo label inventory.
    dataset = RoboMMEPastFutureActionDataset(root=args.dataset_root,
        prepared_root=args.source_prepared, full_kv_cache_path=args.source_base,
        dreamdojo_labels_root=args.labels_root, episode_indices=[r['episode_index'] for r in rows],
        history_sizes=(72, 2, 1), anchor_frames=2, past_query_slots=32,
        prediction_horizon=32, anchor_mode=PHASE_ANCHOR_MODE, sample_stride=16)
    # TEST ONLY: explicit partial-cache injection after strict constructor success.
    dataset.cache_root, dataset.manifest = base, manifest
    dataset.view_layout, dataset.view_contract = TWO_VIEW_HORIZONTAL, view
    dataset.latent_shape = tuple(view['latent_shape'])
    dataset.dense_cache, dataset.sample_stride = cache, 1
    dataset._latents, dataset._verified_view_payloads = OrderedDict(), set()
    dataset.samples = [(row, t) for row in rows
                       for t in range(row['exec_start_idx'], row['length']-1)]
    return dataset


def verify_real_samples(dataset, rows, work):
    summary = []
    samples_by_key = {(row['episode_index'], current): i
                      for i, (row, current) in enumerate(dataset.samples)}
    for row in rows:
        ep, boundary, length = row['episode_index'], row['exec_start_idx'], row['length']
        starts = list(range(boundary, length-1))
        base = dataset.load_latents(row)
        for current in starts:
            prefix = dataset.observed_prefix(row, current)
            frames = dense_prefix_frames(current, boundary)
            assert prefix['frame_indices'].tolist() == frames
            assert prefix['latents'].shape == (len(frames), 48, 1, 16, 32)
            assert prefix['latents'].dtype == torch.bfloat16
            assert torch.isfinite(prefix['latents']).all()
            assert frames[-1] == current and max(frames) <= current
            # The true execution anchor is invariant across all offset streams.
            torch.testing.assert_close(prefix['latents'][frames.index(boundary)],
                base['latents'][base['frame_indices'].tolist().index(boundary)], atol=0, rtol=0)
        selected = {starts[-1]}
        for offset in range(16):
            available = [t for t in starts if execution_offset(t, boundary) == offset]
            if available:
                selected.add(available[len(available)//2])
        front, wrist = read_execution_rgb(dataset.root, dataset.info, row)
        table = dataset._table(row)
        for current in sorted(selected):
            sample = dataset[samples_by_key[ep, current]]
            assert sample['video'].shape == (3, 9, 256, 512)
            assert sample['history_latents'].shape == (75, 48, 1, 16, 32)
            assert sample['past_action_target'].shape == (34, 8, 64)
            assert sample['action'].shape == (32, 8)
            assert sample['past_action_visual_slots'][-2:].tolist() == [75, 76]
            frames = dense_prefix_frames(current, boundary)
            assert sample['history_valid'].sum() == min(len(frames), 75)
            assert sample['anchor_valid'].tolist() == [boundary > 0, True]
            torch.testing.assert_close(sample['anchor_latents'][1],
                base['latents'][base['frame_indices'].tolist().index(boundary)], atol=0, rtol=0)
            if boundary:
                torch.testing.assert_close(sample['anchor_latents'][0], base['latents'][0], atol=0, rtol=0)
            # Independently compute destinations: observation t predicts action t+1.
            destinations = torch.arange(current+1, current+33)
            expected_action = dataset.norm['action'].forward(table['action'][destinations.clamp(max=length-1)])
            torch.testing.assert_close(sample['action'], expected_action, atol=0, rtol=0)
            assert torch.equal(sample['action_is_pad'], destinations >= length)
            rgb_indices = torch.arange(current, current+33, 4)
            assert torch.equal(sample['image_is_pad'], rgb_indices >= length)
            # Compare actual seek-decoded training RGB to the same online mosaic
            # applied to independently decoded original execution RGB frames.
            for j, index in enumerate(rgb_indices.clamp(max=length-1).tolist()):
                online = mosaic(front[index-boundary], wrist[index-boundary], TWO_VIEW_HORIZONTAL)
                torch.testing.assert_close(sample['video'][:, j], online, atol=0, rtol=0)
            label_frames = sample['past_action_transition_frames']
            visible, loss_valid = sample['past_action_transition_valid'], sample['past_action_loss_valid']
            assert (label_frames[:32][visible[:32]] <= current).all()
            assert visible[32:].all() and sample['past_action_valid'][32:].all()
            assert torch.equal(loss_valid, visible & (label_frames < length))
            assert (label_frames[loss_valid] != boundary).all()
            offset, _ = dataset.dreamdojo.offsets[ep]
            indices = label_frames[loss_valid].numpy()
            teacher = (np.asarray(dataset.dreamdojo._latents[offset+indices])
                       - dataset.dreamdojo.mean) / dataset.dreamdojo.std
            torch.testing.assert_close(sample['past_action_target'][loss_valid],
                                       torch.from_numpy(teacher), atol=0, rtol=0)
            assert not sample['past_action_target'][~loss_valid].count_nonzero()
            assert torch.isfinite(sample['past_action_target']).all()
            if current == length-2:
                assert sample['action_is_pad'].sum() == 31
                assert not loss_valid[32:].any()
        del front, wrist
        summary.append(dict(episode_index=ep, semantic_task=row['semantic_task'],
            execution_start=boundary, length=length, prefixes_checked=len(starts),
            offsets_checked=16, getitem_frames=sorted(selected),
            getitem_count=len(selected), status='passed'))
        print(json.dumps(summary[-1]), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default=os.getenv('ROBOMME_DATA_ROOT'))
    parser.add_argument('--source-base', default=os.getenv('ROBOMME_HELIOS_LATENTS'))
    parser.add_argument('--source-prepared', default=os.getenv('ROBOMME_HELIOS_PREPARED'))
    parser.add_argument('--labels-root', default=os.getenv('ROBOMME_DREAMDOJO_LABELS',
        '/mnt/vepfs01/output/spidy.wang/runtime-robomme/dreamdojo_labels_v1/cache/head_wrist64_stride1'))
    parser.add_argument('--smoke-base', required=True)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--episodes', default='0,1500')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-gib', type=float, default=6)
    parser.add_argument('--vae')
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(os.getenv('OMP_NUM_THREADS', '4'))))
    started = time.monotonic()
    work = safe_output(args.work_root, args.dataset_root, args.smoke_base)
    if work == Path(args.source_base).resolve() or Path(args.source_base).resolve() in work.parents:
        raise ValueError('Smoke work root cannot modify the production source cache')
    info, full_rows, _, old_metadata = inputs(Path(args.dataset_root), Path(args.source_base))
    args.full_rows = full_rows
    wanted = set(map(int, args.episodes.split(',')))
    rows = [row for row in full_rows if row['episode_index'] in wanted]
    if len(rows) != 2 or len(wanted) != 2:
        raise ValueError('This harness is intentionally restricted to two actual episodes')
    metadata = json.loads((Path(args.smoke_base)/'config.json').read_text())
    validate_view_metadata(metadata, TWO_VIEW_HORIZONTAL)
    original_manifest_absent = not (Path(args.smoke_base)/'manifest.json').exists()
    base, manifest = build_partial_base(args, rows, metadata, work)
    cache, view = extract_partial_offsets(args, rows, info, base, manifest, old_metadata, work)
    dataset = bind_test_dataset(args, rows, base, manifest, cache, view)
    results = verify_real_samples(dataset, rows, work)
    if original_manifest_absent:
        assert not (Path(args.smoke_base)/'manifest.json').exists()
    report = dict(status='passed', scope='two_real_episodes_only', partial_smoke_only=True,
        production_training_cache_ready=False, base_manifest_production_complete=False,
        test_only_dataset_cache_rebinding=True, view_contract=view,
        model_data=dict(history_sizes=[72, 2, 1], past_query_slots=32, future_query_slots=2,
                        prediction_horizon=32, execution_horizon=16, sample_stride=1,
                        anchor_mode=PHASE_ANCHOR_MODE),
        real_gpu_dense_offset_extraction=True, memory_cap_gib=args.memory_gib,
        independent_rgb_seek_and_online_mosaic_bitwise=True,
        episodes=results, elapsed_seconds=round(time.monotonic()-started, 3),
        smoke_work_root=str(work), original_smoke_base_manifest_unchanged=True)
    atomic_json(Path(args.report), report)
    print(json.dumps(dict(status='passed', report=args.report,
                         elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
