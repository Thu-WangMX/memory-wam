#!/usr/bin/env python3
"""Frozen DreamDojo labels for our LeRobot RoboMME corpus, using LTM's encoder.

Row i is E(image[i-1], image[i]), head32 followed by wrist32. Boundary
codes are retained for reproducibility, but are NOT valid transition targets.
This script never writes inside the input dataset or changes the training env.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme/dreamdojo_labels_v1')
DATA = Path('/mnt/vepfs02/output/kevin_wang/memorywam/data/robomme_all16_anchor1_event_k8_v1_full/lerobot/multitask')
CHECKPOINT_SHA = 'd77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f'
CHECKPOINT_BYTES = 8518404954
IMAGE_KEYS = ['observation.images.cam_high', 'observation.images.cam_left_wrist']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def load_reference(base):
    root = base / 'reference/LTM-pi-main'
    path = root / 'scripts/precompute_dreamdojo_memory.py'
    spec = importlib.util.spec_from_file_location('ltm_dreamdojo_extract', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, root / 'third_party/dreamdojo/runtime'


def load_dataset(ref, data):
    dataset = ref.discover_dataset(data)
    rows = [json.loads(line) for line in (data / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    metadata = {int(row['episode_index']): row for row in rows}
    for source in dataset.episodes:
        boundary = int(metadata[source.episode_index]['exec_start_idx'])
        if not 0 <= boundary < source.expected_length:
            raise ValueError(f'invalid demo/execution boundary: {source.episode_index}: {boundary}')
    return dataset, metadata


def phase_arrays(length, boundary):
    frame_index = np.arange(length, dtype=np.int64)
    phase_id = np.where(frame_index < boundary, 1, 3).astype(np.uint8)
    # LTM's phase convention: 1 demo, 2 transition into execution, 3 execution.
    if boundary > 0:
        phase_id[boundary] = 2
    valid = frame_index > 0
    if boundary > 0:
        valid[boundary] = False
    return phase_id, valid


def expected_config(args, dataset, runtime):
    ref_root = args.base / 'reference/LTM-pi-main'
    reference_files = [ref_root / 'scripts/precompute_dreamdojo_memory.py', runtime / 'dreamdojo_adapter.py']
    reference_files += sorted((runtime / 'external/lam/modules').glob('*.py'))
    return {
        'schema_version': 1, 'dataset_root': str(args.dataset),
        'dataset_fingerprint': dataset.fingerprint,
        'info_sha256': dataset.info_sha256, 'episodes_sha256': dataset.episodes_sha256,
        'total_episodes': len(dataset.episodes), 'total_frames': sum(s.expected_length for s in dataset.episodes),
        'checkpoint': str(args.base / 'models/LAM_400k.ckpt'),
        'checkpoint_sha256': CHECKPOINT_SHA, 'checkpoint_bytes': CHECKPOINT_BYTES,
        'ltm_source': 'https://github.com/buaa-colalab/LTM-pi',
        'ltm_archive_sha256': digest(args.base / 'ltm_pi_source.zip'),
        'reference_sha256': {str(p.relative_to(ref_root)): digest(p) for p in reference_files},
        'extractor_sha256': digest(__file__), 'views': IMAGE_KEYS,
        'latent_dim': 64, 'storage_dtype': 'float32', 'compute_dtype': 'bfloat16',
        'alignment': 'row[i] = concat(E(head[i-1],head[i]), E(wrist[i-1],wrist[i])); row[0]=0',
        'transition_frame_stride': 1, 'preprocess': 'LTM official_lam_video_from_raw; no augmentation',
        'latent': 'deterministic posterior mean z_mu; frozen LAM_400k',
        'phase_id': {'demo': 1, 'demo_to_execution_boundary': 2, 'execution': 3},
        'valid': 'i>0 and i!=exec_start_idx; boundary code retained but excluded from targets and normalization',
        'time_unit': 'stored observation frame index; no simulator-fps assumption',
    }


def initialize(args, ref, runtime):
    dataset, metadata = load_dataset(ref, args.dataset)
    config = expected_config(args, dataset, runtime)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config['fingerprint'] = fingerprint
    path = args.output / 'config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('cache configuration changed; use a new output directory')
    else:
        write_json(path, config)
    return dataset, metadata, config


def cache_path(output, source):
    return output / 'episodes' / f'episode_{source.episode_index:06d}.npz'


def validate_cache(path, source, row, fingerprint):
    with np.load(path, allow_pickle=False) as f:
        n = source.expected_length
        phase, valid = phase_arrays(n, int(row['exec_start_idx']))
        assert str(f['fingerprint']) == fingerprint, path
        assert int(f['episode_index']) == source.episode_index, path
        assert int(f['exec_start_idx']) == int(row['exec_start_idx']), path
        assert f['latent'].shape == (n, 64) and f['latent'].dtype == np.float32, path
        assert np.isfinite(f['latent']).all() and not f['latent'][0].any(), path
        assert np.array_equal(f['frame_index'], np.arange(n)), path
        assert np.array_equal(f['source_frame_index'], np.arange(n) - 1), path
        assert np.array_equal(f['phase_id'], phase) and np.array_equal(f['valid'], valid), path
        assert np.all(np.diff(f['global_index']) == 1), path
        return {k: f[k].copy() for k in ['latent', 'global_index', 'valid', 'phase_id', 'frame_index']}


def prepare(ref, source, dataset, row, preprocess):
    started = time.monotonic()
    frames = []
    provenance = []
    first = None
    for key in IMAGE_KEYS:
        episode = ref.read_episode(source, dataset.root, key)
        if first is None:
            first = {k: v for k, v in episode.items() if k != 'frames'}
        else:
            for name in first:
                assert np.array_equal(episode[name], first[name]), (source.episode_index, name)
        frames.append(ref.preprocess_frames(episode['frames'], preprocess, chunk_size=32))
        video = ref._episode_video_path(source, dataset.root, key)
        provenance.append({'path': str(video), 'bytes': video.stat().st_size, 'sha256': digest(video)})
        del episode
    assert np.array_equal(first['frame_index'], np.arange(source.expected_length)), source.episode_index
    first.update(lam_frames=frames, source_files=provenance, parquet_sha256=digest(source.path),
                 prepare_seconds=time.monotonic() - started)
    return first


def worker(args, ref, runtime, dataset, metadata, config):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoint = Path(config['checkpoint'])
    verified = json.loads((checkpoint.parent / 'verified.json').read_text())
    assert verified['sha256'] == CHECKPOINT_SHA and checkpoint.stat().st_size == CHECKPOINT_BYTES
    identity = ref.CheckpointIdentity(checkpoint, CHECKPOINT_SHA, CHECKPOINT_BYTES, ref.CHECKPOINT_FORMAT, -1)
    started = time.monotonic()
    model, encode, preprocess, identity = ref.load_dreamdojo_encoder(runtime, identity, 'cuda:0')
    print(json.dumps({'event': 'model_ready', 'rank': args.rank, 'load_seconds': time.monotonic()-started,
                      'checkpoint_step': identity.step, 'torch': torch.__version__}), flush=True)
    # Greedy bin packing balances the long video episodes across GPUs.
    shards = [[] for _ in range(args.world_size)]
    loads = [0] * args.world_size
    for source in sorted(dataset.episodes, key=lambda s: (-s.expected_length, s.episode_index)):
        rank = min(range(args.world_size), key=lambda r: loads[r])
        shards[rank].append(source)
        loads[rank] += source.expected_length
    sources = sorted(shards[args.rank], key=lambda s: s.episode_index)
    if args.episode_ids:
        wanted = {int(x) for x in args.episode_ids.split(',')}
        sources = [s for s in dataset.episodes if s.episode_index in wanted]
    pending = []
    for source in sources:
        path = cache_path(args.output, source)
        if path.exists():
            validate_cache(path, source, metadata[source.episode_index], config['fingerprint'])
        else:
            pending.append(source)
    if args.limit:
        pending = pending[:args.limit]
    count = 0
    frames_done = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = None
        for i, source in enumerate(pending):
            tick = time.monotonic()
            if future is None:
                future = pool.submit(prepare, ref, source, dataset, metadata[source.episode_index], preprocess)
            episode = future.result()
            future = None
            if i+1 < len(pending):
                nxt = pending[i+1]
                future = pool.submit(prepare, ref, nxt, dataset, metadata[nxt.episode_index], preprocess)
            latent_views = []
            for lam_frames in episode.pop('lam_frames'):
                latent_views.append(ref.encode_causal_latents(model, encode, lam_frames, device='cuda:0',
                                    pair_batch_size=args.batch_size, transition_frame_stride=1))
            latent = np.concatenate(latent_views, axis=1)
            row = metadata[source.episode_index]
            phase, valid = phase_arrays(len(latent), int(row['exec_start_idx']))
            arrays = {k: episode[k] for k in ['frame_index', 'global_index', 'timestamp', 'task_index']}
            arrays.update(latent=latent, source_frame_index=episode['frame_index']-1, phase_id=phase, valid=valid,
                          episode_index=np.int64(source.episode_index), exec_start_idx=np.int64(row['exec_start_idx']),
                          semantic_task=np.array(row['semantic_task']), fingerprint=np.array(config['fingerprint']),
                          source_provenance=np.array(json.dumps({'videos':episode['source_files'],
                                                                'parquet_sha256':episode['parquet_sha256']})))
            path = cache_path(args.output, source)
            ref.atomic_write_npz(path, arrays)
            validate_cache(path, source, row, config['fingerprint'])
            count += 1
            frames_done += len(latent)
            progress = {'event': 'episode_done', 'rank': args.rank, 'episode_index': source.episode_index,
                        'semantic_task': row['semantic_task'], 'frames': len(latent),
                        'valid_transitions': int(valid.sum()), 'episodes_new': count, 'episodes_pending': len(pending)-count,
                        'frames_new': frames_done, 'elapsed_seconds': time.monotonic()-started,
                        'episode_seconds': time.monotonic()-tick, 'prepare_seconds': episode['prepare_seconds'],
                        'peak_cuda_gib': torch.cuda.max_memory_allocated()/1024**3, 'updated_at': time.time()}
            write_json(args.output / 'progress' / f'rank{args.rank}.json', progress)
            print(json.dumps(progress), flush=True)
    print(json.dumps({'event':'worker_complete','rank':args.rank,'new_episodes':count,'elapsed_seconds':time.monotonic()-started}), flush=True)


def finalize(args, ref, dataset, metadata, config):
    records = []
    chunks = []
    for source in dataset.episodes:
        path = cache_path(args.output, source)
        chunk = validate_cache(path, source, metadata[source.episode_index], config['fingerprint'])
        chunk['episode_index'] = np.full(source.expected_length, source.episode_index, dtype=np.int32)
        chunks.append(chunk)
        records.append({'episode_index':source.episode_index, 'semantic_task':metadata[source.episode_index]['semantic_task'],
                        'length':source.expected_length, 'valid_transitions':int(chunk['valid'].sum()),
                        'exec_start_idx':int(metadata[source.episode_index]['exec_start_idx']),
                        'path':str(path.relative_to(args.output)), 'sha256':digest(path)})
    keys = list(chunks[0])
    arrays = {k:np.concatenate([c[k] for c in chunks]) for k in keys}
    order = np.argsort(arrays['global_index'], kind='stable')
    arrays = {k:v[order] for k,v in arrays.items()}
    assert np.array_equal(arrays['global_index'], np.arange(config['total_frames']))
    values = arrays.pop('latent')
    ref.atomic_write_npy(args.output / 'latents.npy', values)
    ref.atomic_write_npz(args.output / 'frame_index.npz', arrays)
    ref.atomic_write_jsonl(args.output / 'episodes.jsonl', records)
    stats = {}
    for name, mask in [('all',arrays['valid']), ('demo',arrays['valid'] & (arrays['phase_id']==1)),
                       ('execution',arrays['valid'] & (arrays['phase_id']==3))]:
        v = values[mask].astype(np.float64)
        mean, std = v.mean(axis=0), v.std(axis=0)
        assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>1e-6).all(), name
        stats[name] = {'count':len(v),'mean':mean.tolist(),'std':std.tolist(),'ddof':0}
    tasks = sorted(set(r['semantic_task'] for r in records))
    summary = {'status':'complete', 'fingerprint':config['fingerprint'], 'completed_at':time.time(),
               'total_episodes':len(records), 'total_frames':len(values), 'valid_transitions':int(arrays['valid'].sum()),
               'excluded_boundary_transitions':int((arrays['phase_id']==2).sum()),
               'tasks':{t:sum(r['semantic_task']==t for r in records) for t in tasks},
               'latent_shape':list(values.shape), 'normalization':stats,
               'files_sha256':{name:digest(args.output/name) for name in ['latents.npy','frame_index.npz','episodes.jsonl']}}
    write_json(args.output / 'manifest.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='normalization'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','worker','finalize','status'])
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--dataset',type=Path,default=DATA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--episode-ids')
    args = parser.parse_args()
    args.output = args.output or args.base / 'cache/head_wrist64_stride1'
    if not 0 <= args.rank < args.world_size or args.batch_size < 1:
        parser.error('invalid rank/world size/batch size')
    ref, runtime = load_reference(args.base)
    dataset, metadata, config = initialize(args, ref, runtime)
    if args.command == 'worker':
        worker(args, ref, runtime, dataset, metadata, config)
    elif args.command == 'finalize':
        finalize(args, ref, dataset, metadata, config)
    else:
        print(json.dumps({'episodes':len(dataset.episodes),'frames':config['total_frames'],
                          'cached_episodes':sum(cache_path(args.output,s).exists() for s in dataset.episodes),
                          'with_demo':sum(int(r['exec_start_idx'])>0 for r in metadata.values()),
                          'output':str(args.output),'fingerprint':config['fingerprint']}))


if __name__ == '__main__':
    main()
