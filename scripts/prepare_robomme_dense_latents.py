"""Prepare causal every-frame training latents outside Kevin's read-only data.

Run one shard per GPU using --shard-index/--num-shards. Episodes are decoded
once on CPU; execution offsets are encoded one at a time on the chosen GPU.
Existing payloads are verified before reuse. Run --finalize after all shards
finish: only a verified complete 1600 x 16 inventory receives manifest.json.

Example smoke (does NOT produce a complete training manifest):
  python scripts/prepare_robomme_dense_latents.py --output /our/dense-cache \
      --episodes 0 --offsets 1,7,15 --device cuda:0
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from fastwam.memory.robomme_clock import CACHE_SCHEMA, decision_frames, phase_sample_indices
from fastwam.memory.robomme_dense_clock import (
    DENSE_SCHEMA, dense_clock_contract, dense_execution_frames)
from fastwam.memory.robomme_views import (
    LEGACY_VIEW_LAYOUT, TWO_VIEW_HORIZONTAL, view_spec, base_cache_schema,
    validate_view_metadata)


LATENT_SHAPE = (48, 1, 24, 20)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_torch_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def file_lock(path):
    # Extraction runs on Linux. flock also serializes accidental overlapping
    # shards; a process crash releases the lock without a stale-lock protocol.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def parse_indices(value, upper):
    if value == 'all':
        return list(range(upper))
    result = sorted(set(int(item) for item in value.split(',')))
    if not result or result[0] < 0 or result[-1] >= upper:
        raise ValueError(f'Expected comma-separated indices in [0,{upper}) or all')
    return result


def safe_output(output, dataset, base):
    output = Path(output).resolve()
    protected = [Path(dataset).resolve(), Path(base).resolve()]
    if os.getenv('ROBOMME_KEVIN_DATA'):
        protected.append(Path(os.environ['ROBOMME_KEVIN_DATA']).resolve())
    if any('kevin' in part.lower() for part in output.parts):
        raise ValueError('Dense output must be in our namespace, never a Kevin directory')
    for path in protected:
        if output == path or path in output.parents or output in path.parents:
            raise ValueError('Dense output must not overlap the read-only source directories')
    return output


def inputs(dataset, base, view_layout=None):
    from fastwam.datasets.robomme import TASKS
    dataset, base = Path(dataset).resolve(), Path(base).resolve()
    info = json.loads((dataset / 'meta/info.json').read_text())
    rows = [json.loads(line) for line in (dataset / 'meta/episodes.jsonl').read_text().splitlines() if line]
    source = json.loads((base / 'manifest.json').read_text())
    spec = validate_view_metadata(source['metadata'], view_layout)
    expected = dict(schema=base_cache_schema(spec['view_layout']), complete=True, episode_count=1600,
                    semantic_tasks=TASKS, vae_phase_reset=True, replan_stride=16,
                    temporal_sampling='phase_local_endpoint_linspace4',
                    encoding='separate_demo_execution_causal_vae',
                    color_contract='simulator_rgb_preserved',
                    latent_shape=spec['latent_shape'], latent_dtype='bfloat16',
                    mosaic=spec['mosaic'])
    for name, value in expected.items():
        if source['metadata'].get(name) != value:
            raise ValueError(f'Incompatible base cache: {name}')
    if (info.get('total_episodes') != 1600
            or [r['episode_index'] for r in rows] != list(range(1600))
            or Counter(r['semantic_task'] for r in rows) != Counter({t:100 for t in TASKS})
            or set(source['episodes']) != set(map(str, range(1600)))):
        raise ValueError('Expected the complete audited 1600-episode source')
    metadata = dict(
        source_manifest_sha256=sha256(base / 'manifest.json'),
        vae_model_sha256=source['metadata']['vae_model_sha256'],
        info_sha256=sha256(dataset / 'meta/info.json'),
        episodes_sha256=sha256(dataset / 'meta/episodes.jsonl'),
        tasks_sha256=sha256(dataset / 'meta/tasks.jsonl'),
        encoding=expected['encoding'], temporal_sampling=expected['temporal_sampling'],
        latent_shape=spec['latent_shape'], latent_dtype='bfloat16',
        base_cache_root=str(base), dataset_root=str(dataset), episode_count=1600,
        offsets=list(range(16)), dense_clock_contract=dense_clock_contract(),
        stored_execution_anchor=False, stored_demo=False,
        offset_zero='base_cache_reference', source_cache_parity='bitwise')
    if spec['view_layout'] != LEGACY_VIEW_LAYOUT:
        metadata.update(view_layout=spec['view_layout'], view_layout_version=spec['view_layout_version'],
                        view_contract=spec, mosaic=spec['mosaic'])
    return info, rows, source, metadata


def initialize(output, metadata):
    output.mkdir(parents=True, exist_ok=True)
    config = dict(schema=DENSE_SCHEMA, metadata=metadata)
    with file_lock(output / '.initialize.lock'):
        path = output / 'config.json'
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError('Existing dense-cache configuration differs from the current source')
        else:
            atomic_json(path, config)


def relative_file(episode, offset, suffix='pt'):
    return f'episodes/episode_{episode:06d}/offset_{offset:02d}.{suffix}'


def base_payload(base, source, row):
    path = (base / source['episodes'][str(row['episode_index'])]).resolve()
    if base.resolve() not in path.parents:
        raise ValueError('Base cache payload escapes its directory')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    spec = validate_view_metadata(source['metadata'])
    if spec['view_layout'] != LEGACY_VIEW_LAYOUT and (
            payload.get('schema') != base_cache_schema(spec['view_layout'])
            or payload.get('view_contract') != spec
            or payload.get('source_manifest_sha256') != source['metadata'].get('source_manifest_sha256')
            or sha256(path) != source.get('payload_sha256', {}).get(str(row['episode_index']))):
        raise ValueError('Base payload layout/provenance/hash differs')
    frames = decision_frames(row['length'], row['exec_start_idx'])
    if (payload['frame_indices'].tolist() != frames
            or payload['execution_start'] != row['exec_start_idx']
            or payload['latents'].shape != (len(frames), *spec['latent_shape'])
            or payload['latents'].dtype != torch.bfloat16
            or not torch.isfinite(payload['latents']).all()):
        raise ValueError(f'Invalid base payload for episode {row["episode_index"]}')
    return path, payload


def validate_dense_payload(payload, row, offset, metadata):
    frames = dense_execution_frames(row['length'], row['exec_start_idx'], offset)[1:]
    expected = dict(schema=DENSE_SCHEMA, episode_index=row['episode_index'],
                    execution_start=row['exec_start_idx'], offset=offset,
                    source_manifest_sha256=metadata['source_manifest_sha256'])
    if metadata.get('view_layout', LEGACY_VIEW_LAYOUT) != LEGACY_VIEW_LAYOUT:
        spec = validate_view_metadata(metadata, dense=True)
        expected['view_contract'] = spec
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError('Dense payload identity/clock/provenance differs')
    if (not isinstance(payload.get('frame_indices'), torch.Tensor)
            or payload['frame_indices'].dtype != torch.int64
            or payload['frame_indices'].tolist() != frames
            or not isinstance(payload.get('latents'), torch.Tensor)
            or payload['latents'].shape != (len(frames), *metadata['latent_shape'])
            or payload['latents'].dtype != torch.bfloat16
            or not torch.isfinite(payload['latents']).all()):
        raise ValueError('Dense payload frame indices, shape, dtype, or values differ')


def make_entry(path, relative, row, offset, metadata, source_payload_sha256):
    count = len(dense_execution_frames(row['length'], row['exec_start_idx'], offset))
    return dict(source='base_cache' if offset == 0 else 'dense_cache', path=relative,
                frame_count=count if offset == 0 else count-1,
                sha256=sha256(path), episode_index=row['episode_index'], offset=offset,
                execution_start=row['exec_start_idx'],
                source_manifest_sha256=metadata['source_manifest_sha256'],
                source_payload_sha256=source_payload_sha256)


def validate_entry(entry, output, base, source, row, offset, metadata, source_digest):
    expected_path = source['episodes'][str(row['episode_index'])] if offset == 0 else relative_file(row['episode_index'], offset)
    root = base if offset == 0 else output
    path = (root / expected_path).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Cache entry escapes its root')
    expected = make_entry(path, expected_path, row, offset, metadata, source_digest)
    if entry != expected:
        raise ValueError(f'Existing cache entry/hash differs: episode={row["episode_index"]}, offset={offset}')
    if offset:
        validate_dense_payload(torch.load(path, map_location='cpu', weights_only=True), row, offset, metadata)


def read_execution_rgb(dataset, info, row):
    """Decode each RGB stream once; retain CPU uint8 rather than float mosaics."""
    import cv2
    from fastwam.datasets.robodojo import episode_paths
    _, paths = episode_paths(dataset, info, row)
    readers = [cv2.VideoCapture(str(path)) for path in paths[:2]]
    start, length = row['exec_start_idx'], row['length']
    try:
        if not all(reader.isOpened() for reader in readers):
            raise ValueError('Cannot open front/wrist source videos')
        first = start
        if first and not all(reader.set(cv2.CAP_PROP_POS_FRAMES, first)
                             and abs(reader.get(cv2.CAP_PROP_POS_FRAMES)-first) < .5
                             for reader in readers):
            for reader in readers:
                reader.release()
            readers = [cv2.VideoCapture(str(path)) for path in paths[:2]]
            first = 0
        frames = [[], []]
        for frame in range(first, length):
            for camera, reader in enumerate(readers):
                ok, image = reader.read()
                if not ok:
                    raise ValueError(f'Video ended before source frame {frame}')
                if frame >= start:
                    frames[camera].append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if any(len(view) != length-start for view in frames):
            raise ValueError('Execution video length differs from metadata')
        return frames
    finally:
        for reader in readers:
            reader.release()


@torch.no_grad()
def encode_execution(vae, frame, endpoints, device):
    from experiments.robomme.streaming_vae import CausalVAEStream
    stream = CausalVAEStream(vae)
    sampled = phase_sample_indices(endpoints)
    for position in range(len(endpoints)):
        indices = sampled[:1] if position == 0 else sampled[1+4*(position-1):1+4*position]
        images = torch.stack([frame(index) for index in indices], dim=1).unsqueeze(0)
        encoded = stream.append(images.to(device=device, dtype=torch.bfloat16))
    return encoded[0].permute(1,0,2,3).unsqueeze(2).cpu().contiguous()


def resolve_vae(path):
    if path:
        return Path(path).resolve()
    from fastwam.models.wan22.helpers.loader import _resolve_configs
    _, _, config, _ = _resolve_configs('Wan-AI/Wan2.2-TI2V-5B', 'Wan-AI/Wan2.1-T2V-1.3B', True)
    config.download_if_necessary()
    return Path(config.path).resolve()


def extract(args, output, dataset, base, info, rows, source, metadata):
    selected = [row for row in rows if row['episode_index'] in parse_indices(args.episodes, len(rows))
                and row['episode_index'] % args.num_shards == args.shard_index]
    offsets = parse_indices(args.offsets, 16)
    vae = None
    for row in selected:
        episode = row['episode_index']
        with file_lock(output / relative_file(episode, 0, 'lock')):
            tick = time.monotonic()
            source_path, original = base_payload(base, source, row)
            source_digest = sha256(source_path)
            pending = []
            for offset in offsets:
                sidecar = output / relative_file(episode, offset, 'json')
                if sidecar.exists():
                    validate_entry(json.loads(sidecar.read_text()), output, base, source, row, offset, metadata, source_digest)
                elif offset and (output / relative_file(episode, offset)).exists():
                    # Recover a crash after the atomic .pt rename but before its sidecar.
                    path = output / relative_file(episode, offset)
                    validate_dense_payload(torch.load(path, map_location='cpu', weights_only=True), row, offset, metadata)
                    atomic_json(sidecar, make_entry(path, relative_file(episode, offset), row, offset, metadata, source_digest))
                else:
                    pending.append(offset)
            if not pending:
                print(json.dumps(dict(episode=episode, status='verified_existing', offsets=offsets)), flush=True)
                continue
            if vae is None:
                path = resolve_vae(args.vae)
                if sha256(path) != metadata['vae_model_sha256']:
                    raise ValueError('Encoding VAE weights differ from the base observation cache')
                if str(args.device).startswith('cuda'):
                    device = torch.device(args.device)
                    torch.cuda.set_device(device)
                    if args.memory_gib:
                        total = torch.cuda.get_device_properties(device).total_memory
                        torch.cuda.set_per_process_memory_fraction(min(args.memory_gib*2**30/total, 1), device)
                from fastwam.models.wan22.helpers.loader import _load_registered_model
                vae = _load_registered_model(str(path), 'wan_video_vae', torch.bfloat16, args.device).eval().requires_grad_(False)
            front, wrist = read_execution_rgb(dataset, info, row)
            from fastwam.datasets.robomme import mosaic
            @lru_cache(maxsize=32)
            def frame(index):
                local = index-row['exec_start_idx']
                if not 0 <= local < len(front):
                    raise ValueError('Attempt to read outside the observed execution phase')
                return mosaic(front[local], wrist[local],
                              view_layout=metadata.get('view_layout', LEGACY_VIEW_LAYOUT))
            # Canonical replay verifies image layout, color, phase reset, weights,
            # precision and sampling against the already audited source cache.
            canonical = encode_execution(vae, frame, dense_execution_frames(row['length'], row['exec_start_idx']), args.device)
            demo_count = original['frame_indices'].tolist().index(row['exec_start_idx'])
            reference = original['latents'][demo_count:]
            if not torch.equal(canonical, reference):
                delta = (canonical.float()-reference.float()).abs()
                raise ValueError(f'Canonical VAE replay differs: episode={episode}, max_error={delta.max().item()}, mean_error={delta.mean().item()}')
            if 0 in pending:
                atomic_json(output / relative_file(episode, 0, 'json'), make_entry(
                    source_path, source['episodes'][str(episode)], row, 0, metadata, source_digest))
            for offset in pending:
                if offset == 0:
                    continue
                endpoints = dense_execution_frames(row['length'], row['exec_start_idx'], offset)
                values = encode_execution(vae, frame, endpoints, args.device)
                if not torch.equal(values[0], reference[0]):
                    raise ValueError(f'Dense execution anchor differs: episode={episode}, offset={offset}')
                payload = dict(schema=DENSE_SCHEMA, episode_index=episode,
                    execution_start=row['exec_start_idx'], offset=offset,
                    frame_indices=torch.tensor(endpoints[1:], dtype=torch.int64),
                    latents=values[1:].clone(), source_manifest_sha256=metadata['source_manifest_sha256'])
                if metadata.get('view_layout', LEGACY_VIEW_LAYOUT) != LEGACY_VIEW_LAYOUT:
                    payload['view_contract'] = metadata['view_contract']
                validate_dense_payload(payload, row, offset, metadata)
                path = output / relative_file(episode, offset)
                atomic_torch_save(path, payload)
                atomic_json(output / relative_file(episode, offset, 'json'), make_entry(
                    path, relative_file(episode, offset), row, offset, metadata, source_digest))
                print(json.dumps(dict(episode=episode, offset=offset, frames=len(endpoints)-1,
                                     seconds=round(time.monotonic()-tick,3), status='written')), flush=True)
            frame.cache_clear()
            del front, wrist, original, canonical, reference


def finalize(output, base, rows, source, metadata):
    with file_lock(output / '.finalize.lock'):
        missing = [(r['episode_index'], offset) for r in rows for offset in range(16)
                   if not (output / relative_file(r['episode_index'], offset, 'json')).is_file()]
        if missing:
            raise ValueError(f'Dense cache incomplete: {len(missing)} entries missing; first20={missing[:20]}')
        episodes = {}
        for row in rows:
            episode = row['episode_index']
            source_path, _ = base_payload(base, source, row)
            source_digest = sha256(source_path)
            offsets = {}
            for offset in range(16):
                entry = json.loads((output / relative_file(episode, offset, 'json')).read_text())
                validate_entry(entry, output, base, source, row, offset, metadata, source_digest)
                offsets[str(offset)] = entry
            episodes[str(episode)] = dict(offsets=offsets)
            if episode % 100 == 0:
                print(json.dumps(dict(status='finalize_verified', episodes=episode+1)), flush=True)
        # Source manifest must not change while all payloads are verified.
        if sha256(base / 'manifest.json') != metadata['source_manifest_sha256']:
            raise ValueError('Base manifest changed during finalization')
        manifest = dict(schema=DENSE_SCHEMA, complete=True, metadata=metadata, episodes=episodes)
        atomic_json(output / 'manifest.json', manifest)
        print(json.dumps(dict(status='complete', episodes=len(episodes), offsets=16,
                              manifest=str(output/'manifest.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default=os.getenv('ROBOMME_DATA_ROOT'))
    parser.add_argument('--base-cache', default=os.getenv('ROBOMME_HELIOS_LATENTS'))
    parser.add_argument('--view-layout', choices=[LEGACY_VIEW_LAYOUT, TWO_VIEW_HORIZONTAL],
                        default=os.getenv('ROBOMME_VIEW_LAYOUT', LEGACY_VIEW_LAYOUT))
    parser.add_argument('--output', required=True)
    parser.add_argument('--vae')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-gib', type=float, default=6)
    parser.add_argument('--episodes', default='all', help='all or comma-separated episode IDs')
    parser.add_argument('--offsets', default='all', help='all or comma-separated offsets 0..15')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--finalize', action='store_true', help='Verify all 1600 x 16 entries; no GPU is loaded')
    args = parser.parse_args()
    if not args.dataset_root or not args.base_cache:
        parser.error('Set --dataset-root/--base-cache or source experiments/robomme/env.sh')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.memory_gib < 0:
        parser.error('Invalid shard index/count or memory budget')
    torch.set_num_threads(max(1, int(os.getenv('OMP_NUM_THREADS', '4'))))
    dataset, base = Path(args.dataset_root).resolve(), Path(args.base_cache).resolve()
    output = safe_output(args.output, dataset, base)
    info, rows, source, metadata = inputs(dataset, base, args.view_layout)
    initialize(output, metadata)
    if args.finalize:
        finalize(output, base, rows, source, metadata)
    else:
        extract(args, output, dataset, base, info, rows, source, metadata)


if __name__ == '__main__':
    main()
