"""Re-encode the audited phase clock with front/wrist horizontal RGB views.

Kevin's RGB, original latents, text assets, and normalization remain read-only.
The new cache and rebound prepared assets are written only below our runtime-
robomme directory. GPU shards write resumable per-episode payloads; --finalize
verifies all 1600 episodes and publishes the training manifest/provenance.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil

import torch

from scripts.prepare_robomme_dense_latents import (
    atomic_json, atomic_torch_save, base_payload, encode_execution, file_lock,
    inputs, parse_indices, read_execution_rgb, resolve_vae, safe_output, sha256,
)
from fastwam.memory.robomme_clock import decision_frames
from fastwam.memory.robomme_views import (
    LEGACY_VIEW_LAYOUT, TWO_VIEW_HORIZONTAL, base_cache_schema, mosaic, view_spec,
)

RUNTIME_ROOT = Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme')


def runtime_output(path, dataset, base, source_prepared):
    output = safe_output(path, dataset, base)
    root, prepared = RUNTIME_ROOT.resolve(), Path(source_prepared).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError(f'New view artifacts must be inside our runtime directory: {root}')
    if output == prepared or output in prepared.parents or prepared in output.parents:
        raise ValueError('New view output must not overlap the original prepared assets')
    return output


def source_assets(source_prepared, dataset, base):
    prepared = Path(source_prepared)
    provenance = json.loads((prepared / 'provenance.json').read_text())
    for name, path in [('info', dataset/'meta/info.json'),
                       ('episodes', dataset/'meta/episodes.jsonl'),
                       ('tasks', dataset/'meta/tasks.jsonl'),
                       ('latents', base/'manifest.json'),
                       ('stats', prepared/'dataset_stats.json')]:
        if provenance.get(name+'_sha256') != sha256(path):
            raise ValueError(f'Original prepared provenance changed: {name}')
    for name, digest in provenance['text_sha256'].items():
        path = (prepared / 'text_cache' / name).resolve()
        if not path.is_relative_to((prepared/'text_cache').resolve()) or sha256(path) != digest:
            raise ValueError('Original text asset path/checksum differs')
    return provenance


def metadata_for_source(source, source_hash, prepared_hash):
    spec = view_spec(TWO_VIEW_HORIZONTAL)
    metadata = dict(source['metadata'])
    metadata.update(schema=base_cache_schema(TWO_VIEW_HORIZONTAL), complete=True,
                    view_layout=TWO_VIEW_HORIZONTAL, view_layout_version=spec['view_layout_version'],
                    view_contract=spec, latent_shape=spec['latent_shape'], mosaic=spec['mosaic'],
                    source_manifest_sha256=source_hash,
                    source_prepared_provenance_sha256=prepared_hash,
                    source_clock='unchanged_decision_frames_and_phase_sample_indices',
                    pixel_reencoding=True)
    return metadata


def encode_base_episode(vae, frame, row, device, *, encode_phase=encode_execution):
    """Each call to encode_phase constructs a fresh causal VAE stream.

    Keeping the exact legacy demo endpoints also keeps its final short interval.
    No encoder feature state crosses from demo to execution.
    """
    frames = decision_frames(row['length'], row['exec_start_idx'])
    demo = [index for index in frames if index < row['exec_start_idx']]
    execution = [index for index in frames if index >= row['exec_start_idx']]
    parts = [encode_phase(vae, frame, phase, device) for phase in (demo, execution) if phase]
    return frames, torch.cat(parts, dim=0)


def validate_payload(payload, row, metadata, source_digest):
    frames = decision_frames(row['length'], row['exec_start_idx'])
    expected = dict(schema=metadata['schema'], episode_index=row['episode_index'],
                    execution_start=row['exec_start_idx'], view_contract=metadata['view_contract'],
                    source_manifest_sha256=metadata['source_manifest_sha256'],
                    source_payload_sha256=source_digest)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('Two-view base payload identity/layout/provenance differs')
    if (not isinstance(payload.get('frame_indices'), torch.Tensor)
            or payload['frame_indices'].dtype != torch.int64
            or payload['frame_indices'].tolist() != frames
            or not isinstance(payload.get('latents'), torch.Tensor)
            or payload['latents'].shape != (len(frames), *metadata['latent_shape'])
            or payload['latents'].dtype != torch.bfloat16
            or not torch.isfinite(payload['latents']).all()):
        raise ValueError('Two-view base payload clock/shape/dtype differs')


def relative_file(episode):
    return f'episodes/episode_{episode:06d}.pt'


def entry_for(path, row, metadata, source_digest):
    return dict(path=relative_file(row['episode_index']), sha256=sha256(path),
                episode_index=row['episode_index'], source_payload_sha256=source_digest,
                source_manifest_sha256=metadata['source_manifest_sha256'])


def extract(args, dataset, base, output, info, rows, source, metadata):
    selected = set(parse_indices(args.episodes, len(rows)))
    vae = None
    for row in rows:
        episode = row['episode_index']
        if episode not in selected or episode % args.num_shards != args.shard_index:
            continue
        path = output / relative_file(episode)
        sidecar = path.with_suffix('.json')
        with file_lock(path.with_suffix('.lock')):
            original_path, original = base_payload(base, source, row)
            source_digest = sha256(original_path)
            del original
            if path.exists():
                validate_payload(torch.load(path, map_location='cpu', weights_only=True),
                                 row, metadata, source_digest)
                entry = entry_for(path, row, metadata, source_digest)
                if sidecar.exists() and json.loads(sidecar.read_text()) != entry:
                    raise ValueError('Existing base cache sidecar/checksum differs')
                if not sidecar.exists():
                    atomic_json(sidecar, entry)
                print(json.dumps(dict(episode=episode, status='verified_existing')), flush=True)
                continue
            if sidecar.exists():
                raise ValueError('Base cache sidecar exists without its payload')
            if vae is None:
                vae_path = resolve_vae(args.vae)
                if sha256(vae_path) != metadata['vae_model_sha256']:
                    raise ValueError('Two-view reencoding requires the exact original VAE weights')
                if str(args.device).startswith('cuda'):
                    device = torch.device(args.device)
                    torch.cuda.set_device(device)
                    if args.memory_gib:
                        total = torch.cuda.get_device_properties(device).total_memory
                        torch.cuda.set_per_process_memory_fraction(min(args.memory_gib*2**30/total, 1), device)
                from fastwam.models.wan22.helpers.loader import _load_registered_model
                vae = _load_registered_model(str(vae_path), 'wan_video_vae',
                                             torch.bfloat16, args.device).eval().requires_grad_(False)
            # Reuse the exact OpenCV decoding/color path, including the demo.
            front, wrist = read_execution_rgb(dataset, info, dict(row, exec_start_idx=0))
            @lru_cache(maxsize=32)
            def frame(index):
                return mosaic(front[index], wrist[index], TWO_VIEW_HORIZONTAL)
            frames, latents = encode_base_episode(vae, frame, row, args.device)
            payload = dict(schema=metadata['schema'], episode_index=episode,
                           execution_start=row['exec_start_idx'], frame_indices=torch.tensor(frames),
                           latents=latents, view_contract=metadata['view_contract'],
                           source_manifest_sha256=metadata['source_manifest_sha256'],
                           source_payload_sha256=source_digest)
            validate_payload(payload, row, metadata, source_digest)
            atomic_torch_save(path, payload)
            atomic_json(sidecar, entry_for(path, row, metadata, source_digest))
            frame.cache_clear()
            del front, wrist, latents, payload
            print(json.dumps(dict(episode=episode, frames=len(frames), status='written')), flush=True)


def copy_prepared(source_prepared, prepared_output, metadata, output, provenance):
    prepared_output.mkdir(parents=True, exist_ok=True)
    files = [('dataset_stats.json', provenance['stats_sha256'])]
    files.extend((f'text_cache/{name}', digest) for name, digest in provenance['text_sha256'].items())
    for name, digest in files:
        src, dst = source_prepared/name, prepared_output/name
        if sha256(src) != digest:
            raise ValueError('Source prepared asset changed during finalization')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256(dst) != digest:
            raise ValueError('Existing rebound prepared asset differs')
        if not dst.exists():
            shutil.copyfile(src, dst)
        if sha256(dst) != digest:
            raise ValueError('Rebound prepared asset checksum differs')
    rebound = dict(provenance, latents_sha256=sha256(output/'manifest.json'),
                   view_contract=metadata['view_contract'],
                   source_prepared_provenance_sha256=metadata['source_prepared_provenance_sha256'])
    path = prepared_output/'provenance.json'
    if path.exists() and json.loads(path.read_text()) != rebound:
        raise ValueError('Existing rebound provenance differs; choose a fresh output')
    atomic_json(path, rebound)


def finalize(base, output, rows, source, metadata, source_prepared, prepared_output, provenance):
    with file_lock(output/'.finalize.lock'):
        episodes, digests = {}, {}
        for row in rows:
            episode = row['episode_index']
            path = output/relative_file(episode)
            sidecar = path.with_suffix('.json')
            if not path.is_file() or not sidecar.is_file():
                raise ValueError(f'Two-view base incomplete: episode {episode}')
            original_path, original = base_payload(base, source, row)
            del original
            source_digest = sha256(original_path)
            validate_payload(torch.load(path, map_location='cpu', weights_only=True),
                             row, metadata, source_digest)
            entry = entry_for(path, row, metadata, source_digest)
            if json.loads(sidecar.read_text()) != entry:
                raise ValueError(f'Base payload sidecar/checksum differs: {episode}')
            episodes[str(episode)], digests[str(episode)] = entry['path'], entry['sha256']
        if sha256(base/'manifest.json') != metadata['source_manifest_sha256']:
            raise ValueError('Original source manifest changed during finalization')
        if sha256(source_prepared/'provenance.json') != metadata['source_prepared_provenance_sha256']:
            raise ValueError('Original prepared provenance changed during finalization')
        atomic_json(output/'manifest.json', dict(metadata=metadata, episodes=episodes,
                                               payload_sha256=digests))
        copy_prepared(source_prepared, prepared_output, metadata, output, provenance)
        print(json.dumps(dict(status='complete', episodes=len(episodes),
                              base_cache=str(output), prepared=str(prepared_output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default=os.getenv('ROBOMME_DATA_ROOT'))
    parser.add_argument('--source-base-cache', default=os.getenv('ROBOMME_HELIOS_LATENTS'))
    parser.add_argument('--source-prepared', default=os.getenv('ROBOMME_HELIOS_PREPARED'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepared-output', help='Defaults to the sibling prepared directory; written only by --finalize')
    parser.add_argument('--vae')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-gib', type=float, default=8)
    parser.add_argument('--episodes', default='all')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if not args.dataset_root or not args.source_base_cache or not args.source_prepared:
        parser.error('Provide original dataset/base-cache/prepared paths or source env.sh')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.memory_gib < 0:
        parser.error('Invalid shard or memory settings')
    torch.set_num_threads(max(1, int(os.getenv('OMP_NUM_THREADS', '4'))))
    dataset, base, prepared = [Path(x).resolve() for x in
                               (args.dataset_root, args.source_base_cache, args.source_prepared)]
    output = runtime_output(args.output, dataset, base, prepared)
    prepared_output = runtime_output(args.prepared_output or output.parent/'prepared', dataset, base, prepared)
    if output == prepared_output or output in prepared_output.parents or prepared_output in output.parents:
        raise ValueError('New base cache and prepared outputs must be separate directories')
    info, rows, source, _ = inputs(dataset, base, LEGACY_VIEW_LAYOUT)
    provenance = source_assets(prepared, dataset, base)
    metadata = metadata_for_source(source, sha256(base/'manifest.json'), sha256(prepared/'provenance.json'))
    with file_lock(output/'.initialize.lock'):
        path = output/'config.json'
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise ValueError('Existing two-view base configuration differs')
        if not path.exists():
            atomic_json(path, metadata)
    if args.finalize:
        finalize(base, output, rows, source, metadata, prepared, prepared_output, provenance)
    else:
        extract(args, dataset, base, output, info, rows, source, metadata)


if __name__ == '__main__':
    main()
