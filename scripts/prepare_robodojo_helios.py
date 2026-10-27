"""Prepare RoboDojo statistics/text and continuous causal VAE history.

Use `assets` on CPU, then `latents` on a GPU (torchrun is supported). Sources
are read-only. Completed episode shards are resumable only with the same
dataset, image transform, VAE weights and decision contract.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from fastwam.datasets.robodojo import (
    HORIZON, RGB_STRIDE, MOSAIC, MEMORY_TASKS, dataset_metadata, episode_paths, episode_video,
    select_episodes, sha256, text_filename,
)
from fastwam.datasets.lerobot.full_kv_dataset import SCHEMA_VERSION


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def prepare_assets(args):
    _, tasks, _ = dataset_metadata(args.root)
    source = Path(args.source_prepared)
    output = Path(args.output)
    provenance = {
        'schema': 'robodojo_helios_assets_v1', 'dataset_root': str(Path(args.root).resolve()),
        'dataset_info_sha256': sha256(Path(args.root) / 'meta/info.json'),
        'source_stats_sha256': sha256(source / 'action_stats.json'),
        'mosaic': MOSAIC, 'text_padding': '128; zero invalid context; FastWAM all-visible mask',
    }
    source_paths = [source / 'text_cache' / text_filename(task) for task in tasks.values()]
    provenance['source_text_sha256'] = {p.name: sha256(p) for p in source_paths}
    if (output / 'provenance.json').exists():
        if json.loads((output / 'provenance.json').read_text()) != provenance:
            raise ValueError('Prepared output belongs to different inputs; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'text_cache').mkdir(exist_ok=True)
    stats = json.loads((source / 'action_stats.json').read_text())
    converted = {}
    for field in ('action', 'state'):
        for key in ('mean', 'std'):
            value = torch.tensor(stats[field][key])
            if value.shape != (14,) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid source {field}.{key}')
        converted[field] = {'default': {'global_' + key: value for key, value in stats[field].items()
                                       if key in ('mean', 'std', 'min', 'max', 'q01', 'q99')}}
    write_json(output / 'dataset_stats.json', converted)
    for path in source_paths:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        context, mask = payload['context'], payload['mask'].bool()
        if context.ndim != 2 or context.shape[1] != 4096 or not 0 < len(context) <= 128:
            raise ValueError(f'Invalid text context: {path}')
        if mask.shape != (len(context),):
            raise ValueError(f'Invalid text mask: {path}')
        padded = context.new_zeros((128, 4096))
        padded[:len(context)] = context
        padded_mask = torch.zeros(128, dtype=torch.bool)
        padded_mask[:len(mask)] = mask
        torch.save({'context': padded, 'mask': padded_mask}, output / 'text_cache' / path.name)
    write_json(output / 'provenance.json', provenance)
    print(json.dumps({'status': 'assets_ready', 'output': str(output), 'text_count': len(source_paths)}), flush=True)


def prepare_latents(args):
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available():
        raise RuntimeError('Latent preparation requires a GPU')
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    device = torch.device(f'cuda:{local}')
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    info, _, _ = dataset_metadata(root)
    episodes = select_episodes(root, args.tasks, split='all')
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    _, _, vae_config, _ = _resolve_configs(
        model_id='Wan-AI/Wan2.2-TI2V-5B', tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B',
        redirect_common_files=True)
    vae_config.download_if_necessary()
    contract = {
        'schema_version': SCHEMA_VERSION, 'replan_stride': HORIZON,
        'temporal_subframes': 4, 'temporal_subframe_stride': RGB_STRIDE,
        'latent_shape': [48, 1, 24, 20], 'latent_dtype': 'bfloat16',
        'mosaic': MOSAIC, 'color_contract': 'simulator_rgb_preserved',
        'encoding': 'continuous_episode_stride4_causal_vae',
        'source_format': 'robodojo_mp4_rgb', 'dataset_info_sha256': sha256(root / 'meta/info.json'),
        'dataset_episodes_sha256': sha256(root / 'meta/episodes.jsonl'),
        'vae_model_sha256': sha256(vae_config.path),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'episodes').mkdir(exist_ok=True)
    vae = None
    for row in episodes[rank::world]:
        ep, length = int(row['episode_index']), int(row['length'])
        indices = torch.arange(0, length, HORIZON)
        path = output / 'episodes' / f'episode_{ep:06d}.pt'
        parquet, videos = episode_paths(root, info, row)
        source_files = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
                        for p in [parquet, *videos]}
        if path.exists():
            old = torch.load(path, map_location='cpu', weights_only=True)
            if (old.get('contract') != contract or old.get('source_files') != source_files
                    or not torch.equal(old['frame_indices'], indices)
                    or old['latents'].shape != (len(indices), 48, 1, 24, 20)
                    or old['latents'].dtype != torch.bfloat16
                    or not torch.isfinite(old['latents']).all()):
                raise ValueError(f'Incompatible existing latent shard: {path}')
            print(f'rank={rank} episode={ep} verified_existing=true', flush=True)
            continue
        if vae is None:
            vae = _load_registered_model(vae_config.path, 'wan_video_vae',
                                         torch_dtype=torch.bfloat16, device=str(device)).eval()
        video = episode_video(root, info, row, list(range(0, int(indices[-1]) + 1, RGB_STRIDE)))
        with torch.no_grad():
            encoded = vae.encode([video.to(device=device, dtype=torch.bfloat16)], device=device, tiled=False)[0]
        latents = encoded.permute(1, 0, 2, 3).unsqueeze(2).cpu().contiguous()
        if latents.shape != (len(indices), 48, 1, 24, 20) or not torch.isfinite(latents).all():
            raise ValueError(f'Invalid encoded latents for episode {ep}: {latents.shape}')
        temporary = path.with_suffix(f'.tmp.{os.getpid()}')
        torch.save({'frame_indices': indices, 'latents': latents, 'contract': contract,
                    'source_files': source_files}, temporary)
        temporary.replace(path)
        print(f'rank={rank} episode={ep} decisions={len(indices)} status=encoded', flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        entries = {f'{root.name}/{int(row["episode_index"])}':
                   f'episodes/episode_{int(row["episode_index"]):06d}.pt' for row in episodes}
        for relative in entries.values():
            if not (output / relative).is_file():
                raise FileNotFoundError(output / relative)
        write_json(output / 'manifest.json', {'metadata': {**contract, 'complete': True,
                   'episode_count': len(entries), 'scope_tasks': args.tasks,
                   'max_episodes': args.max_episodes}, 'episodes': entries})
        print(json.dumps({'status': 'latents_ready', 'episodes': len(entries), 'output': str(output)}), flush=True)
    if world > 1:
        # Do not let another node launch training before rank 0 publishes the manifest.
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['assets', 'latents'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-prepared')
    parser.add_argument('--tasks', nargs='+', default=list(MEMORY_TASKS))
    parser.add_argument('--max-episodes', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error('--max-episodes must be positive')
    if args.mode == 'assets':
        if not args.source_prepared:
            parser.error('assets requires --source-prepared')
        prepare_assets(args)
    else:
        prepare_latents(args)


if __name__ == '__main__':
    main()
