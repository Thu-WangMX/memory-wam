"""Optionally fill the exact complete-clip cache; training also fills it lazily.

Use a saved RoboMME config, the production environment, and matching hardware /
autocast. Different execution identities automatically use separate namespaces.
This loads only the frozen VAE, never a trainable DiT or a training checkpoint.
"""
import argparse
import json
import os
from types import MethodType, SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.robomme import episode_video
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.helpers.loader import _resolve_configs, _load_registered_model
from fastwam.memory.robomme_future_clip_cache import configure_future_clip_cache, _sha
from fastwam.utils.config_resolvers import register_default_resolvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Saved config.yaml with robomme_provenance')
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--autocast', choices=['off', 'bf16', 'fp16'], default='off')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=0, help='0 fills the complete selected shard')
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard index/count')
    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    dataset = instantiate(cfg.data.train)
    if dataset.prediction_horizon != 32:
        raise ValueError('This cache currently serves the complete 9-frame predict32 clip')
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'no': torch.float32}[str(cfg.mixed_precision)]
    _, _, asset, _ = _resolve_configs(cfg.model.model_id, cfg.model.tokenizer_model_id)
    asset.download_if_necessary()
    if _sha(asset.path) != str(cfg.robomme_provenance.vae_sha256):
        raise ValueError('VAE asset differs from the saved training provenance')
    vae = _load_registered_model(asset.path, 'wan_video_vae', torch_dtype=dtype, device=args.device)
    vae.eval().requires_grad_(False)
    model = SimpleNamespace(vae=vae, device=args.device, torch_dtype=dtype)
    model._encode_video_latents = MethodType(FastWAM._encode_video_latents, model)
    os.environ['ROBOMME_FUTURE_CLIP_CACHE_ROOT'] = args.cache_root
    amp_dtype = torch.float16 if args.autocast == 'fp16' else torch.bfloat16
    checked = written = hits = 0
    with torch.no_grad(), torch.autocast(torch.device(args.device).type, dtype=amp_dtype,
                                        enabled=args.autocast != 'off'):
        configure_future_clip_cache(model, dataset, cfg)
        cache = getattr(model, '_future_clip_cache', None)
        if cache is None:
            raise RuntimeError('Cache producer verification failed; no precomputation performed')
        print(json.dumps(dict(namespace=cache.namespace, root=str(cache.root),
                              execution=cache.contract['execution'])), flush=True)
        for index in range(args.shard_index, len(dataset), args.num_shards):
            row, start = dataset.samples[index]
            checked += 1
            if cache.get(row['episode_index'], start) is not None:
                hits += 1
            else:
                indices = [min(start + offset, row['length'] - 1) for offset in cache.contract['offsets']]
                video = episode_video(dataset.root, dataset.info, row, indices,
                                      view_layout=dataset.view_layout).unsqueeze(0)
                latent = model._encode_video_latents(video.to(device=args.device, dtype=dtype), tiled=False)
                written += int(cache.put(row['episode_index'], start, latent[0].detach().cpu()))
            if checked % 100 == 0:
                print(json.dumps(dict(checked=checked, written=written, hits=hits)), flush=True)
            if args.max_samples and checked >= args.max_samples:
                break
    print(json.dumps(dict(done=True, checked=checked, written=written, hits=hits)), flush=True)


if __name__ == '__main__':
    main()
