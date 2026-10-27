"""Validate the RoboDojo memory contract before the existing FastWAM trainer."""
import json
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from fastwam.datasets.robodojo import HORIZON, RGB_STRIDE, MOSAIC, sha256
from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def validate_recipe(cfg):
    if cfg.eval_every != 0:
        raise ValueError('Generic FastWAM validation lacks historical input; keep eval_every=0 for Helios')
    if cfg.model.memory_backend != 'helios' or cfg.model.action_rope_spatial_mode != 'memorywam':
        raise ValueError('This recipe preserves the existing Anchor-Helios / memorywam RoPE contract')
    if (cfg.model.proprio_dim, cfg.model.action_dit_config.action_dim) != (14, 14):
        raise ValueError('RoboDojo arx_x5 requires 14D joint action and proprioception')
    ds = instantiate(cfg.data.train)
    info = ds.root / 'meta/info.json'
    stats = Path(cfg.data.train.pretrained_norm_stats)
    cache = Path(cfg.data.train.full_kv_cache_path) / 'manifest.json'
    source_assets = ds.prepared_root / 'provenance.json'
    if json.loads(source_assets.read_text())['dataset_info_sha256'] != sha256(info):
        raise ValueError('Prepared statistics/text belong to a different dataset')
    manifest = json.loads(cache.read_text())
    report = {
        'benchmark': 'RoboDojo', 'model': 'Anchor-Helios', 'mosaic': MOSAIC,
        'action_horizon': HORIZON, 'replan_steps': HORIZON, 'rgb_stride': RGB_STRIDE,
        'episodes': len(ds.episodes), 'decision_samples': len(ds),
        'episode_indices': [int(row['episode_index']) for row in ds.episodes],
        'dataset_info_sha256': sha256(info), 'stats_sha256': sha256(stats),
        'history_manifest_sha256': sha256(cache),
        'history_vae_sha256': manifest['metadata']['vae_model_sha256'],
        'prepared_provenance_sha256': sha256(source_assets),
        'global_batch': int(cfg.batch_size) * int(cfg.gradient_accumulation_steps) * int(os.getenv('WORLD_SIZE', '1')),
        'not_starwam_32_24_reproduction': True,
    }
    with open_dict(cfg):
        cfg.robodojo_provenance = report
    return report


@hydra.main(config_path='../../configs', config_name='train', version_base='1.3')
def main(cfg: DictConfig):
    report = validate_recipe(cfg)
    if int(os.getenv('RANK', '0')) == 0:
        print('ROBODOJO_PREFLIGHT ' + json.dumps(report), flush=True)
    if os.getenv('ROBODOJO_PREFLIGHT_ONLY') == '1':
        return
    run_training(cfg)


if __name__ == '__main__':
    main()
