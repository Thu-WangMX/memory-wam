"""Audit real prepared dense prefixes/labels and original offset-zero parity.

Supports a small preparation smoke without accepting partial caches in training.
The temporary subset manifest is built outside the production cache and removed.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from fastwam.datasets.robomme import RoboMMEHeliosDataset
from fastwam.datasets.robomme_dense_cache import DenseRoboMMECache
from fastwam.datasets.robomme_past_action import RoboMMEPastActionSparse16x8FlowDataset
from fastwam.datasets.robomme_past_future import RoboMMEPastFutureActionDataset
from fastwam.memory.robomme_clock import PHASE_ANCHOR_MODE
from fastwam.memory.robomme_dense_clock import DENSE_SCHEMA, dense_prefix_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--episodes', default='0')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    ids = [int(x) for x in args.episodes.split(',')]
    root = Path(args.cache)
    metadata = json.loads((root/'config.json').read_text())['metadata']
    kwargs = dict(root=os.environ['ROBOMME_DATA_ROOT'],
                  prepared_root=os.environ['ROBOMME_HELIOS_PREPARED'],
                  full_kv_cache_path=os.environ['ROBOMME_HELIOS_LATENTS'],
                  episode_indices=ids, history_sizes=(96,2,1), anchor_frames=2,
                  prediction_horizon=32, anchor_mode=PHASE_ANCHOR_MODE)
    reference = RoboMMEHeliosDataset(**kwargs)
    report = dict(episodes=ids, checked_samples=0, offset_zero_exact=True, current_observation_exact=True)
    with tempfile.TemporaryDirectory(prefix='dense-sampling-audit-', dir=root.parent) as tmp:
        subset = Path(tmp)
        manifest = dict(schema=DENSE_SCHEMA, complete=True, metadata=metadata, episodes={})
        for ep in ids:
            entries = {}
            for offset in range(16):
                rel = f'episodes/episode_{ep:06d}/offset_{offset:02d}'
                entry = json.loads((root/(rel+'.json')).read_text())
                entries[str(offset)] = entry
                if offset:
                    dst = subset/entry['path']
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(root/entry['path'], dst)
            manifest['episodes'][str(ep)] = dict(offsets=entries)
        (subset/'manifest.json').write_text(json.dumps(manifest))
        cache = DenseRoboMMECache(subset, kwargs['root'], kwargs['full_kv_cache_path'],
                                  reference.manifest, reference.episodes)
        all_frames = sum(len(range(r['exec_start_idx'], r['length']-1)) for r in reference.episodes)
        report['all_starts'] = all_frames
        for row in reference.episodes:
            for t in range(row['exec_start_idx'], row['length']-1):
                prefix = cache.prefix(row, t, reference.load_latents(row))
                assert prefix['frame_indices'].tolist() == dense_prefix_frames(t, row['exec_start_idx'])
                assert prefix['frame_indices'][-1] == t
                if (t-row['exec_start_idx'])%16 == 0:
                    old = reference.observed_prefix(row,t)
                    torch.testing.assert_close(prefix['latents'], old['latents'], atol=0, rtol=0)
        classes = [RoboMMEHeliosDataset, RoboMMEPastActionSparse16x8FlowDataset, RoboMMEPastFutureActionDataset]
        for cls in classes:
            extra = {} if cls is RoboMMEHeliosDataset else dict(dreamdojo_labels_root=os.environ['ROBOMME_DREAMDOJO_LABELS'])
            ds = cls(**kwargs, **extra)
            ds.dense_cache = cache
            for row in ds.episodes:
                e, n = row['exec_start_idx'], row['length']
                candidates = sorted(set(list(range(e, min(e+17,n-1))) + [n-2]))
                for current in candidates:
                    ds.samples = [(row,current)]
                    sample = ds[0]
                    assert sample['frame_index'] == current
                    assert sample['action'].shape == (32,8)
                    assert sample['video'].shape == (3,9,384,320)
                    assert sample['action_is_pad'].sum().item() == max(0,current+33-n)
                    assert not sample['action_is_pad'][0]
                    if 'past_action_target' in sample:
                        assert torch.isfinite(sample['past_action_target']).all()
                        assert sample['past_action_target'].shape[-2:] == (8,64)
                    report['checked_samples'] += 1
            if cls is not RoboMMEHeliosDataset:
                # Exercise the real recipe/provenance binder without pretending
                # that a two-episode smoke cache is a finalized production cache.
                from hydra import compose, initialize_config_dir
                from experiments.robomme.train import validate_recipe
                ds.sample_stride = 1
                ds.samples = [(row, t) for row in ds.episodes
                              for t in range(row['exec_start_idx'], row['length']-1)]
                task = ('robomme_helios_past32_flow' if cls is RoboMMEPastActionSparse16x8FlowDataset
                        else 'robomme_helios_pastfuture32_flow')
                with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'), version_base='1.3'):
                    cfg = compose(config_name='train', overrides=[f'task={task}', 'wandb.enabled=false'])
                with patch('experiments.robomme.train.instantiate', return_value=ds):
                    provenance = validate_recipe(cfg)
                assert provenance['sampling']['dataset_len'] == all_frames
                assert provenance['sampling']['sample_stride'] == 1
                assert provenance['sampling']['model_memory_contract']['prediction_horizon'] == 32
                report.setdefault('preflight_contracts', []).append(task)
        report['classes'] = [cls.__name__ for cls in classes]
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
