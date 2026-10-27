"""Real-data action alignment, padding, balanced sampling, all-task smoke."""
import argparse
import json
import os
from pathlib import Path
from collections import Counter
import torch
from fastwam.datasets.robomme import RoboMMEHeliosDataset, TASKS
from fastwam.memory.helios_history import memory_variant_profile, select_history, episode_anchors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-variant',choices=['multiscale','multiscale_long99'],default='multiscale')
    parser.add_argument('--output',help='Override the report path (does not modify the data)')
    args=parser.parse_args()
    profile=memory_variant_profile(args.memory_variant)
    torch.set_num_threads(4)
    ds = RoboMMEHeliosDataset(os.environ['ROBOMME_DATA_ROOT'],os.environ['ROBOMME_HELIOS_PREPARED'],
                             os.environ['ROBOMME_HELIOS_LATENTS'],
                             history_sizes=profile['history_sizes'],anchor_frames=profile['anchor_frames'],
                             sample_stride=int(os.getenv('ROBOMME_SAMPLE_STRIDE', '16')),
                             dense_latents_path=os.getenv('ROBOMME_DENSE_LATENTS'))
    counts=Counter(ds.samples[i][0]['semantic_task'] for i in ds.sample_epoch_indices(42))
    assert max(counts.values())-min(counts.values()) <= 1
    report=[]
    for task in TASKS:
        ep=ds.task_episodes[task][0]
        for idx in (ds.episode_samples[ep][0],ds.episode_samples[ep][-1]):
            sample=ds[idx]
            row,t=ds.samples[idx]; table=ds._table(row)
            valid=~sample['action_is_pad']
            target=ds.norm['action'].forward(table['action'][t+1:min(t+17,row['length'])])
            torch.testing.assert_close(sample['action'][valid],target)
            torch.testing.assert_close(sample['proprio'][0],ds.norm['state'].forward(table['observation.state'][t]))
            payload=ds.observed_prefix(row,t)
            prefix=payload['latents']
            expected,expected_valid=select_history(prefix,profile['history_sizes'],args.memory_variant)
            torch.testing.assert_close(sample['history_latents'],expected)
            torch.testing.assert_close(sample['history_valid'],expected_valid)
            torch.testing.assert_close(sample['anchor_latents'],episode_anchors(prefix,2)[0])
            torch.testing.assert_close(sample['history_latents'][-1],payload['latents'][sample['decision_index']])
            assert sample['video'].shape==(3,5,384,320)
            assert sample['action'].shape==(16,8)
            report.append(dict(task=task,episode=ep,frame=t,valid_actions=int(valid.sum()),
                               observed_latents=len(prefix),retained_latents=int(expected_valid.sum())))
    result=dict(memory_variant=args.memory_variant,history_sizes=profile['history_sizes'],
                decisions=len(ds),task_sampling=dict(counts),samples=report)
    filename='alignment_check.json' if args.memory_variant=='multiscale' else 'alignment_check_long99.json'
    output=Path(args.output) if args.output else Path(os.environ['ROBOMME_HELIOS_PREPARED'])/filename
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
