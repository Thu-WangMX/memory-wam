#!/usr/bin/env python3
"""Compact status of the DreamDojo extraction pipeline."""
import json
from pathlib import Path
import time

BASE=Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme/dreamdojo_labels_v1')
CACHE=BASE/'cache/head_wrist64_stride1'


def main():
    config=json.loads((CACHE/'config.json').read_text())
    rows=[json.loads(x) for x in (Path(config['dataset_root'])/'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
    completed={int(p.stem.split('_')[-1]) for p in (CACHE/'episodes').glob('episode_*.npz')}
    frames=sum(r['length'] for r in rows if r['episode_index'] in completed)
    result={'weights_verified':(BASE/'models/verified.json').exists(),'episodes_done':len(completed),
            'episodes_total':len(rows),'frames_done':frames,'frames_total':config['total_frames'],
            'finalized':(CACHE/'manifest.json').exists(), 'pipeline_complete':(BASE/'completion.json').exists()}
    if (BASE/'workers.json').exists():
        workers=json.loads((BASE/'workers.json').read_text())
        elapsed=time.time()-workers['started_at']
        result['elapsed_seconds']=round(elapsed)
        if frames:
            result['estimated_remaining_minutes']=round((config['total_frames']-frames)/frames*elapsed/60,1)
        result['worker_processes']={str(pid):Path(f'/proc/{pid}').exists() for pid in workers['pids']}
    result['workers']=[json.loads(p.read_text()) for p in sorted((CACHE/'progress').glob('rank*.json'))]
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
