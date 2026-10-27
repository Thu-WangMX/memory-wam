"""Restart either cloud campaign with five tasks and persistent policy workers.
Stop the old cloud job in the console before invoking this entrypoint.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['observe_and_pickup','press_button','swap_T','swap_blocks','battery_try']
PLANS = {
 'late': ('cloud_ninetask_late_v1', [68500,65000,62500,60000,57500]),
 'recovery': ('cloud_ninetask_recovery_v1', [40000,35000,15000,25000,7500]),
}

def export_records(old):
    backup = old / 'seeds.sqlite.backup'
    if not backup.is_file():
        raise FileNotFoundError('Missing durable seed backup: ' + str(backup))
    records = {}
    # The manager atomically replaces this backup; open the completed snapshot.
    with sqlite3.connect('file:' + str(backup) + '?mode=ro', uri=True) as db:
        for g,s,state,success,detail in db.execute('SELECT g,s,state,success,detail FROM jobs'):
            records[(g,s)] = dict(group=g,scene_seed=s,state=state,success=success,detail=json.loads(detail or '{}'))
    # Preserve completed episodes printed after the last periodic DB snapshot.
    for log in old.glob('*.log'):
        match = re.search(r'_(observe_and_pickup|press_button|swap_T|swap_blocks|battery_try)_step_(\d+)_d(\d+)_gpu',log.name)
        if not match:
            continue
        task,step,denoise=match.groups();group=f'{task}_step{int(step):06d}_d{denoise}'
        with log.open(errors='replace') as stream:
            for line in stream:
                marker='RMBENCH_EVAL_DIAGNOSTICS '
                if marker not in line:
                    continue
                try:
                    detail=json.loads(line.split(marker,1)[1]);seed=int(detail['scene_seed']);success=bool(detail['official_success'])
                except (ValueError,KeyError):
                    continue
                existing=records.get((group,seed))
                if existing and existing['state']=='done' and bool(existing['success'])!=success:
                    raise RuntimeError(f'Conflicting results for {group} seed {seed}; review before resuming')
                records[(group,seed)]=dict(group=group,scene_seed=seed,state='done',success=int(success),detail=detail)
    return list(records.values())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',choices=PLANS);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    old_name,steps=PLANS[args.plan];name=old_name+'_five_persistent'
    old=ROOT/'eval_launch_logs'/old_name
    rows=export_records(old)
    recovered=[r for r in rows if r['state']=='done' and any(r['group'].startswith(t+'_step') for t in TASKS)]
    print('RESUME',old_name,'completed five-task episodes',len(recovered),flush=True)
    if args.check_only:
        print(json.dumps(dict(name=name,steps=steps,tasks=TASKS,persistent_model=True),indent=2));return
    out=ROOT/'eval_launch_logs'/name;out.mkdir(parents=True,exist_ok=True)
    records_path=out/'resume_records.json';records_path.write_text(json.dumps(rows))
    assignment=ROOT/'runtime/ninetask_eval_assignments.json'
    with (assignment.parent/'ninetask_assignments.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=json.loads(assignment.read_text())
        for owner,assigned in registry.items():
            if owner not in (old_name,name) and set(steps)&set(assigned):
                raise RuntimeError('Checkpoint ownership conflict: '+owner)
        registry.pop(old_name,None);registry[name]=steps
        tmp=assignment.with_suffix('.tmp');tmp.write_text(json.dumps(registry,indent=2));tmp.replace(assignment)
    (old/'retired.json').write_text(json.dumps(dict(reason='Cloud restarted as five-task persistent campaign',replacement=name,time=time.time())))
    cmd=[sys.executable,'-u',str(ROOT/'scripts/eval_ninetask_cloud.py'),'--name',name,
         '--workers-per-gpu','2','--persistent-model','--steps',*map(str,steps),
         '--tasks',*TASKS,'--resume-records',str(records_path)]
    os.execv(sys.executable,cmd)

if __name__=='__main__':
    main()
