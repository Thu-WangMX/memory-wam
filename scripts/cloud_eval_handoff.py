import json, os, pathlib, sqlite3, signal, time, fcntl

def apply_handoff(root):
    root=pathlib.Path(root)
    path=os.environ.get('RMBENCH_PERSISTENT_CONFIG')
    if not path:return
    cp=pathlib.Path(path); control=cp.parent/'handoff_request.json'
    if not control.exists():return
    request=json.loads(control.read_text());cfg=json.loads(cp.read_text())
    ack=cp.parent/'handoff_ack.json'
    with (cp.parent/'handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not ack.exists():
            dbpath=pathlib.Path(cfg['db'])
            if not dbpath.exists():raise RuntimeError('Handoff must run on original cloud node')
            groups=request['groups']
            assert all(g in cfg['groups'] for g in groups)
            with sqlite3.connect(str(dbpath),timeout=60) as db:
                db.execute('PRAGMA busy_timeout=60000');db.execute('BEGIN IMMEDIATE')
                rows=[]
                for g in groups:
                    rows.extend(dict(group=g,scene_seed=s,state=st,success=ok,detail=json.loads(d or '{}')) for s,st,ok,d in db.execute('SELECT s,state,success,detail FROM jobs WHERE g=?',(g,)))
                    done=db.execute("SELECT count(*) FROM jobs WHERE g=? AND state='done'",(g,)).fetchone()[0]
                    db.execute('UPDATE groups SET target=? WHERE g=?',(done,g))
                    db.execute("UPDATE jobs SET state='pending',token=NULL,until=NULL WHERE g=? AND state='leased'",(g,))
                    cfg['groups'][g]['target']=done
                    cfg['groups'][g]['cancelled_for_handoff']=True
                db.commit()
            tmp=cp.with_suffix('.handoff.tmp');tmp.write_text(json.dumps(cfg,indent=2));tmp.chmod(0o600);tmp.replace(cp)
            # Terminate only this campaign's processes using transferred weights.
            keys=['step_%06d'%s for s in request['steps']];pgids=set()
            for proc in pathlib.Path('/proc').iterdir():
                if not proc.name.isdigit():continue
                try:
                    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                    if cp.parent.name in cmd and any(k in cmd for k in keys) and ('eval_robotwin_single.py' in cmd or 'cloud_ninetask_persistent_rollout.py' in cmd):
                        pgids.add(os.getpgid(int(proc.name)))
                except OSError:pass
            own=os.getpgrp()
            for pgid in pgids-{own}:
                try:os.killpg(pgid,signal.SIGTERM)
                except ProcessLookupError:pass
            (cp.parent/'handoff_seed_records.json').write_text(json.dumps(rows))
            ack.write_text(json.dumps(dict(time=time.time(),groups=groups,completed=sum(x['state']=='done' for x in rows),stopped_process_groups=sorted(pgids-{own})),indent=2))
            print('CLOUD_HANDOFF_APPLIED',groups,flush=True)
    if os.environ.get('RMBENCH_QUEUE_GROUP') in request['groups']:
        raise SystemExit(0)
