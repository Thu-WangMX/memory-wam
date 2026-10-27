"""Wait for an existing queue to finish, then launch a staged evaluation campaign."""
import json, os, pathlib, subprocess, sys, time, urllib.request
cfgpath=pathlib.Path(sys.argv[1]); cfg=json.loads(cfgpath.read_text()); root=pathlib.Path(cfg['root']); out=pathlib.Path(cfg['output'])
def publish(phase, **kw):
    p=out/'waiter_status.json'; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(phase=phase,updated=time.time(),**kw),indent=2));tmp.replace(p)
def rpc(c):
    req=urllib.request.Request('http://127.0.0.1:'+str(c['port'])+'/status',data=b'{}',headers={'Authorization':'Bearer '+c['token']})
    return json.load(urllib.request.urlopen(req,timeout=15))
while True:
    try:
        oldmeta=json.loads(pathlib.Path(cfg['wait_for_manifest']).read_text())
        oldcfg=json.loads(pathlib.Path(oldmeta['worker_config_path']).read_text())
        st=rpc(oldcfg)
        done={g:st[g].get('done',0) for g in cfg['wait_for_groups']}
        if all(v>=100 for v in done.values()):break
        publish('waiting_previous_evaluations',completed=done)
    except Exception as exc:publish('waiting_previous_queue_unavailable',error=str(exc))
    time.sleep(30)
# Completed episodes can precede process cleanup; wait for sufficient free memory.
while True:
    try:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        free=[int(x) for x in rows.splitlines()]
        gpus=[i for i,x in enumerate(free) if x>=38000]
        if len(gpus)>=cfg.get('required_gpus',8):break
        publish('waiting_gpu_memory',free_mib=free)
    except Exception as exc:publish('waiting_gpu_query',error=str(exc))
    time.sleep(30)
def launch(args,name):
    with (out/(name+'.log')).open('a') as f:
        return subprocess.Popen(['nohup',sys.executable,'-u',str(root/'scripts/run_with_signal_log.py'),str(out/(name+'_signals.jsonl')),sys.executable,'-u',*args],cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True).pid
coord=launch([str(root/'scripts/rmbench_staged_queue.py'),str(cfgpath)],'coordinator')
for attempt in range(30):
    try:rpc(cfg);break
    except Exception:time.sleep(1)
else:
    publish('coordinator_start_failed',pid=coord);raise RuntimeError('Coordinator startup failed')
snapshot=launch([str(root/'scripts/rmbench_queue_snapshot.py'),str(cfgpath)],'snapshot')
groups=[g for g,v in cfg['groups'].items() if v['denoise_steps']==50];workers=[]
for i,gpu in enumerate(gpus):
    group=groups[i%len(groups)];pid=launch([str(root/'scripts/rmbench_queue_worker.py'),str(cfgpath),str(gpu),group],f'supervisor_gpu{gpu}')
    workers.append(dict(gpu=gpu,group=group,pid=pid))
publish('launched',coordinator_pid=coord,snapshot_pid=snapshot,workers=workers)
