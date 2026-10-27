"""One GPU supervisor, one loaded policy per active group."""
import json,os,sys,time,pathlib,subprocess,signal,urllib.request,datetime
cfg=json.load(open(sys.argv[1]));gpu=int(sys.argv[2]);preferred=sys.argv[3];R=pathlib.Path(cfg['root']);folder=pathlib.Path(cfg['output']);errors={};iteration=0
def terminate_child():
 child=globals().get('p')
 if child is None:return
 # The rollout creates its own session. Killing only the launcher's group
 # leaves it alive and can make the next replica run out of GPU memory.
 selected=[];tag=globals().get('tag')
 if tag:
  for proc in pathlib.Path('/proc').glob('[0-9]*'):
   try:
    args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    if tag in args and ('eval_policy.py' in args or 'eval_robotwin_single.py' in args):selected.append(int(proc.name))
   except OSError:pass
 for pid in selected:
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 if child.poll() is None:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
 try:child.wait(timeout=20)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
def shutdown(signum,frame):
 terminate_child()
 raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,shutdown)
signal.signal(signal.SIGINT,shutdown)
url=cfg.get('queue_url','http://127.0.0.1:'+str(cfg['port']))
def _status_once():
 req=urllib.request.Request(url+'/status',data=b'{}',headers={'Authorization':'Bearer '+cfg['token']})
 with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
def status():
 for attempt in range(6):
  try:return _status_once()
  except Exception:
   if attempt==5:raise
   time.sleep(2**attempt)
while True:
 st=status()
 eligible=list(cfg['groups'])
 unfinished=[g for g in eligible if st[g].get('done',0)<st[g].get('target',50)]
 if not unfinished:break
 active=[g for g in unfinished if st[g].get('done',0)+st[g].get('leased',0)<st[g].get('target',50) and errors.get(g,0)<3 and pathlib.Path(cfg['groups'][g]['checkpoint']).is_file()]
 if not active:
  if all(errors.get(g,0)>=3 for g in unfinished):raise RuntimeError('All remaining groups failed repeatedly; inspect logs')
  time.sleep(30);continue
 priority={v:i for i,v in enumerate(dict.fromkeys(cfg['groups'][g]['checkpoint'] for g in eligible))}
 group=preferred if preferred in active else min(active,key=lambda g:(st[g].get('leased',0),priority[cfg['groups'][g]['checkpoint']],eligible.index(g)))
 preferred=''
 spec=json.load(open(sys.argv[1]))['groups'][group];task=spec['task'];iteration+=1
 env=os.environ.copy()
 for k in list(env):
  if k.startswith('RMBENCH_') or k=='PYTHONPATH':env.pop(k,None)
 env.update(COMPARISON_VENV=cfg['eval_venv'],MEMORY_BACKEND='helios',RMBENCH_TASK=task,RMBENCH_EVAL_ROOT=str(R/'eval_runtime/RMBench'),CHECKPOINT=spec['checkpoint'],EVAL_STATS_PATH=spec['stats'],EVAL_EPISODES=str(spec['target']),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONUTF8='1',RMBENCH_ACTION_EXECUTOR='legacy',RMBENCH_QUEUE_URL=url,RMBENCH_QUEUE_TOKEN=cfg['token'],RMBENCH_QUEUE_GROUP=group)
 env.update(cfg.get('runtime_env', {}))
 if cfg.get('persistent_model'):env['RMBENCH_PERSISTENT_CONFIG']=sys.argv[1]
 env.update(spec.get('env', {}))
 # Unset optional physics overrides explicitly for default-physics groups.
 for key in spec.get('unset_env', []):env.pop(key,None)
 if task=='observe_and_pickup':env['RMBENCH_EVAL_VIDEO_CAMERA']='world_left'
 if task=='press_button':env['RMBENCH_EVAL_VIDEO_CAMERA']='table_top'
 tag=folder.name+f'_{task}_{pathlib.Path(spec["checkpoint"]).stem}_d{spec["denoise_steps"]}_gpu{gpu}_pid{os.getpid()}_attempt{iteration}';log=folder/(tag+'.log')
 cmd=['/usr/bin/bash',str(R/'scripts/eval_comparison.sh'),f'gpu_id={gpu}',f"seed={spec['seed']}",'EVALUATION.start_scene_seed=100000',f"EVALUATION.num_inference_steps={spec['denoise_steps']}",f"EVALUATION.action_horizon={spec.get('action_horizon',16)}",f"EVALUATION.replan_steps={spec.get('replan_steps',16)}",'++EVALUATION.artifact_root='+cfg.get('artifact_root',str(R/'evaluate_results/fastwam_anchor_helios_ninetask')),'EVALUATION.output_dir='+tag]
 cmd.extend(spec.get('overrides',[]))
 with log.open('w') as f:p=subprocess.Popen(cmd,env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('START',group,gpu,p.pid,str(log),flush=True)
 while p.poll() is None:
  time.sleep(15)
  snapshot=status();tmp=folder/f'status_gpu{gpu}_pid{os.getpid()}.tmp';tmp.write_text(json.dumps(snapshot,indent=2));tmp.replace(folder/f'status_gpu{gpu}_pid{os.getpid()}.json')
  text=log.read_text(errors='replace')
  idle_limit=900 if 'RMBENCH_QUEUE_CLAIM' in text else 3600
  if time.time()-log.stat().st_mtime>idle_limit:
   print('STOP stalled worker',p.pid,flush=True);terminate_child()
  if all(snapshot[g].get('done',0)>=snapshot[g].get('target',50) for g in cfg['groups'] if (cfg['groups'][g]['checkpoint']==spec['checkpoint'] if cfg.get('persistent_model') else g==group)) and time.time()-log.stat().st_mtime>120:
   terminate_child()
 rc=p.wait();print('EXIT',group,rc,flush=True)
 if rc:errors[group]=errors.get(group,0)+1;time.sleep(20)
 else:preferred=''
print('SUPERVISOR_FINISHED',json.dumps(status()),flush=True)
