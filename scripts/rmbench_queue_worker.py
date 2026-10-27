"""One GPU supervisor, one loaded policy per active group."""
import json,os,sys,time,pathlib,subprocess,signal,urllib.request,datetime
cfg=json.load(open(sys.argv[1]));gpu=int(sys.argv[2]);preferred=sys.argv[3];R=pathlib.Path(cfg['root']);folder=pathlib.Path(cfg['output']);errors={};iteration=0
url='http://127.0.0.1:'+str(cfg['port'])
def status():
 req=urllib.request.Request(url+'/status',data=b'{}',headers={'Authorization':'Bearer '+cfg['token']})
 with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
while True:
 st=status()
 eligible=list(cfg['groups'])
 if cfg.get('denoise50_then_top2',False):
  fifty=[g for g in eligible if cfg['groups'][g]['denoise_steps']==50]
  unfinished=[g for g in fifty if st[g].get('done',0)<st[g].get('target',100)]
  if unfinished:
   eligible=unfinished
  else:
   ranked=sorted(fifty,key=lambda g:(st[g].get('success',0)/max(st[g].get('done',0),1),cfg['groups'][g]['checkpoint']),reverse=True)
   best={cfg['groups'][g]['checkpoint'] for g in ranked[:2]}
   eligible=[g for g in eligible if cfg['groups'][g]['denoise_steps']==10 and cfg['groups'][g]['checkpoint'] in best]
   selection={'ranked_50':ranked,'selected_10':eligible,'tie_break':'higher checkpoint step'}
   tmp=folder/f'selection_gpu{gpu}.tmp';tmp.write_text(json.dumps(selection,indent=2));tmp.replace(folder/f'selection_gpu{gpu}.json')
 active=[g for g in eligible if st[g].get('done',0)<st[g].get('target',100) and errors.get(g,0)<3]
 if not active:
  if any(st[g].get('done',0)<st[g].get('target',100) for g in eligible):
   print('WAIT eligible groups incomplete but local retry limit reached',flush=True);time.sleep(60);continue
  break
 if cfg.get('checkpoint_ordered',False):
  first=cfg['groups'][active[0]]['checkpoint']
  active=[g for g in active if cfg['groups'][g]['checkpoint']==first]
 group=preferred if preferred in active else min(active,key=lambda g:st[g].get('leased',0))
 spec=cfg['groups'][group];task=spec['task'];iteration+=1
 env=os.environ.copy()
 for k in list(env):
  if k.startswith('RMBENCH_') or k=='PYTHONPATH':env.pop(k,None)
 env.update(COMPARISON_VENV='/root/kevin_wang/envs/fastwam_guidemem',MEMORY_BACKEND='helios',RMBENCH_TASK=task,RMBENCH_EVAL_ROOT=str(R/'eval_runtime/RMBench'),CHECKPOINT=spec['checkpoint'],EVAL_STATS_PATH=spec['stats'],EVAL_EPISODES='100',CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONUTF8='1',RMBENCH_ACTION_EXECUTOR='legacy',RMBENCH_QUEUE_URL=url,RMBENCH_QUEUE_TOKEN=cfg['token'],RMBENCH_QUEUE_GROUP=group)
 env.update(cfg.get('runtime_env', {}))
 env.update(spec.get('env', {}))
 # Unset optional physics overrides explicitly for default-physics groups.
 for key in spec.get('unset_env', []):env.pop(key,None)
 if task=='observe_and_pickup':env['RMBENCH_EVAL_VIDEO_CAMERA']='world_left'
 if task=='press_button':env['RMBENCH_EVAL_VIDEO_CAMERA']='table_top'
 tag=folder.name+f'_{group}_gpu{gpu}_attempt{iteration}';log=folder/(tag+'.log')
 cmd=['/usr/bin/bash',str(R/'scripts/eval_comparison.sh'),f'gpu_id={gpu}',f"seed={spec['seed']}",'EVALUATION.start_scene_seed=100000',f"EVALUATION.num_inference_steps={spec['denoise_steps']}",'EVALUATION.action_horizon=16','EVALUATION.replan_steps=16','++EVALUATION.artifact_root='+str(R/'evaluate_results/fastwam_anchor_helios_v1'),'EVALUATION.output_dir='+tag]
 cmd.extend(spec.get('overrides',[]))
 with log.open('w') as f:p=subprocess.Popen(cmd,env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('START',group,gpu,p.pid,str(log),flush=True)
 while p.poll() is None:
  time.sleep(15)
  snapshot=status();tmp=folder/f'status_gpu{gpu}.tmp';tmp.write_text(json.dumps(snapshot,indent=2));tmp.replace(folder/f'status_gpu{gpu}.json')
  if time.time()-log.stat().st_mtime>900:
   print('STOP stalled worker',p.pid,flush=True);os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
  if snapshot[group].get('done',0)>=snapshot[group].get('target',100) and time.time()-log.stat().st_mtime>120:
   os.killpg(p.pid,signal.SIGTERM)
 rc=p.wait();print('EXIT',group,rc,flush=True)
 if rc:errors[group]=errors.get(group,0)+1;time.sleep(20)
 else:preferred=''
print('SUPERVISOR_FINISHED',json.dumps(status()),flush=True)
