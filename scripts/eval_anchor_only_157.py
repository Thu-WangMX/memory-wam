"""Anchor-only campaign: isolated checkpoints/results; shared proven dynamic queue."""
import contextlib, shutil
import argparse, datetime, fcntl, json, os, pathlib, secrets, signal, socket, sqlite3, subprocess, sys, tempfile, time, urllib.request
R=pathlib.Path(__file__).resolve().parents[1]
RUN='ninetask_fastwam-anchor_only-no-robotwininit_histaugtrue_bs4_ga4_gpu8_ep500_save5000_lr0.0001_cosine_20260929_185233'
TASKS=['observe_and_pickup','press_button','swap_T','swap_blocks','battery_try','rearrange_blocks','blocks_ranking_try','cover_blocks','put_back_block']
def main():
 a=argparse.ArgumentParser();a.add_argument('--name',required=True);a.add_argument('--steps',nargs='+',type=int,required=True);a.add_argument('--workers-per-gpu',type=int,default=2);a.add_argument('--check-only',action='store_true');a.add_argument('--auto-gpus',action='store_true');a.add_argument('--resume-records',type=pathlib.Path);a.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS);a.add_argument('--persistent-model',action='store_true');a.add_argument('--skip-groups',nargs='*',default=[]);a.add_argument('--assignment-owner');args=a.parse_args()
 if not all(c.isalnum() or c in '_-' for c in args.name):raise ValueError('Invalid name')
 assignment_file=R/'runtime/anchor_only_eval_assignments.json'
 if assignment_file.exists():
  assignments=json.loads(assignment_file.read_text())
  for owner,steps in assignments.items():
   overlap=set(args.steps)&set(steps)
   if owner!=(args.assignment_owner or args.name) and overlap:raise RuntimeError(f'Checkpoints {sorted(overlap)} are assigned to {owner}; refusing duplicate evaluation')
 if args.workers_per_gpu not in (1,2):raise ValueError('workers-per-gpu must be 1 or 2')
 v=pathlib.Path(os.environ.get('COMPARISON_VENV',str(R/'eval_a100_runtime/venv')));py=v/'bin/python'
 if not py.exists():raise FileNotFoundError(py)
 run=R/'runs'/RUN;stats=run/'dataset_stats.json';assert stats.is_file(),stats
 seedpath=R/'eval_launch_logs/ninetask_step030000_50eps_20260923_134437/seed_records.json'
 records=json.loads(seedpath.read_text());groups={}
 for s in args.steps:
  for task in args.tasks:
   seeds=sorted({int(x['scene_seed']) for x in records if x['group'].startswith(task+'_step') and x['state']=='done'})[:50]
   assert len(seeds)==50,(task,len(seeds))
   d=10 if task in ('swap_T','swap_blocks','battery_try') else 50
   env={'RMBENCH_QUEUE_YIELD_ON_WAIT':'1'}
   if task=='battery_try':env.update(RMBENCH_ACTION_EXECUTOR='dense',RMBENCH_DENSE_SUBSTEPS='15',RMBENCH_DENSE_STATE_SOURCE='drive_target')
   groups[f'{task}_step{s:06d}_d{d}']=dict(task=task,checkpoint=str(run/f'checkpoints/weights/step_{s:06d}.pt'),stats=str(stats),denoise_steps=d,seed=42,target=50,seeds=seeds,completed=[],env=env,unset_env=['RMBENCH_BATTERY_CONTACT_OFFSET','RMBENCH_BATTERY_TABLE_FRICTION'])
 for skipped in args.skip_groups:
  if skipped not in groups:raise ValueError('Unknown skipped group: '+skipped)
  del groups[skipped]
 if not groups:raise ValueError('No evaluation groups selected')
 group_assignment_file=R/'runtime/anchor_only_group_assignments.json'
 if group_assignment_file.exists():
  assigned=json.loads(group_assignment_file.read_text())
  for group in groups:
   if group in assigned and assigned[group]!=args.name:raise RuntimeError('Group assigned elsewhere: '+group)
 if args.resume_records:
  recovered=json.loads(args.resume_records.read_text())
  for group,spec in groups.items():
   rows=[x for x in recovered if x['group']==group]
   completed={int(x['scene_seed']):x for x in rows if x['state']=='done'}
   if len(completed)>spec['target']:raise ValueError('Recovered episodes exceed target: '+group)
   spec['completed']=[dict(x.get('detail') or {},scene_seed=seed,official_success=bool(x['success'])) for seed,x in sorted(completed.items())]
   spec['seeds']=sorted(set(spec['seeds'])|{int(x['scene_seed']) for x in rows})
   spec['invalid_seeds']=sorted({int(x['scene_seed']) for x in rows if x['state']=='invalid'})
  print('RECOVERED_COMPLETED',sum(len(x['completed']) for x in groups.values()),flush=True)
 for group,spec in groups.items():
  if len(spec['completed'])>=spec['target']:continue
  f=pathlib.Path(spec['checkpoint']);assert f.is_file() and f.stat().st_size>10_000_000_000,f
 if args.check_only:
  print(json.dumps({'groups':len(groups),'episodes':len(groups)*50,'steps':args.steps,'eval_venv':str(v)},indent=2));return
 subprocess.run([str(py),'-c','import torch,sapien,mplib,curobo; assert torch.cuda.device_count()==8, torch.cuda.device_count(); print("Evaluation imports OK; GPUs:",torch.cuda.device_count(),flush=True)'],check=True)
 # SAPIEN must see the NVIDIA Vulkan ICD supplied by the GPU container runtime.
 icds=list(pathlib.Path('/etc/vulkan/icd.d').glob('*nvidia*.json'))+list(pathlib.Path('/usr/share/vulkan/icd.d').glob('*nvidia*.json'))
 if not icds:raise RuntimeError('NVIDIA Vulkan ICD missing: use an evaluation image with NVIDIA graphics/Vulkan support')
 out=pathlib.Path(os.environ.get('EVAL_QUEUE_ROOT',str(R/'eval_launch_logs')))/args.name;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'manager.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 local=pathlib.Path(tempfile.mkdtemp(prefix=args.name+'_'));db=local/'seeds.sqlite'
 backup=out/'seeds.sqlite.backup'
 if backup.exists():
  with sqlite3.connect(str(backup)) as src,sqlite3.connect(str(db)) as dst:src.backup(dst);dst.execute("UPDATE jobs SET state='pending',token=NULL WHERE state='leased'");dst.commit()
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 runtime={'VK_ICD_FILENAMES':str(icds[0]),'TORCH_EXTENSIONS_DIR':os.environ.get('TORCH_EXTENSIONS_DIR',str(R/'runtime/anchor_only_157_torch_extensions')),'MAX_JOBS':'4','OMP_NUM_THREADS':'2','COMPARISON_VENV':str(v)}
 cuda=pathlib.Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))
 if not (cuda/'bin/nvcc').exists():cuda=R/'eval_a100_runtime/cuda'
 if not (cuda/'bin/nvcc').exists():raise RuntimeError('CUDA compiler unavailable')
 runtime['CUDA_HOME']=str(cuda)
 cfg=dict(persistent_model=args.persistent_model,root=str(R),output=str(out),db=str(db),port=port,token=secrets.token_hex(24),groups=groups,eval_venv=str(v),runtime_env=runtime,artifact_root=os.environ.get('EVAL_ARTIFACT_ROOT',str(R/'evaluate_results/fastwam_anchor_only_ninetask')))
 cp=out/'config.json'
 if cp.exists():
  old=json.loads(cp.read_text());assert list(old['groups'])==list(groups),'Use a new name for different checkpoints'
 cp.write_text(json.dumps(cfg,indent=2));cp.chmod(0o600)
 procs=[];coord=None
 def stop(signum,frame):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def status():
  req=urllib.request.Request(f'http://127.0.0.1:{port}/status',data=b'{}',headers={'Authorization':'Bearer '+cfg['token']})
  return json.load(urllib.request.urlopen(req,timeout=15))
 def snapshot(st):
  # Keep an independent local snapshot before publishing to the shared filesystem.
  # A transient shared-disk failure must not terminate all active evaluation workers.
  state={'updated':time.time(),'groups':st,'workers':[dict(gpu=g,slot=k,pid=p.pid,exit=p.poll()) for g,k,p in procs]}
  try:
   local_snapshot=local/'seeds.snapshot.tmp'
   with contextlib.closing(sqlite3.connect(str(db))) as src,contextlib.closing(sqlite3.connect(str(local_snapshot))) as dst:
    src.backup(dst)
   local_snapshot.replace(local/'seeds.sqlite.backup')
   (local/'manager_status.json').write_text(json.dumps(state,indent=2))
   shutil.copyfile(local/'seeds.sqlite.backup',out/'seeds.sqlite.tmp')
   with (out/'seeds.sqlite.tmp').open('rb') as handle:os.fsync(handle.fileno())
   (out/'seeds.sqlite.tmp').replace(backup)
   (out/'manager_status.tmp').write_text(json.dumps(state,indent=2));(out/'manager_status.tmp').replace(out/'manager_status.json')
   return True
  except (OSError,sqlite3.Error) as exc:
   print(f'SNAPSHOT_RETRY: {type(exc).__name__}: {exc}; live database retained at {db}; workers continue',flush=True)
   return False
 try:
  with (out/'coordinator.log').open('a') as log:coord=subprocess.Popen([sys.executable,'-u',str(R/'scripts/cloud_ninetask_eval_queue.py'),str(cp)],stdout=log,stderr=subprocess.STDOUT)
  for retry in range(30):
   try:status();break
   except Exception:
    if coord.poll() is not None:raise RuntimeError('Coordinator failed')
    time.sleep(1)
  else:raise RuntimeError('Coordinator startup timed out')
  for slot in range(0 if args.auto_gpus else args.workers_per_gpu):
   for gpu in range(8):
    with (out/f'worker_gpu{gpu}_slot{slot}.log').open('a') as log:
     p=subprocess.Popen([sys.executable,'-u',str(R/'scripts/cloud_ninetask_eval_worker.py'),str(cp),str(gpu),next(iter(groups))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    procs.append((gpu,slot,p));time.sleep(2)
  launches={};last_launch={};capacity={}
  while True:
   if args.auto_gpus:
    now=time.time()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    for row in rows:
     gpu,total,free=map(int,row.split(','));active=sum(p.poll() is None for g,k,p in procs if g==gpu)
     if not active:capacity[gpu]=args.workers_per_gpu
     if free<40960 or active>=capacity.get(gpu,1) or now-last_launch.get(gpu,0)<180 :continue
     # Launch one at a time; reserve startup time before admitting a second replica.
     if active and (free<40960 or now-last_launch.get(gpu,0)<600):continue
     slot=launches.get(gpu,0);launches[gpu]=slot+1
     with (out/f'worker_gpu{gpu}_slot{slot}.log').open('a') as log:
      p=subprocess.Popen([sys.executable,'-u',str(R/'scripts/cloud_ninetask_eval_worker.py'),str(cp),str(gpu),next(iter(groups))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     procs.append((gpu,slot,p));last_launch[gpu]=now;print('AUTO_LAUNCH',gpu,p.pid,'free_mib',free,flush=True)
   st=status();snapshot(st);done=sum(x.get('done',0) for x in st.values());print(f'{datetime.datetime.now().isoformat()} completed={done}/{len(groups)*50} live_workers={sum(p.poll() is None for _,_,p in procs)}',flush=True)
   if done==len(groups)*50:break
   if not args.auto_gpus and not any(p.poll() is None for _,_,p in procs):raise RuntimeError('All workers exited before completion; inspect worker logs')
   time.sleep(30)
 finally:
  for _,_,p in procs:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for _,_,p in procs:
   try:p.wait(timeout=25)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
  if coord is not None:
   try:snapshot(status())
   finally:coord.terminate();coord.wait(timeout=10)
if __name__=='__main__':main()
