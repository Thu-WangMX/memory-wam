"""Three-node top-checkpoint evaluation, one authoritative local SQLite queue."""
import pathlib,json,os,sys,time,sqlite3,subprocess,signal,socket,secrets,fcntl,urllib.request,threading,hashlib
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
R=pathlib.Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')
NAME='ninetask_top100_20260927'
OUT=R/'eval_launch_logs'/NAME
URL='http://192.168.52.208:39875'
RUN=R/'runs/ninetask_fastwam-anchor-helios-no-robotwininit_histaugtrue_bs16_ga1_gpu8_ep500_save2500_lr0.0001_cosine_20260922_050552_resume-step_002500'
PLAN={'observe_and_pickup':[15000],'rearrange_blocks':[68500,40000],'put_back_block':[68500,42500],'swap_blocks':[67500,65000],'blocks_ranking_try':[42500,32500],'swap_T':[67500,32500,60000,40000],'cover_blocks':[42500,47500],'press_button':[62500,67500]}
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def prepare():
 OUT.mkdir(exist_ok=True)
 if (OUT/'config.json').exists():raise RuntimeError('Already prepared')
 summary=json.loads((R/'runtime/ninetask_all_checkpoint_summary_20260927.json').read_text());groups={};evidence={}
 for task,steps in PLAN.items():
  for step in steps:
   d=10 if task in ('swap_T','swap_blocks') else 50;g=f'{task}_step{step:06d}_d{d}'
   row=next(x for x in summary[str(step)][task] if x['denoise']==d)
   assert row['done']==50,(g,row)
   source=R/row['source'];cfg=json.loads((source/'config.json').read_text());spec=dict(cfg['groups'][g])
   if (source/'seeds.sqlite.backup').exists():
    with sqlite3.connect('file:'+str(source/'seeds.sqlite.backup')+'?mode=ro',uri=True) as db:
     jobs=list(db.execute('SELECT s,state,success,detail FROM jobs WHERE g=?',(g,)))
   else:
    exported=json.loads((source/'seed_records.json').read_text())
    jobs=[(x['scene_seed'],x['state'],x.get('success'),json.dumps(x.get('detail') or {})) for x in exported if x['group']==g]
   done={s:dict(json.loads(detail or '{}'),scene_seed=s,official_success=bool(ok)) for s,st,ok,detail in jobs if st=='done'}
   assert len(done)==50 and sum(x['official_success'] for x in done.values())==row['success']
   spec.update(target=100,completed=list(done.values()),seeds=sorted({s for s,_,_,_ in jobs}|set(spec['seeds'])),invalid_seeds=[s for s,st,_,_ in jobs if st=='invalid'])
   spec.pop('cancelled_for_handoff',None);groups[g]=spec;evidence[g]=dict(source=str(source),done=50,success=row['success'],seeds=sorted(done))
 cfg=dict(root=str(R),output=str(OUT),db='/tmp/'+NAME+'/seeds.sqlite',queue_url=URL,port=39875,token=secrets.token_hex(24),groups=groups)
 atomic(OUT/'config.json',cfg);os.chown(OUT/'config.json',0,1397203185);(OUT/'config.json').chmod(0o640)
 atomic(OUT/'baseline_evidence.json',evidence)
 reg=R/'runtime/ninetask_group_assignments.json'
 with (R/'runtime/ninetask_assignments.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);v=json.loads(reg.read_text());v.update({g:NAME for g in groups});atomic(reg,v)
 print('Prepared',len(groups),'groups; baseline',sum(len(s['completed']) for s in groups.values()),'target',sum(s['target'] for s in groups.values()))
def coordinator():
 sys.path.insert(0,str(R/'scripts'));from cloud_ninetask_eval_queue import Queue
 cfg=json.loads((OUT/'config.json').read_text());local=pathlib.Path(cfg['db']);local.parent.mkdir(exist_ok=True)
 lock=(OUT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 backup=OUT/'seeds.sqlite.backup'
 if not local.exists() and backup.exists():
  with sqlite3.connect(str(backup)) as a,sqlite3.connect(str(local)) as b:a.backup(b)
 q=Queue(str(local),cfg['groups']);stop=threading.Event()
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_POST(self):
   if self.headers.get('Authorization')!='Bearer '+cfg['token']:self.send_error(403);return
   try:
    a=json.loads(self.rfile.read(int(self.headers['Content-Length'])));v=q.call(self.path.strip('/'),a);raw=json.dumps(v).encode();self.send_response(200);self.end_headers();self.wfile.write(raw)
   except Exception as e:self.send_error(400,str(e))
 server=ThreadingHTTPServer(('192.168.52.208',cfg['port']),Handler)
 threading.Thread(target=server.serve_forever,daemon=True).start()
 signal.signal(signal.SIGTERM,lambda *a:stop.set());signal.signal(signal.SIGINT,lambda *a:stop.set())
 def snapshot():
  st=q.call('status',{})
  with q.lock,sqlite3.connect(str(OUT/'seeds.sqlite.tmp')) as dst:q.db.backup(dst)
  (OUT/'seeds.sqlite.tmp').replace(backup)
  atomic(OUT/'manager_status.json',dict(updated=time.time(),groups=st,workers=[]))
  return st
 print('COORDINATOR_READY',URL,flush=True)
 while not stop.is_set():
  st=snapshot();print('PROGRESS',sum(v.get('done',0) for v in st.values()),'/' + str(sum(v['target'] for v in st.values())),flush=True)
  # Keep serving briefly after completion so remote supervisors can shut down.
  if all(v.get('done',0)>=v['target'] for v in st.values()):
   atomic(OUT/'complete.json',dict(time=time.time(),groups=st));stop.wait(600);break
  stop.wait(15)
 snapshot();server.shutdown()
def restore():
 import tos
 creds=json.loads((R/'runtime/tos_eval_archive/credentials.json').read_text())
 os.environ['NO_PROXY']=os.environ['no_proxy']='localhost,127.0.0.1,.tos-cn-beijing.ivolces.com'
 client=tos.TosClientV2(creds['TOS_ACCESS_KEY'],creds['TOS_SECRET_KEY'],'tos-cn-beijing.ivolces.com','cn-beijing')
 cfg=json.loads((OUT/'config.json').read_text())
 steps=sorted({int(pathlib.Path(s['checkpoint']).stem.split('_')[1]) for s in cfg['groups'].values()})
 for step in steps:
  dst=RUN/f'checkpoints/weights/step_{step:06d}.pt'
  if dst.exists():continue
  receipt=json.loads((R/f'runtime/tos_eval_archive/step_{step:06d}.json').read_text());meta=receipt['remote_manifest']['files']['weights.pt']
  key=creds['TOS_PREFIX'].strip('/')+'/'+RUN.name+'/checkpoints/'+str(step).zfill(7)+'/weights.pt'
  tmp=dst.with_suffix('.restore.part');print('RESTORE_START',step,flush=True)
  client.download_file(creds['TOS_BUCKET'],key,str(tmp),task_num=8,part_size=64*1024*1024,enable_checkpoint=True)
  h=hashlib.sha256()
  with tmp.open('rb') as f:
   for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
  assert tmp.stat().st_size==meta['size'] and h.hexdigest()==meta['sha256'],step
  tmp.chmod(0o644);tmp.replace(dst);atomic(OUT/f'restored_{step}.json',dict(step=step,size=meta['size'],sha256=h.hexdigest(),time=time.time()))
  print('RESTORE_VERIFIED',step,flush=True)
 print('RESTORE_ALL_COMPLETE',flush=True)
def node(mode):
 prior={'a100':'eval_a100_runtime/campaigns/dev_recovery7500_20260927_a100','157':'eval_launch_logs/dev_recovery7500_20260927_157','new':'eval_launch_logs/dev_recovery7500_20260927_new'}
 old=json.loads((R/prior[mode]/'config.json').read_text());shared=json.loads((OUT/'config.json').read_text())
 base=R/('eval_a100_runtime/campaigns' if mode=='a100' else 'eval_launch_logs');folder=base/(NAME+'_'+mode);folder.mkdir(exist_ok=True)
 lock=(folder/'node.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=dict(shared,output=str(folder),eval_venv=old['eval_venv'],runtime_env=old['runtime_env'],artifact_root=old['artifact_root'],persistent_model=False)
 cp=folder/'config.json';atomic(cp,cfg);cp.chmod(0o600)
 def status():
  req=urllib.request.Request(URL+'/status',data=b'{}',headers={'Authorization':'Bearer '+cfg['token']})
  with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
 env=os.environ.copy();env.update(old['runtime_env']);env.update(NO_PROXY='localhost,127.0.0.1,192.168.52.208',no_proxy='localhost,127.0.0.1,192.168.52.208',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NVIDIA_DRIVER_CAPABILITIES='all',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
 subprocess.run([cfg['eval_venv']+'/bin/python','-c','import torch,sapien,mplib,curobo; assert torch.cuda.device_count()==8; print("Imports and 8 GPUs OK",flush=True)'],env=env,check=True)
 ps=[];last={};launches={};offset={'new':0,'157':6,'a100':12}[mode];counter=0
 def stopping(*a):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
 try:
  while True:
   st=status()
   if all(v.get('done',0)>=v['target'] for v in st.values()):break
   for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).splitlines():
    gpu,total,free=map(int,row.split(','));active=sum(p.poll() is None for g,p in ps if g==gpu)
    if free<40960 or active>=2 or time.time()-last.get(gpu,0)<180 or launches.get(gpu,0)>=10:continue
    ready=[g for g,s in cfg['groups'].items() if pathlib.Path(s['checkpoint']).is_file() and st[g].get('done',0)+st[g].get('leased',0)<s['target']]
    if not ready:continue
    preferred=ready[(offset+counter)%len(ready)];counter+=1
    with (folder/f'worker_gpu{gpu}_{counter}.log').open('a') as log:
     p=subprocess.Popen([sys.executable,'-u',str(R/'scripts/top100_eval_worker.py'),str(cp),str(gpu),preferred],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    ps.append((gpu,p));last[gpu]=time.time();launches[gpu]=launches.get(gpu,0)+1;print('LAUNCH',gpu,p.pid,preferred,'free_mib',free,flush=True)
   atomic(folder/'manager_status.json',dict(updated=time.time(),groups=st,workers=[dict(gpu=g,pid=p.pid,exit=p.poll()) for g,p in ps]))
   time.sleep(20)
 finally:
  for g,p in ps:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for g,p in ps:
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
  try:atomic(folder/'manager_status.json',dict(updated=time.time(),groups=status(),workers=[dict(gpu=g,pid=p.pid,exit=p.poll()) for g,p in ps]))
  except Exception:pass
def launch(mode):
 if mode in ('coordinator','restore','report'):folder=OUT
 else:folder=R/('eval_a100_runtime/campaigns' if mode=='a100' else 'eval_launch_logs')/(NAME+'_'+mode)
 folder.mkdir(exist_ok=True)
 py=str(R/'eval_a100_runtime/venv/bin/python') if mode=='restore' else sys.executable
 with (folder/(mode+'.log')).open('a') as log:
  p=subprocess.Popen([py,'-u',str(R/'scripts/eval_top100_campaign.py'),mode],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (folder/(mode+'.pid')).write_text(str(p.pid));print('START',mode,p.pid,flush=True)
def report():
 baseline=json.loads((OUT/'baseline_evidence.json').read_text())
 while True:
  st=json.loads((OUT/'manager_status.json').read_text())
  lines=['# 9-task mixed-run top-checkpoint evaluation (100 episodes)', '',
         'Updated (UTC): '+time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(st['updated'])), '',
         '| Task | Checkpoint | Denoising | Previous 50 | Added episodes | Total | Success rate |',
         '|---|---:|---:|---:|---:|---:|---:|']
  for g,v in st['groups'].items():
   task,tail=g.rsplit('_step',1);step,d=tail.split('_d');b=baseline[g];n=v.get('done',0);ok=v.get('success',0)
   lines.append(f'| {task} | {int(step)} | {d} | {b["success"]}/50 | {ok-b["success"]}/{n-50} | {ok}/{n} | {ok/n:.1%} |')
  lines.extend(['','All tasks use the original executor. Completed baseline seeds are preserved. Battery is excluded.'])
  f=OUT/'results.md';t=f.with_suffix('.tmp');t.write_text('\n'.join(lines)+'\n');t.replace(f)
  if all(v.get('done',0)>=v['target'] for v in st['groups'].values()):return
  time.sleep(30)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='coordinator':coordinator()
 elif mode=='restore':restore()
 elif mode=='report':report()
 elif mode=='launch':launch(sys.argv[2])
 else:node(mode)
