"""Archive only this mixed run after all nine tasks finish, then unlink verified local weights."""
import argparse,fcntl,importlib.util,json,logging,os,pathlib,re,sys,time
R=pathlib.Path(__file__).resolve().parents[1]
RUN=R/'runs/ninetask_fastwam-anchor-helios-no-robotwininit_histaugtrue_bs16_ga1_gpu8_ep500_save2500_lr0.0001_cosine_20260922_050552_resume-step_002500'
STATE=R/'runtime/tos_eval_archive'
TASKS={'observe_and_pickup','press_button','swap_T','swap_blocks','battery_try','rearrange_blocks','blocks_ranking_try','cover_blocks','put_back_block'}
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
backend=module('archive_backend',R.parent/'starwam/starwam/tools/checkpoint_tos/backend.py')
uploader=module('archive_payload',R/'scripts/upload_checkpoints.py')
def atomic(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def campaigns():
 found=[]
 for base in [R/'eval_launch_logs',R/'eval_a100_runtime/campaigns']:
  if not base.exists():continue
  for cp in base.glob('*/config.json'):
   if (cp.parent/'retired.json').exists():continue
   try:
    cfg=json.loads(cp.read_text());groups=cfg.get('groups',{});relevant={}
    for g,v in groups.items():
     if not isinstance(v,dict):continue
     ck=pathlib.Path(v.get('checkpoint',''))
     if ck.parent==RUN/'checkpoints/weights' and re.fullmatch(r'step_\d+\.pt',ck.name):relevant[g]=v
    if not relevant:continue
    statuspath=cp.parent/'manager_status.json'
    if not statuspath.exists():found.append((cp.parent,relevant,None));continue
    status=json.loads(statuspath.read_text());found.append((cp.parent,relevant,status.get('groups',{})))
   except (OSError,ValueError) as e:raise RuntimeError(f'Cannot inspect campaign {cp}: {e}')
 return found

def eligibility(step):
 taskset=set();evidence=[];newest=0
 for folder,groups,status in campaigns():
  subset={g:v for g,v in groups.items() if pathlib.Path(v['checkpoint']).name==f'step_{step:06d}.pt'}
  if not subset:continue
  if status is None:return False,'campaign has no status: '+str(folder),[]
  complete=set()
  for g,v in subset.items():
   s=status.get(g,{})
   if s.get('done',0)<50 or s.get('done',0)<s.get('target',50) or s.get('leased',0):return False,'unfinished or leased '+g,[]
   complete.add(v['task'])
  if TASKS.issubset(complete):taskset=complete
  evidence.append({'campaign':str(folder),'groups':{g:status[g] for g in subset}})
  for log in folder.glob(f'*step_{step:06d}*.log'):newest=max(newest,log.stat().st_mtime)
 if not TASKS.issubset(taskset):return False,'no complete nine-task campaign',evidence
 if time.time()-newest<900:return False,'waiting 15 minutes after last rollout log',evidence
 ckname=f'step_{step:06d}.pt'
 for proc in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if str(RUN) in cmd and ckname in cmd and ('eval_policy.py' in cmd or 'eval_robotwin_single.py' in cmd):return False,'local evaluator still uses checkpoint',evidence
  except (OSError,PermissionError):pass
 return True,'complete',evidence

def links(step):
 primary=RUN/f'checkpoints/weights/step_{step:06d}.pt';payload=RUN/f'tos_payload/step_{step:06d}/weights.pt'
 paths=[p for p in [payload,primary] if p.exists()]
 if not paths:return [],None
 st=paths[0].stat()
 for p in paths:
  if p.is_symlink() or not p.resolve().is_relative_to(RUN.resolve()):raise RuntimeError('Unsafe cleanup path '+str(p))
  if (p.stat().st_dev,p.stat().st_ino)!=(st.st_dev,st.st_ino):raise RuntimeError('Payload is not the same checkpoint inode')
 if st.st_nlink!=len(paths):raise RuntimeError(f'Unaccounted hard links: inode has {st.st_nlink}, expected {len(paths)}')
 return paths,st

def process(store,step,evidence):
 receipt=STATE/f'step_{step:06d}.json';paths,before=links(step)
 if not paths:return
 primary=RUN/f'checkpoints/weights/step_{step:06d}.pt';payload=RUN/f'tos_payload/step_{step:06d}'
 logging.info('VERIFY_OR_UPLOAD step=%s local_bytes=%s',step,before.st_size)
 if not (payload/'weights.pt').exists():uploader.stage_payload(RUN,primary)
 result=store.upload_checkpoint(payload,run_name=RUN.name,checkpoint_id=step,source_world_size=8,state_root=STATE/'upload_state'/str(step))
 verified=store.verify_checkpoint(RUN.name,step)
 if result['manifest_sha256']!=verified['manifest_sha256']:raise RuntimeError('Manifest changed between upload and verification')
 ok,reason,evidence=eligibility(step)
 if not ok:logging.info('KEEP step=%s reason=%s',step,reason);return
 paths,after=links(step)
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('Local weight changed during archive')
 raw=store._get_manifest_bytes(RUN.name,step,missing_ok=False)
 manifest=store._parse_manifest(raw)
 data={'step':step,'run':str(RUN),'verified_at':time.time(),'remote':verified,'remote_manifest':manifest,'evaluation_evidence':evidence,'local_paths':[str(p) for p in paths],'status':'verified_pending_delete','allocated_bytes':after.st_blocks*512}
 atomic(receipt,data)
 for p in paths:
  current=p.stat()
  if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('Cleanup target changed')
  p.unlink();logging.info('UNLINK %s',p)
 data.update(status='deleted',deleted_at=time.time());atomic(receipt,data)
 logging.info('ARCHIVED_AND_DELETED step=%s freed_allocated_bytes=%s destination=%s',step,data['allocated_bytes'],verified['destination'])

def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',action='store_true');a.add_argument('--doctor',action='store_true');args=a.parse_args();STATE.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
 if args.plan:
  for p in sorted((RUN/'checkpoints/weights').glob('step_*.pt')):
   step=int(p.stem[5:]);ok,reason,_=eligibility(step);print(step,ok,reason)
  return
 lock=(STATE/'archive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 credentials=json.loads((STATE/'credentials.json').read_text());os.environ.update(credentials)
 os.environ['NO_PROXY']=os.environ['no_proxy']='localhost,127.0.0.1,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com'
 store=backend.TosCheckpointStore(backend.TosUploadConfig(endpoint='https://tos-cn-beijing.ivolces.com',region='cn-beijing',bucket=credentials['TOS_BUCKET'],prefix=credentials['TOS_PREFIX']))
 store.check_access()
 if args.doctor:print('TOS access verified');return
 while True:
  try:
   report={}
   for p in sorted((RUN/'checkpoints/weights').glob('step_*.pt')):
    step=int(p.stem[5:]);receipt=STATE/f'step_{step:06d}.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status')=='deleted':continue
    try:
     ok,reason,evidence=eligibility(step);report[str(step)]={'eligible':ok,'reason':reason}
     if ok:process(store,step,evidence)
    except Exception as e:logging.exception('KEEP_LOCAL step=%s archive failed',step);report[str(step)]={'error':str(e)}
   atomic(STATE/'status.json',{'updated':time.time(),'checkpoints':report})
  except Exception:logging.exception('Watcher cycle failed; no unverified deletion')
  time.sleep(300)
if __name__=='__main__':main()
