"""Archive the evaluated checkpoints explicitly selected on 2026-10-09.
Reuse TOS manifest SHA256/CRC64 verification and inode-safe deletion.
Default is a read-only plan; --apply performs upload, verification and unlink.
"""
import argparse,fcntl,importlib.util,json,logging,os,sqlite3,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
STATE=R/'runtime/evaluated_checkpoint_archive_20261009'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
archive=module('evaluated_archive_helpers',R/'scripts/archive_anchor_only_unused.py')
def evidence(item):
 run=R/'runs'/item['run'];ck=run/f"checkpoints/weights/step_{item['step']:06d}.pt"
 if run.resolve()!=run or run.parent!=R/'runs' or not run.is_dir():raise RuntimeError('Unsafe run path')
 if 'evidence_config' in item:
  cfg=json.loads(Path(item['evidence_config']).read_text())
  for g,spec in item['groups'].items():
   actual=cfg['groups'][g]
   if actual['checkpoint']!=str(ck) or actual['task']!=spec['task'] or actual['target']!=spec['target']:raise RuntimeError('Evaluation binding changed')
 else:
  cfg=json.loads(Path(item['evidence_manifest']).read_text())
  if cfg['run']!=str(run) or item['step'] not in cfg['steps']:raise RuntimeError('Evaluation run mismatch')
 rows={}
 with sqlite3.connect('file:'+item['database']+'?mode=ro',uri=True) as db:
  for g,spec in item['groups'].items():
   counts={s:n for s,n in db.execute('select state,count(*) from jobs where g=? group by state',(g,))}
   if counts.get('done',0)<spec['target'] or counts.get('leased',0):raise RuntimeError('Evaluation incomplete or leased: '+g)
   rows[g]=dict(counts=counts,target=spec['target'],success=db.execute("select sum(success) from jobs where g=? and state='done'",(g,)).fetchone()[0])
 # Refuse deletion when this host still references the specific weight, or
 # when a trainer/uploader is active on this run. Completed remote queues
 # have already been checked against their shared persisted result database.
 for proc in Path('/proc').glob('[0-9]*'):
  if int(proc.name)==os.getpid():continue
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except OSError:continue
  if str(run) in cmd and (ck.name in cmd or 'upload_checkpoints.py' in cmd or 'train.py' in cmd):raise RuntimeError('Run still referenced by process '+proc.name)
 return rows

def bind(item):
 archive.RUN=R/'runs'/item['run'];archive.KEEP=set();archive.ARCHIVE=[item['step']]
 archive.STATE=STATE/item['run'];archive.STATE.mkdir(exist_ok=True)
 archive.ensure_not_used=lambda step:evidence(item)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
 items=json.loads((STATE/'plan.json').read_text())['items'];logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
 with (STATE/'archive.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not args.apply:
   total=0
   for item in items:
    proof=evidence(item);bind(item);paths,st=archive.local_links(item['step']);amount=st.st_blocks*512 if st else 0;total+=amount
    print(json.dumps(dict(run=item['run'],step=item['step'],allocated_gib=round(amount/1024**3,3),evaluation=proof)),flush=True)
   print('TOTAL_GIB',round(total/1024**3,3),flush=True);return
  backend=module('evaluated_tos_backend',R.parent/'starwam/starwam/tools/checkpoint_tos/backend.py')
  uploader=module('evaluated_tos_payload',R/'scripts/upload_checkpoints.py')
  store=backend.TosCheckpointStore(backend.TosUploadConfig(endpoint=os.environ.get('TOS_ENDPOINT','https://tos-cn-beijing.ivolces.com'),region=os.environ.get('TOS_REGION','cn-beijing'),bucket=os.environ['TOS_BUCKET'],prefix=os.environ['TOS_PREFIX']))
  store.check_access();report={}
  for item in items:
   key=item['run']+'/'+str(item['step']);bind(item)
   try:
    proof=evidence(item);archive.atomic(archive.STATE/f"evaluation_{item['step']:06d}.json",dict(item=item,verified_at=time.time(),groups=proof))
    archive.process(store,uploader,backend,item['step'])
    receipt=json.loads((archive.STATE/f"step_{item['step']:06d}.json").read_text())
    if receipt['status']!='deleted':raise RuntimeError('Archive did not finish deletion')
    report[key]=dict(status='deleted',allocated_bytes=receipt['allocated_bytes'],destination=receipt['remote']['destination'])
   except Exception as exc:
    logging.exception('KEEP_LOCAL %s',key);report[key]=dict(status='error',error=str(exc))
   archive.atomic(STATE/'status.json',dict(updated=time.time(),checkpoints=report,freed_bytes=sum(v.get('allocated_bytes',0) for v in report.values())))
  if any(v['status']=='error' for v in report.values()):raise SystemExit(1)
  logging.info('ALL_COMPLETE checkpoints=%d freed_gib=%.3f',len(report),sum(v['allocated_bytes'] for v in report.values())/1024**3)
if __name__=='__main__':main()
