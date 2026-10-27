"""Run requested dense final checkpoints to completion, then resume anchor-only."""
import fcntl,json,os,pathlib,signal,sqlite3,subprocess,sys,time
R=pathlib.Path(__file__).resolve().parents[1]
BASE=pathlib.Path(sys.argv[1]);PLAN=json.loads((BASE/'plan.json').read_text())
child=None
lock=(BASE/'relay.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def record(state,**extra):
 p=BASE/'relay_status.tmp';p.write_text(json.dumps(dict(updated=time.time(),state=state,**extra),indent=2));p.replace(BASE/'relay_status.json')
def stop(signum,frame):
 print('RELAY_STOP',signal.Signals(signum).name,flush=True)
 if child is not None and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:child.kill()
 record('stopped',signal=signum);raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def completed(name):
 folder=R/'eval_launch_logs'/name;cp=folder/'config.json';dbp=folder/'seeds.sqlite.backup'
 if not cp.exists() or not dbp.exists():return False
 cfg=json.loads(cp.read_text())
 with sqlite3.connect('file:'+str(dbp)+'?mode=ro',uri=True) as db:
  return all(db.execute("SELECT count(*) FROM jobs WHERE g=? AND state='done'",(g,)).fetchone()[0]>=s['target'] for g,s in cfg['groups'].items())
def execute(args,name,phase):
 global child
 folder=R/'eval_launch_logs'/name;folder.mkdir(exist_ok=True)
 with (folder/'launcher.log').open('a') as log:child=subprocess.Popen(args,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 (folder/'launcher.pid').write_text(str(child.pid)+'\n');record('running',phase=phase,campaign=name,manager_pid=child.pid);print('PHASE_START',phase,name,child.pid,flush=True)
 rc=child.wait();print('PHASE_EXIT',phase,name,rc,flush=True);return rc
try:
 for phase in PLAN['phases']:
  name=phase['name'];args=[sys.executable,'-u',str(R/'scripts/eval_checkpoint_manifest.py'),'--name',name,'--manifest',phase['manifest'],'--workers-per-gpu',str(phase['workers_per_gpu']),'--auto-gpus','--persistent-model']
  for attempt in range(3):
   if completed(name):break
   rc=execute(args,name,phase['group'])
   if completed(name):break
   record('retry_wait',phase=phase['group'],attempt=attempt+1,exit_code=rc);time.sleep(60)
  if not completed(name):raise RuntimeError('Priority campaign incomplete after retries: '+name)
  record('phase_complete',phase=phase['group']);print('VERIFIED_100_COMPLETE',name,flush=True)
 name=PLAN['anchor_name']
 if not completed(name):
  execute(PLAN['anchor_command'],name,'anchor_only_resume')
  if not completed(name):raise RuntimeError('Anchor queue exited incomplete; inspect its logs')
 record('all_complete');print('ALL_CAMPAIGNS_COMPLETE',flush=True)
except Exception as exc:
 record('error',error=repr(exc));raise
