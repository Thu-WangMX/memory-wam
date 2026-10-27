import pathlib,os,signal,sys,json,time,re
R=pathlib.Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')
mode=sys.argv[1];execute='--apply' in sys.argv
ids={'new':[358151,358158,358835,358836,358838,358840],'157':[3809441,3809443,3809445]}[mode]
folder=R/'eval_launch_logs'/('ninetask_top100_20260927_'+mode)
selected=set();procs={}
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  ppid=int(next(s.split(':')[1] for s in (p/'status').read_text().splitlines() if s.startswith('PPid:')))
  procs[int(p.name)]=(ppid,cmd)
 except (OSError,PermissionError,StopIteration):pass
for pid in ids:
 if pid not in procs or 'top100_eval_worker.py' not in procs[pid][1] or folder.name not in procs[pid][1]:continue
 logs=list(folder.glob(f'*_pid{pid}_attempt*.log'))
 if not logs:continue
 latest=max(logs,key=lambda f:int(re.search(r'_attempt(\d+)',f.name)[1]))
 txt=latest.read_text(errors='replace')
 if 'RMBENCH_QUEUE_CLAIM' in txt:print('KEEP_ROLLOUT',pid);continue
 selected.add(pid);print('UPGRADE_LOADING_WORKER',pid,latest.name)
# Include detached orphaned rollouts only from this campaign.
for pid,(ppid,cmd) in procs.items():
 if ppid==1 and folder.name in cmd and 'eval_policy.py' in cmd:selected.add(pid);print('ORPHAN',pid)
changed=True
while changed:
 changed=False
 for pid,(ppid,cmd) in procs.items():
  if ppid in selected and pid not in selected:selected.add(pid);changed=True
print('Selected own process tree size',len(selected))
if execute:
 for pid in selected:
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 time.sleep(5)
 for pid in selected:
  try:
   cmd=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if cmd==procs[pid][1]:os.kill(pid,signal.SIGKILL)
  except (FileNotFoundError,ProcessLookupError):pass
 print('Repair applied; node manager will refill available slots')
