"""Progress-based lease renewal prevents crashed/hung workers holding quota forever."""
import os,json,time,threading,urllib.request,socket
class Client:
 def __init__(self):
  self.url=os.environ['RMBENCH_QUEUE_URL'];self.token=os.environ['RMBENCH_QUEUE_TOKEN'];self.group=os.environ['RMBENCH_QUEUE_GROUP'];self.owner=socket.gethostname()+':'+str(os.getpid());self.current=None;self.progress=time.monotonic();self.lock=threading.RLock()
  threading.Thread(target=self._renew,daemon=True).start()
 def rpc(self,op,**kw):
  data=dict(group=self.group,owner=self.owner,**kw)
  if os.environ.get('RMBENCH_QUEUE_FILE_RPC_DIR'):
   from rmbench_file_rpc import call
   return call(os.environ['RMBENCH_QUEUE_FILE_RPC_DIR'],op,data)
  req=urllib.request.Request(self.url+'/'+op,data=json.dumps(data).encode(),headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
  for attempt in range(3):
   try:
    with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 def touch(self):self.progress=time.monotonic()
 def claim(self):
  while True:
   r=self.rpc('claim')
   if r['state']=='complete':return None
   if r['state']=='wait':
    if os.environ.get('RMBENCH_QUEUE_YIELD_ON_WAIT')=='1':return None
    time.sleep(5);continue
   with self.lock:self.current=r;self.touch()
   print('RMBENCH_QUEUE_CLAIM '+json.dumps(r),flush=True);return r['seed']
 def _renew(self):
  while True:
   time.sleep(20)
   with self.lock:
    if self.current is None:continue
    if time.monotonic()-self.progress>240:
     print('RMBENCH_QUEUE_FATAL no progress for 240s',flush=True);os._exit(75)
    try:
     r=self.rpc('heartbeat',seed=self.current['seed'],token=self.current['token'])
     if not r.get('ok'):os._exit(76)
    except Exception:pass
 def submit(self,op,**kw):
  with self.lock:
   assert self.current is not None
   r=self.rpc(op,seed=self.current['seed'],token=self.current['token'],**kw)
   if not r.get('ok'):raise RuntimeError('Queue lease expired: result not committed')
   self.current=None
 def invalid(self,reason):self.submit('invalid',detail={'reason':reason})
 def finish(self,success,detail):self.submit('finish',success=success,detail=detail)
