"""Local-disk SQLite coordinator. Workers use HTTP; never share SQLite over NFS."""
import json,sqlite3,time,uuid,threading,os
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
class Queue:
 def __init__(self,path,manifest):
  self.lock=threading.RLock();self.db=sqlite3.connect(path,check_same_thread=False)
  self.db.executescript("CREATE TABLE IF NOT EXISTS jobs (g TEXT,s INTEGER,state TEXT,token TEXT,owner TEXT,until REAL,success INTEGER,detail TEXT,PRIMARY KEY(g,s)); CREATE TABLE IF NOT EXISTS groups (g TEXT PRIMARY KEY,target INTEGER,next_seed INTEGER);")
  with self.db:
   for g,v in manifest.items():
    self.db.execute('INSERT OR IGNORE INTO groups VALUES(?,?,?)',(g,int(v.get('target',100)),max(v['seeds']+[100000])+1))
    for s in v['seeds']:self.db.execute("INSERT OR IGNORE INTO jobs(g,s,state) VALUES(?,?,'pending')",(g,s))
    for invalid_seed in v.get('invalid_seeds',[]):self.db.execute("UPDATE jobs SET state='invalid' WHERE g=? AND s=? AND state='pending'",(g,invalid_seed))
    for r in v['completed']:
     self.db.execute("INSERT OR REPLACE INTO jobs(g,s,state,success,detail) VALUES(?,?,'done',?,?)",(g,r['scene_seed'],int(r['official_success']),json.dumps(r)))
 def call(self,op,a):
  with self.lock,self.db:
   now=time.time();self.db.execute("UPDATE jobs SET state='pending',token=NULL WHERE state='leased' AND until<?",(now,))
   g=a.get('group')
   if op=='status':
    return {g:{'target':target,**{st:n for st,n in self.db.execute('SELECT state,count(*) FROM jobs WHERE g=? GROUP BY state',(g,))},'success':self.db.execute("SELECT coalesce(sum(success),0) FROM jobs WHERE g=? AND state='done'",(g,)).fetchone()[0]} for g,target in self.db.execute('SELECT g,target FROM groups').fetchall()}
   if not self.db.execute('SELECT 1 FROM groups WHERE g=?',(g,)).fetchone():raise ValueError('unknown group')
   if op=='claim':
    target=self.db.execute('SELECT target FROM groups WHERE g=?',(g,)).fetchone()[0]
    done=self.db.execute("SELECT count(*) FROM jobs WHERE g=? AND state='done'",(g,)).fetchone()[0]
    active=self.db.execute("SELECT count(*) FROM jobs WHERE g=? AND state='leased'",(g,)).fetchone()[0]
    if done>=target:return {'state':'complete'}
    if done+active>=target:return {'state':'wait'}
    row=self.db.execute("SELECT s FROM jobs WHERE g=? AND state='pending' ORDER BY s LIMIT 1",(g,)).fetchone()
    if row:s=row[0]
    else:
     s=self.db.execute('SELECT next_seed FROM groups WHERE g=?',(g,)).fetchone()[0]
     self.db.execute('UPDATE groups SET next_seed=? WHERE g=?',(s+1,g));self.db.execute("INSERT INTO jobs(g,s,state) VALUES(?,?,'pending')",(g,s))
    token=uuid.uuid4().hex;self.db.execute("UPDATE jobs SET state='leased',token=?,owner=?,until=? WHERE g=? AND s=?",(token,a['owner'],now+300,g,s));return {'state':'leased','seed':s,'token':token}
   s=a['seed'];token=a['token'];row=self.db.execute('SELECT state,token FROM jobs WHERE g=? AND s=?',(g,s)).fetchone()
   if row and row[0]=='done' and row[1]==token and op=='finish':return {'ok':True}
   if not row or row!=('leased',token):return {'ok':False,'stale':True}
   if op=='heartbeat':self.db.execute('UPDATE jobs SET until=? WHERE g=? AND s=?',(now+300,g,s))
   elif op in ('finish','invalid','release'):
    state={'finish':'done','invalid':'invalid','release':'pending'}[op]
    self.db.execute('UPDATE jobs SET state=?,success=?,detail=? WHERE g=? AND s=?',(state,int(a.get('success',False)) if op=='finish' else None,json.dumps(a.get('detail',{})),g,s))
   else:raise ValueError('bad operation')
   return {'ok':True}
def serve(db,manifest,port,token):
 q=Queue(db,manifest)
 class Handler(BaseHTTPRequestHandler):
  def do_POST(self):
   if self.headers.get('Authorization')!='Bearer '+token:self.send_error(403);return
   try:
    a=json.loads(self.rfile.read(int(self.headers['Content-Length'])));v=q.call(self.path.strip('/'),a);data=json.dumps(v).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
   except Exception as e:self.send_error(400,str(e))
 server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.serve_forever()
if __name__=='__main__':
 import sys
 cfg=json.load(open(sys.argv[1]));serve(cfg['db'],cfg['groups'],cfg['port'],cfg['token'])
