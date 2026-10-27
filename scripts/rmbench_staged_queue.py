"""Persistent 50-episode screening, top-three extension, then denoise-10 eval."""
import json
import sys
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from rmbench_eval_queue import Queue

class StagedQueue(Queue):
    def __init__(self, cfg):
        super().__init__(cfg['db'], cfg['groups'])
        self.cfg = cfg
        with self.lock, self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS stage_meta (id INTEGER PRIMARY KEY, payload TEXT)')
            if not self.db.execute('SELECT 1 FROM stage_meta WHERE id=1').fetchone():
                self.db.execute('UPDATE groups SET target=0')
                for g in self.fifty():
                    self.db.execute('UPDATE groups SET target=50 WHERE g=?', (g,))
                self.save(dict(phase='screen50', selected=[]))
    def fifty(self):
        return [g for g,v in self.cfg['groups'].items() if v['denoise_steps']==50]
    def save(self, state):
        self.db.execute('INSERT OR REPLACE INTO stage_meta VALUES(1,?)', (json.dumps(state),))
    def advance(self):
        state=json.loads(self.db.execute('SELECT payload FROM stage_meta WHERE id=1').fetchone()[0])
        def count(g):
            return self.db.execute("SELECT count(*),coalesce(sum(success),0) FROM jobs WHERE g=? AND state='done'",(g,)).fetchone()
        if state['phase']=='screen50' and all(count(g)[0]>=50 for g in self.fifty()):
            ranked=sorted(self.fifty(),key=lambda g:(count(g)[1]/count(g)[0],self.cfg['groups'][g]['checkpoint']),reverse=True)
            state.update(phase='extend100', selected=ranked[:3], screening={g:list(count(g)) for g in ranked})
            for g in state['selected']:
                self.db.execute('UPDATE groups SET target=100 WHERE g=?',(g,))
            self.save(state)
        if state['phase']=='extend100' and all(count(g)[0]>=100 for g in state['selected']):
            checkpoints={self.cfg['groups'][g]['checkpoint'] for g in state['selected']}
            tens=[g for g,v in self.cfg['groups'].items() if v['denoise_steps']==10 and v['checkpoint'] in checkpoints]
            assert len(tens)==3
            state.update(phase='denoise10', selected_10=tens)
            for g in tens:
                self.db.execute('UPDATE groups SET target=100 WHERE g=?',(g,))
            self.save(state)
        if state['phase']=='denoise10' and all(count(g)[0]>=100 for g in state['selected_10']):
            state['phase']='complete';self.save(state)
        target=Path(self.cfg['output'])/'stage.json';tmp=target.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(target)
    def call(self, op, data):
        with self.lock, self.db:
            self.advance()
            return super().call(op,data)

def main():
    cfg=json.load(open(sys.argv[1]));queue=StagedQueue(cfg)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get('Authorization')!='Bearer '+cfg['token']:
                self.send_error(403);return
            try:
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result=queue.call(self.path.strip('/'),data)
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
            except Exception as exc:
                self.send_error(400,str(exc))
    ThreadingHTTPServer(('127.0.0.1',cfg['port']),Handler).serve_forever()
if __name__=='__main__':main()
