"""Observe 16 checkpoints: descending 30-trial screen, top-three extend to 100.

Runs on ecs-a100. SQLite stays on local disk, with shared filesystem backups.
Reuses official RMBench episode outcomes/lease protocol, never training masks.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rmbench_eval_queue import Queue

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = 'observe_and_pickup_helios_futuremaskaux_robotwininit_bs4_ga1_gpu8_ep500_lr1e-4_20261003_053337'
STEPS = list(range(8000, 0, -500))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


class CampaignQueue(Queue):
    def __init__(self, db, groups):
        super().__init__(str(db), groups)
        self.groups = groups
        with self.lock, self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS campaign (id INTEGER PRIMARY KEY, payload TEXT)')
            if not self.db.execute('SELECT 1 FROM campaign WHERE id=1').fetchone():
                self.db.execute('UPDATE groups SET target=30')
                self.save(dict(phase='screen30', selected=[], tie_break='higher checkpoint step'))

    def state(self):
        return json.loads(self.db.execute('SELECT payload FROM campaign WHERE id=1').fetchone()[0])

    def save(self, state):
        self.db.execute('INSERT OR REPLACE INTO campaign VALUES(1,?)', (json.dumps(state),))

    def advance(self):
        state = self.state()
        counts = {g: self.db.execute("SELECT count(*),coalesce(sum(success),0) FROM jobs WHERE g=? AND state='done'", (g,)).fetchone() for g in self.groups}
        if state['phase'] == 'screen30' and all(n == 30 for n, _ in counts.values()):
            ranked = sorted(self.groups, key=lambda g:(counts[g][1], self.groups[g]['step']), reverse=True)
            state.update(phase='extend100', selected=ranked[:3], screening={g:dict(completed=counts[g][0],success=counts[g][1]) for g in ranked})
            for g in state['selected']:
                self.db.execute('UPDATE groups SET target=100 WHERE g=?', (g,))
            self.save(state)
        if state['phase'] == 'extend100' and all(counts[g][0] == 100 for g in state['selected']):
            state['phase'] = 'complete'
            self.save(state)

    def call(self, op, data):
        with self.lock, self.db:
            result = super().call(op, data)
            self.advance()
            # A finish may have advanced the stage; return current targets.
            return super().call(op, data) if op == 'status' else result

    def snapshot(self, path):
        with self.lock:
            state = self.state()
            state['results'] = self.call('status', {})
            state['episodes'] = [dict(group=g,scene_seed=s,success=bool(success),detail=json.loads(detail or '{}'))
                for g,s,success,detail in self.db.execute("SELECT g,s,success,detail FROM jobs WHERE state='done' ORDER BY g,s")]
            write_json(path / 'results.json', state)
            tmp = path / 'queue_backup.tmp'
            dst = sqlite3.connect(str(tmp))
            self.db.backup(dst)
            dst.close()
            tmp.replace(path / 'queue_backup.sqlite')
        return state


def gpu_free():
    p = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,check=True,timeout=20)
    return {int(i):int(free) for i,free in (row.split(',') for row in p.stdout.strip().splitlines())}


def command(group, gpu, attempt, output, port, token):
    run = ROOT / 'runs' / RUN_NAME
    env = {k:v for k,v in os.environ.items() if not k.startswith('RMBENCH_') and k not in ('PYTHONPATH','CUDA_VISIBLE_DEVICES')}
    env.update(COMPARISON_VENV=str(ROOT/'eval_a100_runtime/venv'),
        RMBENCH_EVAL_ROOT=str(ROOT/'eval_runtime/RMBench'), MEMORY_BACKEND='helios',
        RMBENCH_TASK='observe_and_pickup', EVAL_EPISODES='100',
        CHECKPOINT=str(run/'checkpoints/weights'/f'{group}.pt'),
        EVAL_STATS_PATH=str(run/'dataset_stats.json'),
        PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', CUDA_HOME=str(ROOT/'eval_a100_runtime/cuda'),
        TORCH_EXTENSIONS_DIR=f'/tmp/memwam_extensions_{os.environ.get("USER","spidy.wang")}',
        OMP_NUM_THREADS='2', NO_PROXY='localhost,127.0.0.1', no_proxy='localhost,127.0.0.1',
        RMBENCH_ACTION_EXECUTOR='legacy', RMBENCH_EVAL_VIDEO_CAMERA='world_left',
        RMBENCH_QUEUE_URL=f'http://127.0.0.1:{port}', RMBENCH_QUEUE_TOKEN=token, RMBENCH_QUEUE_GROUP=group,
        RMBENCH_QUEUE_YIELD_ON_WAIT='1', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONUTF8='1')
    tag = f'{output.name}_{group}_gpu{gpu}_attempt{attempt}'
    cmd = ['bash', str(ROOT/'scripts/eval_comparison.sh'),f'gpu_id={gpu}','seed=42',
           'EVALUATION.start_scene_seed=100000','EVALUATION.num_inference_steps=50',
           'EVALUATION.action_horizon=16','EVALUATION.replan_steps=16',
           f'++EVALUATION.artifact_root={output}/artifacts', f'EVALUATION.output_dir={tag}']
    return cmd, env, output/'logs'/f'{tag}.log'


def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=18965)
    parser.add_argument('--interval', type=int, default=20)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    (output/'logs').mkdir(parents=True,exist_ok=True)
    local = Path('/tmp')/f'{output.name}_{os.getuid()}'
    local.mkdir(exist_ok=True)
    lock = (local/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run = ROOT/'runs'/RUN_NAME
    for step in STEPS:
        assert (run/'checkpoints/weights'/f'step_{step:06d}.pt').is_file()
    assert (run/'dataset_stats.json').is_file()
    db = local/'queue.sqlite'
    if not db.exists() and (output/'queue_backup.sqlite').exists():
        shutil.copyfile(output/'queue_backup.sqlite',db)
    groups = {f'step_{s:06d}':dict(step=s,seeds=[100000],completed=[]) for s in STEPS}
    queue = CampaignQueue(db,groups)
    token = secrets.token_hex(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            if self.headers.get('Authorization') != 'Bearer '+token:
                self.send_error(403); return
            try:
                result=queue.call(self.path.strip('/'),json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                data=json.dumps(result).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(400,str(exc))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    active=[]
    errors={g:0 for g in groups}
    attempts={g:len(list((output/'logs').glob(f'*{g}_gpu*_attempt*.log'))) for g in groups}
    retry_after={g:0 for g in groups}
    terminate=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:terminate.set())
    signal.signal(signal.SIGINT,lambda *_:terminate.set())
    write_json(output/'manifest.json',dict(host=socket.gethostname(),run=str(run),steps=STEPS,
        screen_episodes=30,top_k=3,final_episodes=100,seed=42,start_scene_seed=100000,
        inference_steps=50,replan_steps=16,stats=str(run/'dataset_stats.json'),
        free_threshold_mib=40960,loading_reservation_mib=40960,poll_seconds=args.interval,
        controller_pid=os.getpid(),local_database=str(db),port=args.port))
    print('CAMPAIGN_START',output,flush=True)
    try:
        while not terminate.is_set():
            now=time.time()
            status=queue.call('status',{})
            for job in list(active):
                p=job['proc'];rc=p.poll()
                if rc is None and now-job['log'].stat().st_mtime > 1200:
                    print('STOP_STALLED',job['group'],p.pid,flush=True);stop(p);rc=p.returncode
                if rc is None: continue
                active.remove(job)
                g=job['group']
                advanced=status[g].get('done',0)>job['done_at_start']
                if rc != 0 or not advanced:
                    errors[g]+=1;retry_after[g]=now+60
                print('EXIT',g,'gpu',job['gpu'],'rc',rc,'done',status[g].get('done',0),'errors',errors[g],flush=True)
            state=queue.snapshot(output)
            if state['phase']=='complete':
                print('CAMPAIGN_COMPLETE',json.dumps(state['selected']),flush=True);break
            free=gpu_free()
            # Until a real episode is committed, only one preflight process runs.
            validated=any(v.get('done',0)>0 for v in status.values())
            preflight_active=bool(active) and not validated
            counts={g:sum(j['group']==g for j in active) for g in groups}
            for job in active:
                if job['loading']:
                    # A claim is logged only after policy initialization finishes.
                    with job['log'].open('rb') as f:
                        f.seek(max(0,job['log'].stat().st_size-30000))
                        if b'RMBENCH_QUEUE_CLAIM' in f.read():job['loading']=False
                if job['loading']:free[job['gpu']]-=40960
            while not preflight_active:
                eligible=[g for g in groups if status[g].get('done',0)+status[g].get('leased',0)<status[g]['target']
                          and counts[g]<2 and errors[g]<3 and retry_after[g]<=now]
                available=[g for g,f in free.items() if f>=40960]
                if not eligible or not available:break
                # Prefer an unassigned checkpoint, in descending step order.
                group=min(eligible,key=lambda g:(counts[g],-groups[g]['step']))
                gpu=max(available,key=lambda g:(free[g],-g))
                attempts[group]+=1
                cmd,env,log=command(group,gpu,attempts[group],output,args.port,token)
                with log.open('w') as f:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                job=dict(proc=proc,group=group,gpu=gpu,log=log,loading=True,started=now,done_at_start=status[group].get('done',0))
                active.append(job);counts[group]+=1;free[gpu]-=40960
                print('START',group,'gpu',gpu,'pid',proc.pid,'log',log,flush=True)
                if not validated:break
            write_json(output/'scheduler.json',dict(updated=time.time(),phase=state['phase'],preflight_passed=validated,
                active=[{k:(str(v) if isinstance(v,Path) else v) for k,v in j.items() if k!='proc'}|{'pid':j['proc'].pid} for j in active],
                gpu_effective_free_mib=free,errors=errors,
                blocked=[g for g in groups if errors[g]>=3 and status[g].get('done',0)<status[g]['target']]))
            terminate.wait(args.interval)
    finally:
        for job in active:stop(job['proc'])
        queue.snapshot(output)
        server.shutdown()


if __name__=='__main__':main()
