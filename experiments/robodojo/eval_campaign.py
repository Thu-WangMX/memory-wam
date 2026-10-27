"""54 configurations / 42 equally weighted tasks / 840 scored episodes."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

DIMENSIONS={
 'Generalization':['stack_bowls','push_T','pack_objects_into_box','fold_clothes','hang_mugs','sweep_blocks','pour_liquid_into_cup','make_toast','arrange_largest_number','sort_nesting_dolls_by_size','store_laptop_and_headphones','stack_blocks'],
 'Precision':['fasten_screws','plug_in_charger','insert_tubes','pour_balls_into_vase','play_Xylophone','deposit_coin','insert_key','build_tower'],
 'Long-Horizon':['put_bottles_into_dustbin','fill_pen_holder','classify_objects','play_tic_tac_toe','fill_egg_holder','organize_table','make_kong','play_stacking_toy'],
 'Memory':['cover_blocks','match_and_pick_from_conveyor','swap_blocks','swap_T','press_by_number','imitate_sorting_sequence'],
 'Open':['align_blocks','general_pickup','stack_blocks_by_language','solve_equation','classify_objects_by_language','pick_from_conveyor_by_image','store_tools_in_toolbox','pour_by_language']}


def jobs():
    return [{'dimension':dim,'base_task':task,'task':name,'episodes':10 if dim=='Generalization' else 20}
            for dim,tasks in DIMENSIONS.items() for task in tasks
            for name in ([task,task+'_random'] if dim=='Generalization' else [task])]


def client_lanes(slots, server_layout, server_indices=None):
    if not 1<=len(slots)<=len(server_layout) or any(g<0 or slots.count(g)>2 for g in slots):
        raise ValueError('Require 1..server_count clients, at most two per simulator GPU')
    indices=list(range(len(slots))) if server_indices is None else server_indices
    available={row['index']:row for row in server_layout}
    if len(indices)!=len(slots) or len(set(indices))!=len(indices) or any(i not in available for i in indices):
        raise ValueError('Each client requires a distinct existing policy server index')
    return [{'lane':lane,'gpu':gpu,'server_index':index,'port':available[index]['port']}
            for lane,(gpu,index) in enumerate(zip(slots,indices))]


def entries(path):
    if not path.exists():return []
    # Simulator result writes may overlap another lane's summary update.
    for attempt in range(10):
        try:
            data=json.loads(path.read_text());break
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.1)
    rows=sorted(data['details'].items(),key=lambda x:int(x[0]))
    if len(rows)!=data['eval_time'] or len({v['layout_id'] for _,v in rows})!=len(rows):raise ValueError(f'Invalid/duplicate episode results: {path}')
    return rows


def aggregate(records):
    task_scores={}; dimension_scores={}
    for dim,tasks in DIMENSIONS.items():
        for task in tasks:
            names=[task,task+'_random'] if dim=='Generalization' else [task]
            quotas=[10,10] if dim=='Generalization' else [20]
            values=[]
            for name,quota in zip(names,quotas):
                rows=records.get(name,[])
                if len(rows)!=quota:break
                values.extend(rows)
            else:
                task_scores[task]={k:100*sum(float(v[field]) for v in values)/20
                                   for k,field in [('success_rate','success'),('score','score')]}
        if all(t in task_scores for t in tasks):
            dimension_scores[dim]={k:sum(task_scores[t][k] for t in tasks)/len(tasks) for k in ('success_rate','score')}
    overall={k:sum(v[k] for v in task_scores.values())/42 for k in ('success_rate','score')} if len(task_scores)==42 else None
    return {'tasks':task_scores,'dimensions':dimension_scores,'overall':overall,'complete_base_tasks':len(task_scores),
            'weighting':'42 base tasks equally weighted; dimension weights 12:8:8:6:8','official_full_protocol':False}


def main():
    from client_server.ws.model_client import WsModelClient
    from experiments.robodojo.eval_servers import stop
    runtime=Path(os.environ['ROBODOJO_RUNTIME']); root=Path(os.environ['ROBODOJO_FASTWAM_REPO']); stack=runtime/'eval-stack'
    run=os.environ['EVAL_RUN_ID']; host=os.environ['POLICY_HOST']
    if Path(run).name!=run or run in ('.','..'):raise ValueError('Invalid run identity')
    slots=[int(s) for s in os.environ.get('SIM_GPU_SLOTS','0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7').split(',')]
    base_port=int(os.environ.get('POLICY_BASE_PORT','8200')); server_dir=runtime/'eval-server-logs'/run
    identity=json.loads((server_dir/'identity.json').read_text())
    inference_steps=int(os.getenv('POLICY_INFERENCE_STEPS','10'))
    if inference_steps<=0:raise ValueError('POLICY_INFERENCE_STEPS must be positive')
    if identity['step']!=20000 or identity['server_count']!=16 or identity['inference_steps']!=inference_steps:raise ValueError('Wrong server identity')
    if [s['port'] for s in identity['layout']]!=list(range(base_port,base_port+16)):raise ValueError('Server ports differ')
    indices=os.getenv('POLICY_SERVER_INDICES')
    lanes=client_lanes(slots,identity['layout'],[int(i) for i in indices.split(',')] if indices is not None else None)
    out=runtime/'eval-campaigns'/run; out.mkdir(parents=True,exist_ok=True)
    singleton=(out/'campaign.lock').open('a'); fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    client_layout={'policy_host':host,'clients':len(lanes),'lanes':lanes}
    (out/'client-layout.json').write_text(json.dumps(client_layout,indent=2))
    print('CLIENT_LAYOUT '+json.dumps(client_layout),flush=True)
    plan=jobs(); assert len(plan)==54 and sum(j['episodes'] for j in plan)==840
    plan.sort(key=lambda j:(j['dimension']!='Open',j['task']))
    manifest={'policy':'FastWAMAnchorHelios','jobs':plan,'episodes':840,'identity':identity,'layout_seed':0,'policy_seed':42}
    previous=out/'manifest.json'
    if previous.exists() and json.loads(previous.read_text())!=manifest:raise ValueError('Cannot mix checkpoint/protocol identities in one campaign')
    previous.write_text(json.dumps(manifest,indent=2))
    def paths(task):
        child=f'{run}_{task}'
        parent=stack/'eval_result/RoboDojo'/task/'FastWAMAnchorHelios/arx_x5'/f'0_ckpt_name={child},action_type=joint'
        return child,parent/child/'_result.json',parent/f'_resume_{child}.json'
    states={}; pending=queue.Queue(); lock=threading.Lock(); cancel=threading.Event(); children={}
    task_timeout=float(os.getenv('SIM_TASK_TIMEOUT_SECONDS','43200'))
    if task_timeout<=0:raise ValueError('SIM_TASK_TIMEOUT_SECONDS must be positive')
    def persist():
        records={j['task']:[v for _,v in entries(paths(j['task'])[1])] for j in plan}
        for name,data in [('status',states),('summary',aggregate(records))]:
            tmp=out/(name+'.tmp'); tmp.write_text(json.dumps(data,indent=2)); tmp.replace(out/(name+'.json'))
    for job in plan:
        task=job['task']
        for folder,suffix in [('tasks','.py'),('config','.yml')]:
            if not (stack/'task/RoboDojo'/folder/(task+suffix)).is_file():raise FileNotFoundError(task)
        for index in range(job['episodes']):
            layout=stack/'Assets/Eval_Layout/RoboDojo/arx_x5/0'/f'{task}_{index}.json'
            if not layout.is_file():raise FileNotFoundError(layout)
        child,result,resume=paths(task); old=entries(result)
        if len(old)>job['episodes']:raise ValueError(f'Too many episodes: {task}')
        if 0<len(old)<job['episodes']:
            saved=json.loads(resume.read_text())
            if saved['run_id']!=child or saved['details']!=dict(old):raise ValueError(f'Resume mismatch: {task}')
        complete=len(old)==job['episodes']; states[task]={**job,'status':'complete' if complete else 'pending','completed_episodes':len(old)}
        if not complete:pending.put(job)
    persist()
    if os.getenv('EVAL_DRY_RUN')=='1':
        print(json.dumps({'configs':54,'episodes':840,'clients':len(lanes),'pending_configs':pending.qsize()})); return
    # Existing simulator jobs and other users' GPU allocations must be respected.
    for gpu in set(slots):
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        required=slots.count(gpu)*8500
        if free<required:raise RuntimeError(f'GPU {gpu}: {free} MiB free, requires {required}; do not stop others to make room')
    if pending.empty():return
    for row in lanes:
        client=WsModelClient(url=f'ws://{host}:{row["port"]}',evaluation_id=run,trial_id=f'health-{row["lane"]}',max_connect_attempts=1,request_timeout_s=30)
        try:status=client.call(func_name='status')
        finally:client.close()
        if status['checkpoint_step']!=20000 or status['inference_steps']!=inference_steps:raise ValueError('Live policy mismatch')
    def worker(lane):
        while not cancel.is_set():
            try:job=pending.get_nowait()
            except queue.Empty:return
            task=job['task']; child,result,resume=paths(task); old=dict(entries(result))
            row=lanes[lane]
            env=dict(os.environ,POLICY_PORT=str(row['port']),SIM_GPU_ID=str(row['gpu']),EVAL_TASK=task,EVAL_SEED='0',EVAL_NUM=str(job['episodes']),EVAL_RUN_ID=child,ROBODOJO_MAX_BASH_RETRIES='1')
            env.pop('ANCHOR_HELIOS_SMOKE_STEPS',None); env.pop('ANCHOR_HELIOS_SMOKE_OUTPUT',None)
            path=out/f'lane-{lane}-{task}-{time.time_ns()}.log'
            with path.open('w') as log:
                proc=subprocess.Popen(['bash',str(root/'experiments/robodojo/eval_client.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.monotonic()
                with lock:
                    children[lane]=proc; states[task].update(status='running',lane=lane,gpu=row['gpu'],port=row['port'],pid=proc.pid,log=str(path)); persist()
                while proc.poll() is None:
                    if cancel.is_set() or time.monotonic()-started>task_timeout:stop(proc);break
                    time.sleep(2)
                rows=entries(result)
                complete=proc.returncode==0 and len(rows)==job['episodes'] and all(dict(rows).get(k)==v for k,v in old.items())
                stop(proc)
                with lock:
                    children.pop(lane,None); states[task].update(status='complete' if complete else 'failed',completed_episodes=len(rows),returncode=proc.returncode); persist()
                if not complete:cancel.set();raise RuntimeError(f'Task {task} failed; preserved results at {result}')
    def interrupted(signum,frame):cancel.set();raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            futures=[pool.submit(worker,i) for i in range(len(lanes))]
            for future in concurrent.futures.as_completed(futures):
                try:future.result()
                except BaseException:cancel.set();raise
    finally:
        cancel.set()
        for child in list(children.values()):stop(child)
        with lock:persist()


if __name__=='__main__':main()
