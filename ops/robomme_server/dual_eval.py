"""One A100 coordinator and an idle CPU-only Volc agent; disjoint task shards."""
import argparse,copy,fcntl,importlib.util,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

ROOT=Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme/launchers')
WORK=ROOT/'dual_eval_20261026'
REPO=Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def base_module():
    spec=importlib.util.spec_from_file_location('dual_base',ROOT/'relay_push75_recorded_20261025.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def launch(node):
    node=Path(node)
    with (node/'worker.log').open('a') as log:
        return subprocess.Popen(['bash','-c','source experiments/robomme/env.sh && exec python -u '+str(Path(__file__).resolve())+' worker --node '+str(node)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def partition(tasks,weights,capacities):
    groups={h:[] for h,c in capacities.items() if c>0};loads={h:0. for h in groups}
    for task in sorted(tasks,key=lambda t:(-weights.get(t,1),t)):
        h=min(groups,key=lambda h:loads[h]/capacities[h]);groups[h].append(task);loads[h]+=weights.get(task,1)
    return groups

def worker(node):
    from experiments.robomme import campaign as c
    node=Path(node);lock=(node/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (node/'finished.json').exists():return
    args=argparse.Namespace(**read(node/'args.json'));manifest=read(node/'manifest.json')
    # Use deterministic host-local lane IDs; the old base launcher can patch this.
    c.build_lanes=lambda gpus,n,p:[(g,str(g) if n==1 else f'{g}_{j}',p+i*n+j) for i,g in enumerate(gpus) for j in range(n)]
    try:code=c.Campaign(args,manifest).run()
    except Exception as exc:
        write(node/'finished.json',dict(exit_code=1,error=repr(exc),time=time.time()));raise
    write(node/'finished.json',dict(exit_code=code,time=time.time()))

def agent():
    WORK.mkdir(exist_ok=True);lock=(WORK/'volc-agent.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=base_module();child=None;node=None;stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopped:
        try:
            if child is not None and child.poll() is not None:
                if not (node/'finished.json').exists():write(node/'finished.json',dict(exit_code=child.returncode,time=time.time()))
                child=None
            free,slots=base.free_slots()
            write(WORK/'volc-heartbeat.json',dict(time=time.time(),host=socket.gethostname(),pid=os.getpid(),free_mib=free,gpus=list(slots),child_pid=child.pid if child else None,node=str(node) if child else None))
            req=WORK/'request.json'
            if child is not None and req.exists():
                d=read(req)
                if d.get('node')==str(node) and not d.get('active'):
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=90)
                    if not (node/'finished.json').exists():write(node/'finished.json',dict(exit_code=130,time=time.time()))
                    child=None
            if child is None and req.exists():
                d=read(req);candidate=Path(d['node'])
                if d.get('active') and not (candidate/'finished.json').exists():
                    requested=read(candidate/'args.json')['gpus']
                    if set(requested)<=set(slots):
                        node=candidate;child=launch(node)
                        print('START',node,child.pid,flush=True)
            time.sleep(5)
        except Exception as exc:
            print('AGENT_ERROR',repr(exc),flush=True);time.sleep(10)
    if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=90)

def dispatch(cli,base):
    from experiments.robomme import campaign as c
    original=c.Campaign
    class Coordinator:
        def __init__(self,args,manifest):self.args,self.manifest=args,manifest
        def run(self):
            out=Path(self.args.output);manifest=self.manifest
            lock=(WORK/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (WORK/'request.json').exists():
                old=read(WORK/'request.json')
                if not (Path(old['node'])/'finished.json').exists():
                    raise RuntimeError('Previous Volc shard has not stopped; refusing overlapping dispatch')
            # Never overwrite an older active generation. Retries use a fresh directory
            # but retain selected task records through the canonical task symlinks.
            generation=out/'generations'/str(time.time_ns());generation.mkdir(parents=True)
            peer=read(WORK/'volc-heartbeat.json') if (WORK/'volc-heartbeat.json').exists() else {}
            remote=peer.get('gpus',[]) if time.time()-peer.get('time',0)<30 and not peer.get('child_pid') else []
            remote_lanes=getattr(cli,'remote_lanes',1)
            remote=[g for g in remote if peer.get('free_mib',{}).get(str(g),0)>=remote_lanes*20*1024]
            local=self.args.gpus
            previous=c.collect_results(out,manifest)
            weights={t:max(1,len(v['missing_episode_ids'])) for t,v in previous['tasks'].items()}
            groups=partition(manifest['tasks'],weights,dict(a100=len(local),volc=len(remote)*remote_lanes))
            nodes={};process=None;stopped=False
            def stop(*_):
                nonlocal stopped
                stopped=True
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            for host,tasks in groups.items():
                node=generation/host;node.mkdir();nodes[host]=node
                args=copy.deepcopy(vars(self.args));args.update(output=str(node),gpus=local if host=='a100' else remote,port_base=27400,lanes_per_gpu=1 if host=='a100' else remote_lanes)
                m=dict(manifest,tasks=tasks,host=host,lanes_per_gpu=args['lanes_per_gpu'],gpu_indices=args['gpus'])
                write(node/'args.json',args);write(node/'manifest.json',m)
                for task in tasks:
                    canonical=out/'tasks'/task;canonical.mkdir(parents=True,exist_ok=True)
                    (node/'tasks').mkdir(exist_ok=True)
                    (node/'tasks'/task).symlink_to(canonical,target_is_directory=True)
            write(out/'allocation.json',dict(groups=groups,gpus=dict(a100=local,volc=remote),generation=str(generation)))
            if 'volc' in nodes:write(WORK/'request.json',dict(active=True,node=str(nodes['volc']),output=str(out)))
            process=launch(nodes['a100'])
            print('DUAL_STARTED',json.dumps(read(out/'allocation.json')),flush=True)
            try:
                while not stopped:
                    if process.poll() is not None and not (nodes['a100']/'finished.json').exists():
                        write(nodes['a100']/'finished.json',dict(exit_code=process.returncode,time=time.time()))
                    summary=c.collect_results(out,manifest);errors=[];states={}
                    for host,node in nodes.items():
                        if (node/'status.json').exists():states[host]=read(node/'status.json');errors+=states[host].get('errors',[])
                        if (node/'finished.json').exists() and read(node/'finished.json').get('exit_code'):
                            errors.append(dict(host=host,worker=read(node/'finished.json')))
                    done=all((n/'finished.json').exists() for n in nodes.values())
                    summary.update(errors=errors,status=('complete' if done and summary['completed']==800 and not errors else 'incomplete' if done else 'running'))
                    write(out/'summary.json',summary)
                    write(out/'status.json',dict(pid=os.getpid(),updated_at=c.now(),host='a100+volc',nodes=states,errors=errors))
                    if done:return 0 if summary['status']=='complete' else 1
                    if 'volc' in nodes:
                        hb=read(WORK/'volc-heartbeat.json')
                        if time.time()-hb.get('time',0)>180:raise RuntimeError('Volc agent heartbeat stale; stop before reassigning its tasks')
                    time.sleep(10)
                return 130
            finally:
                if 'volc' in nodes:write(WORK/'request.json',dict(active=False,node=str(nodes['volc']),output=str(out)))
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=90)
                if 'volc' in nodes:
                    deadline=time.time()+90
                    while not (nodes['volc']/'finished.json').exists() and time.time()<deadline:time.sleep(2)
    c.Campaign=Coordinator;base.RUN=REPO/'runs'/cli.run
    from a100_tail_reservations import pause_for_startup
    pause_for_startup()
    try:return base.worker(cli)
    finally:c.Campaign=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['agent','worker']);p.add_argument('--node');a=p.parse_args()
    if a.mode=='agent':agent()
    else:worker(a.node)
