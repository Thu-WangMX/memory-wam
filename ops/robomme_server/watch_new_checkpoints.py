"""Server-local continuous evaluation of new weights from two authorized runs."""
import argparse, fcntl, importlib.util, json, os, shlex, signal, subprocess, time
from pathlib import Path
ROOT=Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme/launchers')
spec=importlib.util.spec_from_file_location('base',ROOT/'relay_push75_recorded_20261025.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
RUNS=[('robomme_all16_push75_past32x8_future2_dense1_8gpu_b32ga1_s20k_20261024_012027',12500),
      ('robomme_all16_push75_action2_wsd_8gpu_b32ga1_s100k_t-20261024135050-4xrjt',8000),
      ('robomme_all16_push75_past32x8_future2_dense1_resume15k_8gpu_b32ga1_s100k_20261025_201820',15000),
      ('robomme_all16_push75_long444_past32x8_future2_dense1_16gpu_b16ga1_s50k_t-20261025232200-j9lmt',2500)]
WORK=ROOT/'watch_push75_and_wsd_20261025'

def completed(name,step):
    for p in (base.RUNTIME/'evals'/name).glob(f'step_{step:06d}*/summary.json'):
        try:
            summary=json.loads(p.read_text());manifest=json.loads(p.with_name('manifest.json').read_text())
            if (base.complete(summary) and manifest.get('split')=='test'
                and manifest.get('denoising_steps')==5 and manifest.get('max_steps')==1300):
                return str(p)
        except (OSError,ValueError): pass
    return None

def supervise():
    from robomme_gpu_reservation import Reservations
    WORK.mkdir(exist_ok=True)
    lock=(WORK/'lock_d5').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),started_at=base.now(),phase='starting',jobs={},runs=RUNS)
    statefile=WORK/'status.json'
    if statefile.exists():
        state['jobs']=json.loads(statefile.read_text()).get('jobs',{})
    stopped=False;child=None;log=None
    reservations=Reservations(base,WORK)
    def stop(*args):
        nonlocal stopped
        stopped=True
        if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def save():
        state['updated_at']=base.now();base.write(statefile,state)
    while not stopped:
        try:
            if child:
                code=child.poll()
                if code is None:
                    save();time.sleep(60);continue
                log.close();job=state['jobs'][state['current_job']]
                result=completed(job['run'],job['step'])
                job.update(phase='complete' if result else 'failed',exit_code=code,summary=result)
                if code==75:job['attempts']-=1
                child=None;state.update(child_pid=None,current_job=None)
            # The already authorized finite relay owns the GPUs until it completes.
            previous=ROOT/'wsd_8000_after_12500_20261025/status.json'
            prior=json.loads(previous.read_text())
            if prior.get('phase')!='complete':
                reservations.refresh()
                state['phase']='waiting_for_previous_evaluations';save();time.sleep(60);continue
            candidates=[]
            for name,floor in RUNS:
                for p in (base.REPO/'runs'/name/'checkpoints/weights').glob('step_*.pt'):
                    try:step=int(p.stem.split('_')[1])
                    except ValueError:continue
                    if step<=floor:continue
                    key=f'{name}:{step}'
                    if state['jobs'].get(key,{}).get('phase')=='complete':continue
                    result=completed(name,step)
                    if result:
                        state['jobs'][key]=dict(run=name,step=step,phase='complete',summary=result);continue
                    if state['jobs'].get(key,{}).get('attempts',0)>=2:continue
                    if base.checkpoint_ready(p)[0]:candidates.append((p.stat().st_mtime,name,step,key))
            state['pending']=[dict(run=n,step=s) for _,n,s,_ in sorted(candidates)]
            if not candidates:
                reservations.refresh()
                state['phase']='watching_for_new_checkpoint';save();time.sleep(60);continue
            reservations.release()
            _,slots=base.free_slots()
            if not slots:
                state['phase']='waiting_for_memory';save();time.sleep(60);continue
            _,name,step,key=sorted(candidates)[0]
            output=base.RUNTIME/'evals'/name/f'step_{step:06d}_test50_d5_video_autowatch_20261027'
            job=state['jobs'].setdefault(key,dict(run=name,step=step,attempts=0))
            job.update(attempts=job['attempts']+1,phase='running',output=str(output),started_at=base.now())
            command=['python','-u',str(Path(__file__).resolve()),'--worker','--run',name,'--step',str(step),'--output',str(output),'--relay-dir',str(WORK)]
            log=(WORK/f'{name}_step_{step:06d}.log').open('a')
            child=subprocess.Popen(['bash','-c','source experiments/robomme/env.sh && exec '+shlex.join(command)],cwd=base.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='evaluating',current_job=key,child_pid=child.pid,last_error=None);save()
        except Exception as exc:
            state.update(last_error=f'{type(exc).__name__}: {exc}');save()
        time.sleep(60)
    reservations.release()
    state['phase']='stopped';save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--run',choices=[r for r,_ in RUNS]);p.add_argument('--step',type=int);p.add_argument('--output');p.add_argument('--relay-dir')
    args=p.parse_args()
    if args.worker:
        base.RUN=base.REPO/'runs'/args.run
        from dual_eval import dispatch
        raise SystemExit(dispatch(args,base))
    supervise()
