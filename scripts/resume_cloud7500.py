import pathlib,json,sqlite3,re,os,sys,subprocess,time,fcntl
R=pathlib.Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')
old=R/'eval_launch_logs/cloud_ninetask_recovery_v1_five_persistent'
owner='dev_recovery7500_20260927'
records_path=old/'development_resume_records.json'
tasks={'a100':'swap_T','157':'swap_blocks','new':'battery_try'}
names={k:owner+'_'+k for k in tasks}
mode=sys.argv[1]
if mode=='prepare':
    rows={}
    with sqlite3.connect('file:'+str(old/'seeds.sqlite.backup')+'?mode=ro',uri=True) as db:
        for g,s,st,ok,d in db.execute('SELECT g,s,state,success,detail FROM jobs'):
            rows[g,s]=dict(group=g,scene_seed=s,state=st,success=ok,detail=json.loads(d or '{}'))
    recovered=0
    for log in old.glob('*step_007500*.log'):
        group=None
        for line in log.read_text(errors='replace').splitlines():
            if 'PERSISTENT_TASK_START ' in line:
                group=line.split('PERSISTENT_TASK_START ',1)[1].split()[0]
            marker='RMBENCH_EVAL_DIAGNOSTICS '
            if group and marker in line:
                try:d=json.loads(line.split(marker,1)[1]);s=int(d['scene_seed']);ok=int(bool(d['official_success']))
                except (ValueError,KeyError):continue
                prior=rows.get((group,s))
                if prior and prior['state']=='done':
                    assert bool(prior['success'])==bool(ok),(group,s,'conflicting result')
                else:recovered+=1
                rows[group,s]=dict(group=group,scene_seed=s,state='done',success=ok,detail=d)
    records_path.write_text(json.dumps(list(rows.values())))
    records_path.chmod(0o644)
    print('Recovered extra log results',recovered)
    for task in [*tasks.values(),'press_button']:
        g=task+'_step007500_d'+('50' if task=='press_button' else '10')
        done=[r for r in rows.values() if r['group']==g and r['state']=='done']
        print(g,'done',len(done),'success',sum(bool(r['success']) for r in done))
    reg=R/'runtime/ninetask_eval_assignments.json';greg=R/'runtime/ninetask_group_assignments.json'
    with (R/'runtime/ninetask_assignments.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(reg.read_text());data[old.name]=[s for s in data.get(old.name,[]) if s!=7500]
        for k,v in data.items():
            assert k==owner or 7500 not in v,(k,v)
        data[owner]=[7500];t=reg.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(reg)
        data=json.loads(greg.read_text())
        for k,task in tasks.items():data[task+'_step007500_d10']=names[k]
        t=greg.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(greg)
    (old/'retired.json').write_text(json.dumps(dict(reason='User canceled cloud; Press canceled; remaining three tasks moved to development machines',time=time.time(),replacements=names),indent=2))
    sys.exit()
assert mode in tasks
assert (old/'retired.json').exists()
prior={'a100':'eval_a100_runtime/campaigns/dev_tail_67500_2500_a100_20260927','157':'eval_launch_logs/dev_tail_67500_2500_157_20260927','new':'eval_launch_logs/dev_tail_2500_battery_new_20260927'}
cfg=json.loads((R/prior[mode]/'config.json').read_text())
base=R/('eval_a100_runtime/campaigns' if mode=='a100' else 'eval_launch_logs')
out=base/names[mode];out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(cfg.get('runtime_env',{}));env.update(COMPARISON_VENV=cfg['eval_venv'],EVAL_QUEUE_ROOT=str(base),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NVIDIA_DRIVER_CAPABILITIES='all')
if cfg.get('artifact_root'):env['EVAL_ARTIFACT_ROOT']=cfg['artifact_root']
cmd=['python3','-u',str(R/'scripts/eval_ninetask_cloud.py'),'--name',names[mode],'--assignment-owner',owner,'--auto-gpus','--workers-per-gpu','2','--steps','7500','--tasks',tasks[mode],'--resume-records',str(records_path)]
subprocess.run(cmd+['--check-only'],cwd=R,env=env,check=True)
with (out/'launcher.log').open('a') as log:
    p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
(out/'launcher.pid').write_text(str(p.pid));print('START',names[mode],p.pid,flush=True)
time.sleep(10);print('EXIT',p.poll());print((out/'launcher.log').read_text()[-1600:])
