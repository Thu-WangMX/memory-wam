"""Verified step-20000 Helios serving: two independent policies per cloud GPU."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

TRAIN_RUN='robodojo_all35_anchor_helios_histaugtrue_bs16_ga2_gpu8_s20000_20260924_v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stop(child):
    if child is None:return
    try: os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait(timeout=10)


def launch_policy_batches(layout, launch, probe, check, timeout=900):
    """Start one process per GPU concurrently; finish a batch before the next."""
    remaining=list(layout); batch_number=0
    while remaining:
        batch=[]; deferred=[]; seen=set()
        for row in remaining:
            if row['gpu'] in seen:deferred.append(row)
            else:batch.append(row);seen.add(row['gpu'])
        remaining=deferred; batch_number+=1
        print(f'START_BATCH {batch_number} servers={[r["index"] for r in batch]}',flush=True)
        deadlines={}
        for row in batch:
            check();launch(row);deadlines[row['index']]=time.monotonic()+timeout
        pending=list(batch); next_report=0
        while pending:
            check()
            for row in pending[:]:
                if time.monotonic()>=deadlines[row['index']]:
                    raise TimeoutError(f'Server {row["index"]} readiness timeout; inspect server-{row["index"]}.log')
                if probe(row):
                    pending.remove(row)
                    print(f'SERVER_READY {row["index"]} gpu={row["gpu"]} port={row["port"]}',flush=True)
            if pending:
                if time.monotonic()>=next_report:
                    print(f'BATCH_LOADING {batch_number} ready={len(batch)-len(pending)}/{len(batch)} waiting={[r["index"] for r in pending]}',flush=True)
                    next_report=time.monotonic()+30
                time.sleep(2)
        print(f'BATCH_READY {batch_number}',flush=True)


def main():
    import torch
    from omegaconf import OmegaConf
    from client_server.ws.model_client import WsModelClient
    root=Path(os.environ['ROBODOJO_FASTWAM_REPO']); runtime=Path(os.environ['ROBODOJO_RUNTIME'])
    run=root/'runs'/TRAIN_RUN
    tag=os.environ['EVAL_RUN_ID']
    if Path(tag).name!=tag or not tag or tag in ('.','..'):raise ValueError('Invalid EVAL_RUN_ID')
    if int(os.getenv('MLP_WORKER_NUM','1'))!=1:raise ValueError('Use one cloud worker')
    count=int(os.getenv('POLICY_SERVER_COUNT','16')); per_gpu=int(os.getenv('POLICY_SERVERS_PER_GPU','2'))
    inference_steps=int(os.getenv('POLICY_INFERENCE_STEPS','10'))
    if inference_steps<=0:raise ValueError('POLICY_INFERENCE_STEPS must be positive')
    gpu_count=torch.cuda.device_count()
    devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',') if os.getenv('CUDA_VISIBLE_DEVICES') else list(map(str,range(gpu_count)))
    if per_gpu not in (1,2) or not 1<=count<=gpu_count*per_gpu:raise ValueError('Invalid server/GPU count')
    slots=[g for g in devices for _ in range(per_gpu)][:count]
    if per_gpu==2 and any(torch.cuda.get_device_properties(i).total_memory<70*1024**3 for i in range(gpu_count)):
        raise ValueError('Two full Helios servers per GPU require the 80-GB cloud GPU allocation')
    base_port=int(os.getenv('POLICY_BASE_PORT','8200'))
    if not 1024<=base_port<=65536-count:raise ValueError('Invalid ports')
    out=runtime/'eval-server-logs'/tag; out.mkdir(parents=True,exist_ok=True)
    lock=(out/'servers.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'ready.json').unlink(missing_ok=True)
    source=run/'checkpoints/weights/step_020000.pt'
    cfg=OmegaConf.load(run/'config.yaml')
    if (cfg.model.memory_backend,cfg.model.proprio_dim,cfg.model.action_dit_config.action_dim)!=('helios',14,14):raise ValueError('Wrong model contract')
    if list(cfg.model.helios_config.history_sizes)!=[16,2,1] or cfg.model.helios_config.anchor_frames!=2:raise ValueError('Wrong history contract')
    if sha(run/'dataset_stats.json')!=cfg.robodojo_provenance.stats_sha256:raise ValueError('Statistics changed')
    payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('step')!=20000 or payload.get('memory_backend')!='helios':raise ValueError('Wrong checkpoint')
    if payload['helios_contract']['condition_augmentation']!=cfg.model.helios_config.condition_augmentation:raise ValueError('Training contract mismatch')
    del payload
    # Copy once to node-local storage rather than 16 concurrent shared-FS reads.
    cache_root=os.getenv('POLICY_LOCAL_CACHE')
    if cache_root is None:
        candidates=[Path('/tmp'),Path('/dev/shm')]
        cache_root=next((str(p/'fastwam-helios-eval') for p in candidates
                         if shutil.disk_usage(p).free>=source.stat().st_size+2*1024**3),None)
        if cache_root is None:raise RuntimeError('Set POLICY_LOCAL_CACHE to node-local storage with at least 14 GiB free')
    local=Path(cache_root)/TRAIN_RUN
    local.mkdir(parents=True,exist_ok=True)
    with (local/'copy.lock').open('a') as copy_lock:
        fcntl.flock(copy_lock,fcntl.LOCK_EX)
        print('Verifying checkpoint and preparing one node-local copy',flush=True)
        source_stat=source.stat(); digest=sha(source); checkpoint=local/'step_020000.pt'
        if not checkpoint.exists():
            if shutil.disk_usage(local).free<source_stat.st_size+2*1024**3:raise RuntimeError('Insufficient node-local disk')
            temporary=local/f'step_020000.tmp.{os.getpid()}'
            shutil.copyfile(source,temporary)
            if sha(temporary)!=digest:raise ValueError('Checkpoint copy hash mismatch')
            temporary.replace(checkpoint)
        elif sha(checkpoint)!=digest:raise ValueError('Existing local checkpoint differs')
        if (source.stat().st_size,source.stat().st_mtime_ns)!=(source_stat.st_size,source_stat.st_mtime_ns):raise ValueError('Checkpoint changed')
    deployment={'env_cfg_type':'arx_x5','action_type':'joint','config_path':str(run/'config.yaml'),
                'checkpoint_path':str(checkpoint),'prepared_root':str(cfg.data.train.prepared_root),
                'device':'cuda:0','num_inference_steps':inference_steps,'seed':42}
    (out/'deployment.json').write_text(json.dumps(deployment,indent=2))
    layout=[{'index':i,'gpu':g,'port':base_port+i} for i,g in enumerate(slots)]
    identity={'checkpoint_sha256':digest,'config_sha256':sha(run/'config.yaml'),'stats_sha256':cfg.robodojo_provenance.stats_sha256,
              'step':20000,'layout_seed':0,'policy_seed':42,'inference_steps':inference_steps,'server_count':count,'layout':layout}
    identity_path=out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Run identity changed')
    identity_path.write_text(json.dumps(identity,indent=2)); (out/'server-layout.json').write_text(json.dumps(layout,indent=2))
    print(json.dumps({'host_candidates':subprocess.check_output(['hostname','-I'],text=True).strip(),'ports':[base_port,base_port+count-1],'deployment':str(out/'deployment.json')}),flush=True)
    sock=f'/tmp/helios-text-{os.getpid()}.sock'; env=dict(os.environ,POLICY_TEXT_SOCKET=sock,POLICY_EXPECTED_STEP='20000')
    children=[]; handles=[]; policy_children=[]
    def spawn(command,path,child_env):
        log=path.open('a'); handles.append(log)
        child=subprocess.Popen(command,cwd=root,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child); return child
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        text_process=spawn([sys.executable,'-u','-B','-m','experiments.robodojo.eval_text','--socket',sock,
                          '--source',str(runtime/'prepared/text_cache'),'--output',str(out/'text_cache')],out/'text-service.log',dict(env,CUDA_VISIBLE_DEVICES=''))
        deadline=time.monotonic()+300
        while not Path(sock).exists():
            if text_process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Text service failed to start')
            time.sleep(1)
        log_offsets={}
        def launch_policy(row):
            index=row['index']
            log_path=out/f'server-{index}.log'
            log_offsets[index]=log_path.stat().st_size if log_path.exists() else 0
            child=spawn([sys.executable,'-u','-B','-m','experiments.robodojo.serve','--deployment',str(out/'deployment.json'),
                         '--host','0.0.0.0','--port',str(row['port'])],log_path,dict(env,CUDA_VISIBLE_DEVICES=row['gpu']))
            policy_children.append(child)
        def check_loading():
            if text_process.poll() is not None:raise RuntimeError('Text service exited while loading')
            if any(p.poll() is not None for p in policy_children):raise RuntimeError('Policy exited while loading; inspect server logs')
        def probe_policy(row):
            index=row['index']
            # Wait for this process's ready marker before a real RPC, avoiding
            # expected connection-refused errors while weights are loading.
            with (out/f'server-{index}.log').open('rb') as log:
                log.seek(log_offsets[index])
                if b'HELIOS_SERVER_READY' not in log.read():return False
            try:
                client=WsModelClient(url=f'ws://127.0.0.1:{row["port"]}',evaluation_id=tag,trial_id=f'health-{index}',max_connect_attempts=1,request_timeout_s=30)
                try:status=client.call(func_name='status')
                finally:client.close()
            except Exception as exc:
                print(f'SERVER_RPC_RETRY {index}: {exc}',flush=True)
                return False
            if status['checkpoint_step']!=20000 or status['inference_steps']!=inference_steps:raise ValueError('RPC contract mismatch')
            return True
        launch_policy_batches(layout,launch_policy,probe_policy,check_loading)
        check_loading()
        ready={'status':'ready','identity':identity,'host':socket.gethostname()}
        (out/'ready.json').write_text(json.dumps(ready,indent=2)); print('ALL_HELIOS_SERVERS_READY',flush=True)
        while True:
            codes=[p.poll() for p in policy_children]
            if any(c not in (None,0) for c in codes):raise RuntimeError(f'Policy server failed: {codes}')
            if all(c==0 for c in codes):return
            if text_process.poll() is not None:raise RuntimeError('Text service exited')
            time.sleep(3)
    finally:
        (out/'ready.json').unlink(missing_ok=True)
        for child in reversed(children):stop(child)
        for log in handles:log.close()
        Path(sock).unlink(missing_ok=True)


if __name__=='__main__':main()
