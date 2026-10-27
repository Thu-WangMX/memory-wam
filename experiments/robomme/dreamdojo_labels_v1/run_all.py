#!/usr/bin/env python3
"""Run one frozen-label worker per selected GPU and finalize on success."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme/dreamdojo_labels_v1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    gpus = args.gpus.split(',')
    assert len(set(gpus)) == len(gpus) and all(g.isdigit() for g in gpus)
    logs = args.base / 'logs'
    logs.mkdir(exist_ok=True)
    lock = (args.base / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    script = Path(__file__).with_name('extract_labels.py')
    subprocess.run([sys.executable, str(script), 'init', '--base', str(args.base)], check=True)
    processes = []
    started = time.time()
    try:
        for rank, gpu in enumerate(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            command = [sys.executable, str(script), 'worker', '--base', str(args.base),
                       '--rank', str(rank), '--world-size', str(len(gpus)), '--batch-size', str(args.batch_size)]
            with (logs / f'rank{rank}.log').open('a') as log:
                proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            processes.append(proc)
            print(json.dumps({'event':'started','rank':rank,'gpu':gpu,'pid':proc.pid}), flush=True)
        (args.base / 'workers.json').write_text(json.dumps({'controller_pid':os.getpid(),'started_at':started,
             'gpus':gpus,'pids':[p.pid for p in processes], 'batch_size':args.batch_size}, indent=2)+'\n')
        codes = [p.wait() for p in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f'workers failed: {codes}; completed episodes are reusable')
        subprocess.run([sys.executable, str(script), 'finalize', '--base', str(args.base)], check=True)
        result = {'status':'complete','started_at':started,'completed_at':time.time(),'exit_codes':codes}
    except BaseException:
        # Only terminate this controller's own children on interruption.
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        raise
    (args.base / 'completion.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
