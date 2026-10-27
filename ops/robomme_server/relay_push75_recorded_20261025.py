"""Finite Push75 5k -> 7.5k -> 10k recorded evaluation; wait for completed weights."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

REPO = Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')
RUNTIME = Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme')
RUN = REPO / 'runs/robomme_all16_push75_past32x8_future2_dense1_8gpu_b32ga1_s20k_20261024_012027'
STEPS = (5000, 7500, 10000)
EXISTING_5K = RUNTIME / 'evals' / RUN.name / 'step_005000_test50_d10_video_push75_5k_recorded_20261024_153859'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def free_slots():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    free = {int(g): int(m) for g, m in (row.split(',') for row in raw.splitlines())}
    # Do not add a second RoboMME policy to a card already running one.
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                       '--format=csv,noheader,nounits'], text=True)
    ids = {u.strip(): int(g) for g, u in (row.split(',') for row in gpu_info.splitlines())}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    busy = set()
    for row in processes.splitlines():
        uuid, pid = (x.strip() for x in row.split(','))
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
        except (FileNotFoundError, PermissionError):
            continue
        if b'experiments/robomme/serve.py' in command and uuid in ids:
            busy.add(ids[uuid])
    return free, {g: 1 for g, m in free.items() if m >= 19968 and g not in busy}


def complete(summary):
    return (summary.get('status') == 'complete' and summary.get('completed') == 800
            and summary.get('expected') == 800 and summary.get('videos_saved') == 800
            and summary.get('save_video') is True and not summary.get('errors'))



def checkpoint_ready(path):
    """Require the final Torch ZIP directory and a 60-second quiet period."""
    try:
        stat = path.stat()
        if not path.is_file() or not stat.st_size:
            return False, 'missing_or_empty'
        if time.time() - stat.st_mtime < 60:
            return False, 'recent_write'
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not any(name.endswith('/data.pkl') for name in names):
                return False, 'missing_torch_metadata'
            if not any('/data/' in name for name in names):
                return False, 'missing_tensor_storage'
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return False, 'still_changing'
        return True, 'ready'
    except (OSError, zipfile.BadZipFile, EOFError):
        return False, 'missing_or_incomplete'


def worker(args):
    from experiments.robomme import campaign
    free, counts = free_slots()
    if not counts:
        return 75  # supervisor waits for a free slot, without changing other jobs
    sockets = []
    for base in range(27400, 29000, 40):
        try:
            for port in range(base, base + sum(counts.values())):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(('127.0.0.1', port))
            break
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
            sockets.clear()
    else:
        raise RuntimeError('No available policy port range')

    def lanes(gpus, unused, port_base):
        result = []
        for gpu in gpus:
            for slot in range(counts[gpu]):
                result.append((gpu, f'{gpu}_{slot}', port_base + len(result)))
        return result

    campaign.build_lanes = lanes
    output = Path(args.output)
    command = ['campaign.py', '--run', str(RUN), '--checkpoint', f'step_{args.step:06d}.pt',
               '--output', str(output), '--gpus', *map(str, counts),
               '--lanes-per-gpu', str(max(counts.values())), '--episodes', '50',
               '--chunk-size', '5', '--split', 'test', '--max-steps', '1300',
               '--steps', '5', '--seed', '42', '--port-base', str(base), '--save-video',
               '--policy-memory-gib', '17', '--sim-memory-gib', '1', '--fixed-memory',
               '--dedicated-gpus', '--gpu-budget-gib', '80', '--robodojo-reserve-gib', '0',
               '--memory-margin-gib', '2', '--server-startup-timeout', '3600']
    if (output / 'manifest.json').exists():
        command.append('--resume')
    placement = dict(pid=os.getpid(), at=now(), step=args.step, output=str(output),
                     free_mib=free, lane_counts=counts, lanes=lanes(list(counts), 1, base),
                     command=command)
    write(Path(args.relay_dir) / f'placement_{args.step:06d}.json', placement)
    print('PLACEMENT ' + json.dumps(placement), flush=True)
    sys.argv = command
    return campaign.main()


def supervise(args):
    root = Path(args.relay_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / 'launchers/push75_20261024_012027_test50_d10_relay.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), host=socket.gethostname(), started_at=now(),
                 order=list(STEPS), phase='starting', runs={})
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def update(**values):
        state.update(values, updated_at=now())
        write(root / 'status.json', state)

    for step in STEPS:
        output = RUNTIME / 'evals' / RUN.name / f'step_{step:06d}_test50_d10_video_{root.name}'
        if step == 5000:
            output = EXISTING_5K
        state['runs'][str(step)] = dict(output=str(output), phase='queued')
    update()
    for step in STEPS:
        item = state['runs'][str(step)]
        output = Path(item['output'])
        attempts = 0
        while not stopped:
            if (output / 'summary.json').exists():
                summary = json.loads((output / 'summary.json').read_text())
                if complete(summary):
                    item.update(phase='complete', successes=summary['successes'],
                                success_rate=summary['success_rate'], videos_saved=800)
                    update()
                    break
            if attempts >= 2:
                item['phase'] = 'failed'
                update(phase='failed', reason='Two attempts incomplete; inspect campaign logs')
                return 1
            ready, reason = checkpoint_ready(RUN / 'checkpoints/weights' / f'step_{step:06d}.pt')
            if not ready:
                item.update(phase='waiting_for_checkpoint', reason=reason)
                update(phase='waiting_for_checkpoint', current_step=step)
                time.sleep(60)
                continue
            free, slots = free_slots()
            if not slots:
                update(phase='waiting_for_memory', current_step=step, free_mib=free, minimum_free_mib=20480)
                time.sleep(60)
                continue
            attempts += 1
            item.update(phase='running', attempt=attempts)
            import shlex
            command = ['python', '-u', str(Path(__file__).resolve()), '--worker',
                       '--step', str(step), '--output', str(output), '--relay-dir', str(root)]
            with (root / f'step_{step:06d}_attempt_{attempts}.log').open('a') as log:
                child = subprocess.Popen(['bash', '-c', 'source experiments/robomme/env.sh && exec ' + shlex.join(command)],
                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                update(phase='running', current_step=step, child_pid=child.pid)
                code = child.wait()
            item['last_exit_code'] = code
            update(child_pid=None)
            if code == 75:
                attempts -= 1
                time.sleep(60)
        if stopped:
            update(phase='stopped', child_pid=None)
            return 130
    update(phase='complete', current_step=None, child_pid=None, finished_at=now())
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--relay-dir', required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--step', type=int, choices=STEPS)
    p.add_argument('--output')
    args = p.parse_args()
    raise SystemExit(worker(args) if args.worker else supervise(args))
