"""Keep idle A100 lanes reserved while its peer finishes an evaluation."""
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path('/mnt/vepfs01/output/spidy.wang/runtime-robomme/launchers')
WORK = ROOT / 'watch_push75_and_wsd_20261025'
CONTROL = WORK / 'tail_reservations'
PYTHON = '/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv/bin/python'

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def pause_for_startup():
    CONTROL.mkdir(exist_ok=True)
    # The next worker releases only this guard's allocations, before GPU selection.
    (CONTROL / 'pause_until').write_text(str(time.time() + 300))
    state = read(CONTROL / 'status.json')
    for pid in state.get('holds', {}).values():
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'robomme_gpu_reservation.py' in cmd and str(CONTROL).encode() in cmd:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    deadline = time.time() + 15
    while time.time() < deadline:
        if not any(Path(f'/proc/{p}').exists() for p in state.get('holds', {}).values()):
            break
        time.sleep(.2)

def main():
    CONTROL.mkdir(exist_ok=True)
    lock = (CONTROL / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('base', ROOT / 'relay_push75_recorded_20261025.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    holds = {}
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopped:
            try:
                holds = {g: p for g, p in holds.items() if p.poll() is None}
                state = read(WORK / 'status.json')
                job = state.get('jobs', {}).get(state.get('current_job'), {})
                allocation = read(Path(job.get('output', '/nonexistent')) / 'allocation.json')
                local_done = bool(allocation.get('generation')) and (Path(allocation['generation']) / 'a100' / 'finished.json').exists()
                pause = CONTROL / 'pause_until'
                paused = pause.exists() and float(pause.read_text()) > time.time()
                enabled = state.get('phase') == 'evaluating' and local_done and not paused
                if not enabled:
                    for p in holds.values():
                        p.terminate()
                    for p in holds.values():
                        p.wait(timeout=15)
                    holds.clear()
                else:
                    free, slots = base.free_slots()
                    for g in range(8):
                        if g in holds or g not in slots or free.get(g, 0) < 19 * 1024:
                            continue
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                        with (CONTROL / f'gpu{g}.log').open('a') as log:
                            holds[g] = subprocess.Popen([PYTHON, '-u', str(ROOT / 'robomme_gpu_reservation.py'), '--hold', '--gpu', str(g), '--root', str(CONTROL)], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                base.write(CONTROL / 'status.json', dict(pid=os.getpid(), updated_at=base.now(), enabled=enabled, local_done=local_done, holds={str(g): p.pid for g, p in holds.items()}))
            except Exception as exc:
                print(type(exc).__name__, str(exc), flush=True)
            time.sleep(2)
    finally:
        for p in holds.values():
            if p.poll() is None:
                p.terminate()

if __name__ == '__main__':
    main()
