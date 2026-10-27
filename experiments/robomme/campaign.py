"""Run fixed official RoboMME episode ranges on explicitly allocated GPUs.

Each lane owns one policy server and at most one simulator client. Completed
episodes are retained after a client failure; retries start at the first missing
episode. Missing results are reported separately and never counted as failures.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import socket
import subprocess
import threading
import time
from urllib.request import build_opener, ProxyHandler
from experiments.robomme.memory_budget import MemoryGuard


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def collect_results(output, manifest):
    records = {task: {} for task in manifest['tasks']}
    for path in sorted((Path(output) / 'tasks').glob('*/chunk_*/attempt_*/summary.json')):
        try:
            part = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # A live client may currently be writing its next result.
        task = part['task']
        if (task not in records or part['split'] != manifest['split']
                or part['max_steps'] != manifest['max_steps']
                or part['contract']['checkpoint'] != manifest['checkpoint']
                or part['contract']['data_fingerprint'] != manifest['data_fingerprint']
                or part['contract']['denoising_steps'] != manifest['denoising_steps']
                or part['contract']['seed'] != manifest['seed']):
            raise ValueError(f'Inconsistent evaluation provenance: {path}')
        for row in part['results']:
            episode = row['episode']
            if episode not in manifest['episode_ids'] or episode in records[task]:
                raise ValueError(f'Duplicate or unrequested episode: {task}/{episode}')
            if row['success'] != (row['status'] == 'success'):
                raise ValueError(f'Invalid success label: {task}/{episode}')
            if manifest.get('save_video',False):
                video=row.get('video',{})
                video_path=path.parent/video.get('path','')
                if (video.get('kind')!='original_rollout' or not video_path.is_file()
                        or video_path.stat().st_size!=video.get('bytes')):
                    raise ValueError(f'Missing scored-rollout video: {task}/{episode}')
            records[task][episode] = dict(row, artifact=str(path))
    tasks = {}
    expected = len(manifest['episode_ids'])
    for task, episodes in records.items():
        successes = sum(row['success'] for row in episodes.values())
        tasks[task] = dict(completed=len(episodes), expected=expected, successes=successes,
            environment_errors=sum(row['status'] == 'error' for row in episodes.values()),
            success_rate=successes / expected if len(episodes) == expected else None,
            missing_episode_ids=[i for i in manifest['episode_ids'] if i not in episodes],
            results=[episodes[i] for i in sorted(episodes)])
    completed = sum(row['completed'] for row in tasks.values())
    successes = sum(row['successes'] for row in tasks.values())
    total = expected * len(tasks)
    return dict(updated_at=now(), checkpoint=manifest['checkpoint'], split=manifest['split'],
        expected=total, completed=completed, successes=successes,
        save_video=manifest.get('save_video',False),
        videos_saved=sum(bool(row.get('video')) for episodes in records.values() for row in episodes.values()),
        success_rate_completed=successes / completed if completed else None,
        success_rate=successes / total if completed == total else None, tasks=tasks)


def pending_chunks(manifest, summary, chunk_size):
    """Yield each missing episode exactly once, retaining recorded failures too."""
    preferred = ['PickXtimes', 'VideoPlaceOrder', 'PatternLock']
    ordered = [task for task in preferred if task in manifest['tasks']]
    ordered += [task for task in manifest['tasks'] if task not in ordered]
    missing = {task: set(summary['tasks'][task]['missing_episode_ids']) for task in ordered}
    for start in range(0, len(manifest['episode_ids']), chunk_size):
        end = min(start + chunk_size, len(manifest['episode_ids']))
        for task in ordered:
            position = start
            while position < end:
                if position not in missing[task]:
                    position += 1
                    continue
                first = position
                while position < end and position in missing[task]:
                    position += 1
                yield task, first, position - first


def validate_resume(previous, current):
    if previous.get('save_video',False)!=current.get('save_video',False):
        raise ValueError('Resume changes evaluation provenance: save_video; start a new campaign')
    for key in ('run', 'config', 'config_sha256', 'checkpoint', 'checkpoint_sha256',
                'data_fingerprint', 'history_sizes', 'anchor_frames', 'tasks',
                'episode_ids', 'split', 'max_steps', 'denoising_steps', 'seed',
                'official_episode_metadata'):
        if previous[key] != current[key]:
            raise ValueError(f'Resume changes evaluation provenance: {key}')


def build_lanes(gpus, lanes_per_gpu, port_base):
    if (not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0
            or lanes_per_gpu < 1 or port_base < 1
            or port_base + len(gpus)*lanes_per_gpu > 65536):
        raise ValueError('Invalid GPU allocation, lane count or port range')
    return [(gpu, str(gpu) if lanes_per_gpu == 1 else f'{gpu}_{slot}',
             port_base + i*lanes_per_gpu + slot)
            for i, gpu in enumerate(gpus) for slot in range(lanes_per_gpu)]


def validate_memory_args(args):
    if min(args.policy_memory_gib, args.sim_memory_gib, args.gpu_budget_gib) <= 0:
        raise ValueError('Memory budgets must be positive')
    if args.dedicated_gpus:
        if not args.fixed_memory or args.robodojo_reserve_gib != 0:
            raise ValueError('Dedicated GPUs require --fixed-memory and --robodojo-reserve-gib 0')
        # Driver/Vulkan allocations are outside PyTorch's allocator caps.
        required = args.lanes_per_gpu*(args.policy_memory_gib + args.sim_memory_gib + 2)
        if args.memory_margin_gib < 2 or required + args.memory_margin_gib > args.gpu_budget_gib:
            raise ValueError('Per-lane allocations and renderer overhead exceed the GPU budget')
    elif not (args.lanes_per_gpu == 1
            and args.policy_memory_gib + args.sim_memory_gib < args.stop_at_gib
            < args.gpu_budget_gib <= 35 and args.robodojo_reserve_gib >= 45
            and args.memory_margin_gib >= 2):
        raise ValueError('Shared GPUs require one lane, RoboMME <=35 GiB and RoboDojo reserve >=45 GiB')


def stop_process(process):
    if process is not None and process.poll() is None:
        # Only process groups created by this campaign are signalled.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


class Campaign:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.output = Path(args.output)
        self.repo = Path(__file__).resolve().parents[2]
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.jobs = queue.Queue()
        self.lanes, self.errors = {}, []
        self.guard = None
        if not args.fixed_memory:
            self.guard = MemoryGuard(self.output, args.nvml_path, args.gpus,
                args.stop_at_gib, args.gpu_budget_gib, args.robodojo_reserve_gib,
                args.memory_margin_gib, self.memory_violation)
            for gpu, card in self.guard.last_sample['cards'].items():
                if card['early_stop_gib'] <= args.policy_memory_gib + args.sim_memory_gib + 2:
                    raise RuntimeError(f'GPU {gpu}: insufficient memory after RoboDojo reservation: {card}')
        else:
            write_json(self.output / 'memory_watch.json', dict(mode='fixed_allocator_caps',
                dynamic_watch_enabled=False, updated_at=now(),
                policy_allocator_gib=args.policy_memory_gib,
                simulator_allocator_gib=args.sim_memory_gib,
                lanes_per_gpu=args.lanes_per_gpu, dedicated_gpus=args.dedicated_gpus,
                note='CUDA allocator caps exclude driver and Vulkan allocations'))
        previous = collect_results(self.output, manifest)
        for job in pending_chunks(manifest, previous, args.chunk_size):
            self.jobs.put(job)
        print(f'QUEUE retained={previous["completed"]} missing={previous["expected"]-previous["completed"]} '
              f'chunks={self.jobs.qsize()}', flush=True)

    def memory_violation(self, message):
        with self.lock:
            self.errors.append(dict(memory_guard=message))
        self.stop.set()
        print('MEMORY_GUARD_STOP '+message, flush=True)

    def update(self, lane=None, **values):
        with self.lock:
            if lane is not None:
                self.lanes.setdefault(str(lane), {}).update(values)
            write_json(self.output / 'status.json', dict(updated_at=now(), pid=os.getpid(),
                host=socket.gethostname(), lanes=self.lanes, errors=self.errors,
                queued_chunks=self.jobs.qsize()))

    def run_client(self, gpu, lane, port, task, start, count, server):
        base = self.output / 'tasks' / task / f'chunk_{start:03d}_{start+count-1:03d}'
        next_episode = start
        existing = [int(path.name.split('_')[1].split('.')[0])
                    for path in base.glob('attempt_*')]
        first_attempt = max(existing, default=0) + 1
        for attempt in range(first_attempt, first_attempt + 2):
            if self.stop.is_set():
                return
            if server.poll() is not None:
                raise RuntimeError(f'Policy server exited: GPU {gpu}')
            output = base / f'attempt_{attempt}'
            output.parent.mkdir(parents=True, exist_ok=True)
            command = ['bash', 'experiments/robomme/eval.sh', '--task', task,
                '--split', self.args.split, '--episodes', str(start + count - next_episode),
                '--episode-start', str(next_episode), '--max-steps', str(self.args.max_steps),
                '--port', str(port), '--contract',
                str(self.output / 'servers' / f'gpu{lane}' / 'inference_contract.json'),
                '--output', str(output), '--memory-gib', str(self.args.sim_memory_gib)]
            command.append('--save-video' if self.args.save_video else '--no-save-video')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
            process = None
            with (base / f'attempt_{attempt}.log').open('w') as log:
                try:
                    process = subprocess.Popen(command, cwd=self.repo, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    if self.guard is not None:
                        self.guard.register(process)
                    self.update(lane, phase='evaluating', task=task, episode_start=next_episode,
                        episode_end=start + count - 1, client_pid=process.pid, attempt=attempt)
                    deadline = time.monotonic() + self.args.chunk_timeout
                    while process.poll() is None and not self.stop.wait(2):
                        if server.poll() is not None:
                            raise RuntimeError(f'Policy server exited: GPU {gpu}')
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'Client exceeded {self.args.chunk_timeout}s: {task}')
                    if self.stop.is_set():
                        return
                finally:
                    stop_process(process)
                    if self.guard is not None:
                        self.guard.unregister(process)
            partial = output / 'summary.json'
            if partial.exists():
                rows = json.loads(partial.read_text())['results']
                ids = [row['episode'] for row in rows]
                if ids != list(range(next_episode, next_episode + len(rows))):
                    raise ValueError('Non-contiguous client progress')
                next_episode += len(rows)
            if next_episode == start + count:
                print(f'CHUNK_DONE gpu={gpu} lane={lane} task={task} episodes={start}:{start+count}', flush=True)
                return
            print(f'CLIENT_RETRY gpu={gpu} lane={lane} task={task} next={next_episode} exit={process.returncode}', flush=True)
        with self.lock:
            self.errors.append(dict(task=task, missing=list(range(next_episode, start + count)),
                                    message='Client failed twice; see attempt logs'))

    def worker(self, gpu, lane, port):
        server = None
        server_dir = self.output / 'servers' / f'gpu{lane}'
        server_dir.mkdir(parents=True, exist_ok=True)
        server_log = (server_dir / 'server.log').open('a')
        server_log.write(f'\nSERVER_START {now()}\n')
        server_log.flush()
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
            command = ['python', '-B', 'experiments/robomme/serve.py',
                '--config', self.manifest['config'], '--checkpoint', self.manifest['checkpoint'],
                '--port', str(port), '--steps', str(self.args.steps), '--seed', str(self.args.seed),
                '--output', str(server_dir), '--memory-gib', str(self.args.policy_memory_gib)]
            shell = 'source experiments/robomme/env.sh && exec ' + shlex.join(command)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4',
                FASTWAM_FFN_CHUNK_SIZE='256', FASTWAM_QKV_CHUNK_SIZE='256',
                TRITON_CACHE_DIR=str(self.output / 'cache' / f'gpu{lane}'))
            server = subprocess.Popen(['bash', '-lc', shell], cwd=self.repo, env=env,
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            if self.guard is not None:
                self.guard.register(server)
            self.update(lane, gpu=gpu, phase='loading_model', server_pid=server.pid, port=port)
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + self.args.server_startup_timeout
            while not self.stop.wait(2):
                if server.poll() is not None:
                    raise RuntimeError(f'Policy startup failed: {server_dir / "server.log"}')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/healthz', timeout=3) as response:
                        ready = response.status == 200
                except OSError:
                    ready = False
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Policy startup exceeded {self.args.server_startup_timeout}s')
            if self.stop.is_set():
                return
            self.update(lane, phase='ready')
            print(f'LANE_READY gpu={gpu} lane={lane} port={port}', flush=True)
            while not self.stop.is_set():
                try:
                    task, start, count = self.jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.run_client(gpu, lane, port, task, start, count, server)
                finally:
                    self.jobs.task_done()
            self.update(lane, phase='finished', client_pid=None)
        except Exception as exc:
            with self.lock:
                self.errors.append(dict(gpu=gpu, lane=lane, message=repr(exc)))
            self.update(lane, phase='error', error=repr(exc))
            self.stop.set()
            raise
        finally:
            stop_process(server)
            if self.guard is not None:
                self.guard.unregister(server)
            server_log.close()

    def run(self):
        def interrupted(signum, frame):
            self.stop.set()
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        if self.guard is not None:
            self.guard.start()
        failures = []
        lanes = build_lanes(self.args.gpus, self.args.lanes_per_gpu, self.args.port_base)
        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            futures = [pool.submit(self.worker, gpu, lane, port) for gpu, lane, port in lanes]
            while not all(future.done() for future in futures):
                try:
                    write_json(self.output / 'summary.json', collect_results(self.output, self.manifest))
                except Exception:
                    self.stop.set()
                    raise
                time.sleep(10)
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    failures.append(repr(exc))
        summary = collect_results(self.output, self.manifest)
        if self.guard is not None:
            self.guard.close()
        summary['errors'] = self.errors
        summary['status'] = ('complete' if summary['completed'] == summary['expected']
                             and not failures and not self.errors else 'incomplete')
        write_json(self.output / 'summary.json', summary)
        self.update()
        print('CAMPAIGN_FINISHED ' + json.dumps({k: summary[k] for k in
              ('status', 'completed', 'expected', 'successes', 'success_rate')}), flush=True)
        return 0 if summary['status'] == 'complete' else 1


def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', default='step_020000.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--save-video',action=argparse.BooleanOptionalAction,default=True,
                        help='Save actual rollout video for every episode (default: on)')
    parser.add_argument('--resume', action='store_true', help='Continue only missing episodes in an existing output')
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--lanes-per-gpu', type=int, default=1)
    parser.add_argument('--dedicated-gpus', action='store_true',
                        help='Use explicitly allocated full GPUs without the shared RoboDojo reservation')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--chunk-size', type=int, default=5)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max-steps', type=int, default=1300,
                        help='MME-VLA evaluation uses 1300; RoboMME Challenge uses 1500')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--port-base', type=int, default=18260)
    parser.add_argument('--chunk-timeout', type=int, default=21600)
    parser.add_argument('--server-startup-timeout',type=int,default=1200,
                        help='Seconds allowed for loading models from shared storage')
    parser.add_argument('--policy-memory-gib', type=float, default=20)
    parser.add_argument('--sim-memory-gib', type=float, default=2)
    parser.add_argument('--stop-at-gib', type=float, default=33)
    parser.add_argument('--gpu-budget-gib', type=float, default=35)
    parser.add_argument('--robodojo-reserve-gib', type=float, default=45)
    parser.add_argument('--memory-margin-gib', type=float, default=2)
    parser.add_argument('--fixed-memory', action='store_true',
                        help='Use allocator caps only, without dynamic GPU monitoring or memory-triggered shutdown')
    parser.add_argument('--nvml-path', default='/mnt/vepfs02/output/kevin_wang/memorywam/robomme_native/.venv/lib/python3.11/site-packages/pynvml.py')
    args = parser.parse_args()
    lanes = build_lanes(args.gpus, args.lanes_per_gpu, args.port_base)
    if min(args.episodes, args.chunk_size, args.max_steps, args.steps, args.chunk_timeout, args.server_startup_timeout) < 1:
        raise ValueError('Episode counts and limits must be positive')
    validate_memory_args(args)
    run = Path(args.run).resolve()
    output = Path(args.output).resolve()
    checkpoint = run / 'checkpoints' / 'weights' / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = OmegaConf.load(run / 'config.yaml')
    tasks = list(cfg.robomme_provenance.semantic_tasks)
    if len(tasks) != 16 or len(set(tasks)) != 16:
        raise ValueError('This campaign expects all 16 RoboMME tasks')
    sim = Path(os.getenv('ROBOMME_SIM_ROOT', '/mnt/vepfs02/output/kevin_wang/memorywam/robomme_native'))
    metadata = {}
    for task in tasks:
        path = sim / 'src/src/robomme/env_metadata' / args.split / f'record_dataset_{task}_metadata.json'
        raw = json.loads(path.read_text())
        indexed = {int(row['episode']): row for row in raw['records']}
        metadata[task] = dict(sha256=digest(path), records=[indexed[i] for i in range(args.episodes)])
    if args.resume:
        if not (output / 'manifest.json').is_file():
            raise FileNotFoundError('Resume requires an existing evaluation manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    args.output = str(output)
    manifest = dict(created_at=now(), run=str(run), config=str(run / 'config.yaml'),
        config_sha256=digest(run / 'config.yaml'), checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), data_fingerprint=cfg.model.helios_config.data_fingerprint,
        history_sizes=list(cfg.model.helios_config.history_sizes), anchor_frames=cfg.model.helios_config.anchor_frames,
        tasks=tasks, episode_ids=list(range(args.episodes)), split=args.split,
        max_steps=args.max_steps, denoising_steps=args.steps, seed=args.seed,
        save_video=args.save_video,video_format='front_wrist_native_rgb_h264',
        gpu_indices=args.gpus, host=socket.gethostname(), official_episode_metadata=metadata,
        lanes_per_gpu=args.lanes_per_gpu,
        lanes=[dict(gpu=gpu, lane=lane, port=port) for gpu, lane, port in lanes],
        memory_budget=dict(policy_allocator_gib=args.policy_memory_gib,
            dedicated_gpus=args.dedicated_gpus, lanes_per_gpu=args.lanes_per_gpu,
            mode='fixed_allocator_caps' if args.fixed_memory else 'dynamic_watch',
            simulator_allocator_gib=args.sim_memory_gib, combined_early_stop_gib=args.stop_at_gib,
            requested_per_gpu_gib=args.gpu_budget_gib, robodojo_reserve_gib=args.robodojo_reserve_gib,
            safety_margin_gib=args.memory_margin_gib,
            formula='min(cap, allocatable-max(robodojo_used,reserve)-other_used-unattributed)',
            scope='owned inference and simulator process groups'))
    import fcntl
    with (output / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            previous = json.loads((output / 'manifest.json').read_text())
            validate_resume(previous, manifest)
            # Also reject a still-running pre-locking controller from older code.
            old_status = json.loads((output / 'status.json').read_text())
            old_pid = old_status['pid']
            try:
                argv = Path(f'/proc/{old_pid}/cmdline').read_bytes().split(b'\0')
            except FileNotFoundError:
                argv = []
            if str(output).encode() in argv and b'experiments/robomme/campaign.py' in argv:
                raise RuntimeError(f'Previous campaign is still running: PID {old_pid}')
            archive = output / 'resume_history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            archive.mkdir(parents=True)
            for name in ('manifest.json', 'status.json', 'summary.json', 'memory_watch.json'):
                path = output / name
                if path.exists():
                    (archive / name).write_bytes(path.read_bytes())
            manifest['created_at'] = previous['created_at']
            manifest['last_resumed_at'] = now()
        write_json(output / 'manifest.json', manifest)
        print('CAMPAIGN_START ' + str(output), flush=True)
        return Campaign(args, manifest).run()


if __name__ == '__main__':
    raise SystemExit(main())
