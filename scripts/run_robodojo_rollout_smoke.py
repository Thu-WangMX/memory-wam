"""Bounded real IsaacLab rollout; owns and cleans up its policy/client processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def stop(process):
    if process is None:
        return
    # A shell can exit while IsaacSim descendants remain in its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy-gpu', type=int, required=True)
    parser.add_argument('--sim-gpu', type=int, required=True)
    parser.add_argument('--task', default='cover_blocks')
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    if args.policy_gpu == args.sim_gpu or args.steps <= 0:
        parser.error('Use distinct policy/simulator GPUs and positive steps')
    for gpu, minimum in ((args.policy_gpu, 18000), (args.sim_gpu, 10000)):
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < minimum:
            raise RuntimeError(f'GPU {gpu} has only {free} MiB free, requires {minimum}')
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    server_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.policy_gpu))
    deployment = str(Path(args.deployment).resolve())
    run_name = 'anchor_helios_smoke_' + time.strftime('%Y%m%d_%H%M%S')
    result_path = output / 'simulator_smoke.json'
    client_env = dict(os.environ, POLICY_HOST='127.0.0.1', POLICY_PORT=str(port),
                      SIM_GPU_ID=str(args.sim_gpu), EVAL_TASK=args.task, EVAL_RUN_ID=run_name,
                      EVAL_NUM='1', EVAL_SEED='0', ANCHOR_HELIOS_SMOKE_STEPS=str(args.steps),
                      ANCHOR_HELIOS_SMOKE_OUTPUT=str(result_path), ROBODOJO_MAX_BASH_RETRIES='1')
    server = client = None
    try:
        with (output / 'server.log').open('w') as server_log, (output / 'client.log').open('w') as client_log:
            server = subprocess.Popen(['bash', 'experiments/robodojo/serve.sh', '--deployment', deployment,
                                       '--host', '127.0.0.1', '--port', str(port)],
                                      cwd=root, env=server_env, stdout=server_log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server exited with {server.returncode}; see {output}/server.log')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except (ConnectionError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Policy server did not become ready within 600 seconds')
                    time.sleep(2)
            print(f'Policy ready; starting simulator on GPU {args.sim_gpu}', flush=True)
            client = subprocess.Popen(['bash', 'experiments/robodojo/eval_client.sh'], cwd=root,
                                      env=client_env, stdout=client_log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            deadline = time.monotonic() + 900
            while client.poll() is None and not result_path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Simulator smoke exceeded 900 seconds')
                time.sleep(1)
            if not result_path.is_file():
                code = client.poll()
                raise RuntimeError(f'Rollout smoke failed (exit={code}); see {output}/client.log')
            # IsaacSim can stall during extension teardown after SystemExit.
            # The report is written after the requested actions and before scoring.
            # Give teardown a grace period, then clean up only our process group.
            try:
                client.wait(timeout=20)
            except subprocess.TimeoutExpired:
                stop(client)
            result = json.loads(result_path.read_text())
            if result['control_steps'] != args.steps or result['benchmark_result'] is not False:
                raise ValueError('Incorrect smoke termination/report')
            print(json.dumps(result, indent=2), flush=True)
    finally:
        stop(client)
        stop(server)


if __name__ == '__main__':
    main()
