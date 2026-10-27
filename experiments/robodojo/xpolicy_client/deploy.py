"""Preserve every executed-step observation for the stateful Helios server."""
import json
import os
from pathlib import Path


def eval_one_episode(TASK_ENV, model_client):
    smoke_steps = int(os.environ.get('ANCHOR_HELIOS_SMOKE_STEPS', '0'))
    smoke_output = os.environ.get('ANCHOR_HELIOS_SMOKE_OUTPUT')
    if smoke_steps < 0 or (smoke_steps and not smoke_output):
        raise ValueError('Bounded rollout smoke requires positive steps and a dedicated output path')
    if smoke_steps and Path(smoke_output).exists():
        raise FileExistsError('Use a fresh smoke output path')
    executed = 0
    requests = 0
    model_client.call(func_name='reset')
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name='update_obs', obs=obs)
        actions = model_client.call(func_name='get_action')
        if len(actions) != 16:
            raise ValueError(f'Anchor-Helios must return 16 actions, got {len(actions)}')
        requests += 1
        for index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            executed += 1
            if smoke_steps and executed >= smoke_steps:
                path = Path(smoke_output)
                path.parent.mkdir(parents=True, exist_ok=True)
                report = {'status': 'bounded_rollout_passed', 'control_steps': executed,
                          'action_requests': requests, 'benchmark_result': False,
                          'note': 'Intentionally stopped before scoring; wiring smoke only.'}
                path.write_text(json.dumps(report, indent=2) + '\n')
                print('ANCHOR_HELIOS_SMOKE_COMPLETE ' + json.dumps(report), flush=True)
                # Do not let a truncated rollout enter the benchmark result aggregator.
                raise SystemExit(0)
            if TASK_ENV.is_episode_end() or index + 1 == len(actions):
                break
            model_client.call(func_name='update_obs', obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError('Anchor-Helios requires eval_batch=false and one client per policy')
