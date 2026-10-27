"""Run all available tasks for one checkpoint in a single policy process."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/RoboTwin/script'))
import eval_policy as harness


def queue_status(cfg):
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                'http://127.0.0.1:%s/status' % cfg['port'], data=b'{}',
                headers={'Authorization': 'Bearer ' + cfg['token']})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def run(base, cfg, evaluate=harness.main):
    checkpoint = str(Path(base['ckpt_setting']).resolve())
    groups = {g: s for g, s in cfg['groups'].items()
              if str(Path(s['checkpoint']).resolve()) == checkpoint}
    assert groups
    assert all(str(Path(s['stats']).resolve()) == str(Path(base['dataset_stats_path']).resolve()) for s in groups.values())
    output = Path(base['eval_output_dir']).parent / 'persistent_tasks'
    model = None
    visits = 0
    while True:
        status = queue_status(cfg)
        remaining = [g for g in groups if status[g].get('done', 0) < groups[g]['target']]
        if not remaining:
            print('PERSISTENT_CHECKPOINT_COMPLETE', checkpoint, flush=True)
            return
        available = [g for g in remaining if status[g].get('done', 0) + status[g].get('leased', 0) < groups[g]['target']]
        # All work is claimed: yield this GPU to another checkpoint. No new
        # task process is started for a completed task within this checkpoint.
        if not available:
            print('PERSISTENT_CHECKPOINT_YIELD_ALL_LEASED', checkpoint, flush=True)
            return
        group = min(available, key=lambda g: (status[g].get('leased', 0), list(groups).index(g)))
        spec = groups[group]
        task = spec['task']
        for key in ['RMBENCH_ACTION_EXECUTOR', 'RMBENCH_DENSE_SUBSTEPS',
                    'RMBENCH_DENSE_STATE_SOURCE', 'RMBENCH_EVAL_VIDEO_CAMERA',
                    'RMBENCH_BATTERY_CONTACT_OFFSET', 'RMBENCH_BATTERY_TABLE_FRICTION']:
            os.environ.pop(key, None)
        os.environ.update(RMBENCH_ACTION_EXECUTOR='legacy', RMBENCH_TASK=task,
                          RMBENCH_QUEUE_GROUP=group, RMBENCH_QUEUE_YIELD_ON_WAIT='1')
        os.environ.update(spec.get('env', {}))
        for key in spec.get('unset_env', []):
            os.environ.pop(key, None)
        if task in ('observe_and_pickup', 'press_button'):
            os.environ['RMBENCH_EVAL_VIDEO_CAMERA'] = {'observe_and_pickup': 'world_left', 'press_button': 'table_top'}[task]
        args = dict(base)
        visits += 1
        destination = output / group / ('visit_%04d' % visits)
        destination.mkdir(parents=True, exist_ok=True)
        args.update(task_name=task, num_inference_steps=spec['denoise_steps'], seed=spec['seed'],
                    eval_num_episodes=spec['target'], output_root=str(destination), eval_output_dir=str(destination))
        (destination / 'eval_arguments.json').write_text(json.dumps(args, indent=2))
        if model is not None:
            model.reset()
            model._rmbench_action_executor = type(model._rmbench_action_executor).from_env()
            model.episode_count = 0
            model.num_inference_steps = int(spec['denoise_steps'])
            model.seed = int(spec['seed'])
        print('PERSISTENT_TASK_START', group, 'reuse_model', model is not None, flush=True)
        model = evaluate(args, model=model)
        gc.collect()
        print('PERSISTENT_TASK_END', group, flush=True)


if __name__ == '__main__':
    from test_render import Sapien_TEST
    Sapien_TEST()
    run(harness.parse_args_and_config(), json.loads(Path(os.environ['RMBENCH_PERSISTENT_CONFIG']).read_text()))

