"""Exercise the real 4-worker shell wrapper with fake training/upload executables."""
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest
import test_robomme_launch as launch_support

REPO = Path(__file__).resolve().parents[1]
SCRIPT = 'experiments/robomme/train_past_action_flow_32gpu.sh'


@pytest.fixture
def sandbox():
    helper = launch_support.RoboMMELaunchTest()
    helper.setUp()
    shutil.copy2(REPO / SCRIPT, helper.root / SCRIPT)
    wrapper = 'experiments/robomme/train_past_action_sparse16x8_flow_32gpu.sh'
    shutil.copy2(REPO / wrapper, helper.root / wrapper)
    (helper.root / 'runtime').mkdir()
    (helper.root / 'runtime/memory_ablation_train.env').write_text(
        'export BATCH_SIZE=8 GRAD_ACCUM=2 MAX_STEPS=17 SAVE_EVERY=5000\n'
        'export NNODES=1 NUM_GPUS=8 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=127.0.0.1\n'
        'export RUN_ID=stale_legacy_run MASTER_PORT=12345\n')
    helper.env.update(NNODES='4', NUM_GPUS='32', MLP_WORKER_NUM='4')
    # Default run id must be shared even when workers start at different times.
    helper.env.pop('RUN_ID')
    try:
        yield helper
    finally:
        helper.doCleanups()


def worker(helper, rank, **overrides):
    env = dict(helper.env, MLP_ROLE_INDEX=str(rank), HOSTNAME=f'volc-flow-test-worker-{rank}')
    env.update(overrides)
    return subprocess.Popen(['bash', str(helper.root / SCRIPT)], env=env,
                            cwd=helper.root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_four_workers_share_recipe_run_and_single_uploader(sandbox):
    processes = [worker(sandbox, rank) for rank in [3, 2, 1, 0]]
    for process in reversed(processes):
        code, output = sandbox.finish(process)
        assert code == 0, output
    events = sandbox.events()
    launches = [e for e in events if e['program'] == 'accelerate']
    assert sorted(e['node'] for e in launches) == ['0', '1', '2', '3']
    for e in launches:
        args = e['argv']
        for key, value in [('--num_processes', '32'), ('--num_machines', '4'),
                           ('--machine_rank', e['node']),
                           ('--main_process_ip', 'worker0.example'),
                           ('--main_process_port', '29639'),
                           ('--deepspeed_multinode_launcher', 'standard')]:
            assert args[args.index(key)+1] == value
        for value in ('task=robomme_helios_past_action_flow', 'batch_size=16',
                      'gradient_accumulation_steps=1', 'max_steps=20000', 'save_every=2500',
                      'persistent_workers=true', 'continuous_dataloader=true',
                      'model.helios_config.history_sizes=[96,2,1]',
                      'model.helios_config.anchor_frames=2',
                      'wandb.name=robomme_all16_long99_pastflow16_32gpu_b16ga1_s20000_volc-flow-test'):
            assert value in args
        assert e['visible'] == sandbox.env['CUDA_VISIBLE_DEVICES']
    assert [e['node'] for e in events if '--run' in e['argv']] == ['0']
    assert [e['node'] for e in events if '--doctor' in e['argv']] == ['0']
    runs = list((sandbox.root / 'runs').iterdir())
    assert len(runs) == 1
    for rank in range(4):
        assert (runs[0] / f'train.worker-{rank}.log').is_file()


def test_dry_run_preserves_explicit_recipe_and_does_not_launch(sandbox):
    for rank in range(4):
        code, output = sandbox.finish(worker(
            sandbox, rank, ROBOMME_LAUNCH_DRY_RUN='1', RUN_ID='explicit_fm_run',
            BATCH_SIZE='8', GRAD_ACCUM='2', SAVE_EVERY='1000'))
        assert code == 0, output
        args = shlex.split(output)
        assert args[args.index('--machine_rank')+1] == str(rank)
        assert 'batch_size=8' in args and 'gradient_accumulation_steps=2' in args
        assert 'save_every=1000' in args and 'wandb.name=explicit_fm_run' in args
    assert not sandbox.events()
    assert not list((sandbox.root / 'runs').iterdir())


def test_platform_task_id_works_without_worker_hostname(sandbox):
    sandbox.env.pop('MASTER_PORT')
    for rank in range(4):
        code, output = sandbox.finish(worker(
            sandbox, rank, ROBOMME_LAUNCH_DRY_RUN='1', MLP_TASK_ID='task_20261020_123',
            MLP_WORKER_0_PORT='29677', HOSTNAME=f'pod-{rank}'))
        assert code == 0, output
        args = shlex.split(output)
        assert args[args.index('--machine_rank')+1] == str(rank)
        assert args[args.index('--main_process_port')+1] == '29677'
        assert ('wandb.name=robomme_all16_long99_pastflow16_32gpu_b16ga1_s20000_task_20261020_123'
                in args)
    assert not sandbox.events()


def test_wrong_allocation_or_stale_rank_rejected(sandbox):
    for extra in (dict(MLP_WORKER_NUM='2'), dict(NODE_RANK='0'),
                  dict(MASTER_ADDR='127.0.0.1')):
        code, output = sandbox.finish(worker(sandbox, 3, ROBOMME_LAUNCH_DRY_RUN='1', **extra))
        assert code != 0, output
    assert not sandbox.events()


def test_training_failure_skips_upload(sandbox):
    code, output = sandbox.finish(worker(sandbox, 0, FAKE_FAIL_RANK='0'))
    assert code == 17, output
    assert not any('--run' in e['argv'] for e in sandbox.events())


def test_sparse_wrapper_all_workers_same_variant_without_launch(sandbox):
    wrapper = sandbox.root / 'experiments/robomme/train_past_action_sparse16x8_flow_32gpu.sh'
    for rank in range(4):
        env = dict(sandbox.env, MLP_ROLE_INDEX=str(rank), MLP_TASK_ID='sparse_test',
                   ROBOMME_LAUNCH_DRY_RUN='1', HOSTNAME=f'pod-{rank}')
        process = subprocess.Popen(['bash', str(wrapper)], env=env, cwd=sandbox.root,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        code, output = sandbox.finish(process)
        assert code == 0, output
        args = shlex.split(output)
        assert 'task=robomme_helios_past_action_sparse16x8_flow' in args
        assert 'wandb.name=robomme_all16_long99_pastflow_sparse16x8_32gpu_b16ga1_s20000_sparse_test' in args
        assert args[args.index('--machine_rank')+1] == str(rank)
        assert 'model.helios_config.history_sizes=[96,2,1]' in args
    assert not sandbox.events()
