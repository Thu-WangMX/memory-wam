"""The combined recipe resolves identically on every platform worker."""
from pathlib import Path
import shlex
import shutil
import subprocess

from hydra import compose, initialize_config_dir
import pytest
from test_robomme_flow_launch import sandbox

REPO = Path(__file__).resolve().parents[1]


def test_push75_resolved_contract(monkeypatch):
    with initialize_config_dir(config_dir=str(REPO/'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=['task=robomme_helios_push75_flow'])
    h = cfg.model.helios_config
    assert list(h.history_sizes) == [72, 2, 1]
    assert h.anchor_mode == 'demo_execution_first'
    assert h.history_padding == 'compact'
    assert h.view_layout == cfg.data.train.view_layout == 'two_view_horizontal'
    assert h.past_action.past_query_slots == cfg.data.train.past_query_slots == 32
    assert h.past_action.source_history_slots == 75
    assert h.prediction_horizon == cfg.data.train.prediction_horizon == 32
    assert cfg.data.train.sample_stride == 1
    assert h.past_action.loss_weight == h.past_action.future_loss_weight == .1


def _install(sandbox):
    wrapper = sandbox.root/'experiments/robomme/train_push75.sh'
    shutil.copyfile(REPO/'experiments/robomme/train_push75.sh', wrapper)
    return wrapper


def test_push75_all_workers_share_run_and_camera_cache(sandbox):
    wrapper = _install(sandbox)
    names = []
    for rank in range(4):
        env = dict(sandbox.env, MLP_ROLE_INDEX=str(rank), MLP_TASK_ID='push75_test',
                   HOSTNAME=f'arbitrary-pod-{rank}', ROBOMME_LAUNCH_DRY_RUN='1')
        proc = subprocess.Popen(['bash',str(wrapper)],env=env,cwd=sandbox.root,
                                text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        code,output = sandbox.finish(proc)
        assert code == 0, output
        args = shlex.split(output)
        assert 'task=robomme_helios_push75_flow' in args
        assert 'model.helios_config.history_sizes=[72,2,1]' in args
        assert 'model.helios_config.patch_long=[4,8,8]' in args
        assert 'batch_size=16' in args and 'gradient_accumulation_steps=1' in args
        assert 'max_steps=20000' in args and 'save_every=2500' in args
        assert 'data.train.sample_stride=1' in args
        assert args[args.index('--num_processes')+1] == '32'
        assert args[args.index('--machine_rank')+1] == str(rank)
        for key,suffix in [('full_kv_cache_path','base'),('prepared_root','prepared'),('dense_latents_path','dense')]:
            assert any(x.startswith('data.train.'+key+'=') and x.endswith('/two-view-square256-v1/'+suffix) for x in args)
        names.append(next(x for x in args if x.startswith('wandb.name=')))
    assert len(set(names)) == 1 and 'push75_test' in names[0]
    assert not sandbox.events()


def test_push75_missing_cache_stops_before_launch(sandbox):
    wrapper = _install(sandbox)
    env = dict(sandbox.env, ROBOMME_PUSH75_CACHE_ROOT=str(sandbox.root/'not-built'),
               MLP_TASK_ID='push75_test', ROBOMME_LAUNCH_DRY_RUN='0')
    proc = subprocess.Popen(['bash',str(wrapper)],env=env,cwd=sandbox.root,
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    code,output = sandbox.finish(proc)
    assert code != 0 and 'Missing finalized camera cache' in output
    assert not sandbox.events()
