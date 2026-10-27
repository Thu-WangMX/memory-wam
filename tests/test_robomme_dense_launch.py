"""Dense-start launch/config and trainer-state integration without GPU allocation."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from fastwam.memory.robomme_clock import HORIZON, inference_contract
from fastwam.trainer import Wan22Trainer
from test_robomme_flow_launch import sandbox


REPO = Path(__file__).resolve().parents[1]
VARIANTS = [
    ('past32', 'robomme_helios_past32_flow'),
    ('pastfuture32', 'robomme_helios_pastfuture32_flow'),
    ('pastfuture32_light', 'robomme_helios_pastfuture32_light_flow'),
]


def _install_wrapper(sandbox):
    for name in ['train_horizon32.sh', 'train_dense_horizon32.sh']:
        shutil.copy2(REPO / 'experiments/robomme' / name,
                     sandbox.root / 'experiments/robomme' / name)
    return sandbox.root / 'experiments/robomme/train_dense_horizon32.sh'


def _run(sandbox, wrapper, variant, **extra):
    env = dict(sandbox.env, MLP_ROLE_INDEX='0', MLP_TASK_ID='dense_test', HOSTNAME='pod-0',
               ROBOMME_LAUNCH_DRY_RUN='1')
    env.update(extra)
    process = subprocess.Popen(['bash', str(wrapper), variant], env=env, cwd=sandbox.root,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return sandbox.finish(process)


@pytest.mark.parametrize('variant,task', VARIANTS)
def test_dense_wrapper_all_workers_print_explicit_sampling_and_cache_without_launch(sandbox, variant, task):
    wrapper = _install_wrapper(sandbox)
    cache = str(sandbox.root / 'dense cache with spaces')
    names = []
    for rank in range(4):
        code, output = _run(sandbox, wrapper, variant, MLP_ROLE_INDEX=str(rank),
                            HOSTNAME=f'pod-{rank}', ROBOMME_DENSE_LATENTS=cache)
        assert code == 0, output
        args = shlex.split(output)
        assert 'task=' + task in args
        assert 'data.train.sample_stride=1' in args
        assert 'data.train.dense_latents_path=' + cache in args
        assert 'batch_size=16' in args and 'gradient_accumulation_steps=1' in args
        assert 'max_steps=20000' in args and 'save_every=2500' in args
        assert 'model.helios_config.history_sizes=[96,2,1]' in args
        assert args[args.index('--num_processes') + 1] == '32'
        assert args[args.index('--machine_rank') + 1] == str(rank)
        names.append(next(arg for arg in args if arg.startswith('wandb.name=')))
    assert len(set(names)) == 1
    assert f'dense1_{variant}_dense_test' in names[0]
    assert not sandbox.events()
    assert not list((sandbox.root / 'runs').iterdir())


def test_dense_wrapper_can_explicitly_fall_back_to_legacy_stride(sandbox):
    wrapper = _install_wrapper(sandbox)
    code, output = _run(sandbox, wrapper, 'past32', ROBOMME_SAMPLE_STRIDE='16')
    assert code == 0, output
    assert 'data.train.sample_stride=16' in shlex.split(output)
    assert not sandbox.events()


def test_dense_wrapper_fails_before_launch_if_cache_missing_or_stride_invalid(sandbox):
    wrapper = _install_wrapper(sandbox)
    code, output = _run(sandbox, wrapper, 'past32', ROBOMME_LAUNCH_DRY_RUN='0',
                        ROBOMME_DENSE_LATENTS=str(sandbox.root / 'missing-cache'))
    assert code != 0 and 'not finalized' in output
    for stride in ['0', '2', '8', '32', 'true']:
        code, output = _run(sandbox, wrapper, 'past32', ROBOMME_SAMPLE_STRIDE=stride)
        assert code != 0 and 'ROBOMME_SAMPLE_STRIDE must be 1 or 16' in output
    assert not sandbox.events()


def _compose(task):
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
        return compose(config_name='train', overrides=['task=' + task])


@pytest.mark.parametrize('task', ['robomme_helios', 'robomme_helios_past_action_sparse16x8_flow',
                                 *[task for _, task in VARIANTS]])
def test_old_tasks_default_to_stride16_and_environment_one_is_an_integer(task, monkeypatch):
    monkeypatch.delenv('ROBOMME_SAMPLE_STRIDE', raising=False)
    monkeypatch.delenv('ROBOMME_DENSE_LATENTS', raising=False)
    cfg = _compose(task)
    assert type(cfg.data.train.sample_stride) is int and cfg.data.train.sample_stride == 16
    assert cfg.data.train.dense_latents_path is None
    monkeypatch.setenv('ROBOMME_SAMPLE_STRIDE', '1')
    monkeypatch.setenv('ROBOMME_DENSE_LATENTS', '/tmp/isolated-dense-cache')
    cfg = _compose(task)
    assert type(cfg.data.train.sample_stride) is int and cfg.data.train.sample_stride == 1
    assert cfg.data.train.dense_latents_path == '/tmp/isolated-dense-cache'


@pytest.mark.parametrize('variant,task', VARIANTS)
def test_dense_sampling_does_not_change_predict32_execute16_contract(variant, task, monkeypatch):
    monkeypatch.setenv('ROBOMME_SAMPLE_STRIDE', '1')
    cfg = _compose(task)
    assert cfg.data.train.sample_stride == 1
    assert cfg.model.helios_config.prediction_horizon == 32
    assert cfg.data.train.prediction_horizon == 32
    assert HORIZON == 16
    assert inference_contract()['replan_stride'] == 16


def _sampling_contract():
    return dict(schema_version=1, sample_stride=1, dataset_len=475257, seed=42,
                batch_size=16, world_size=32, gradient_accumulation_steps=1,
                continuous_dataloader=True, sampler_version='task_episode_start_uniform_v1',
                probability={'task': 'uniform', 'episode': 'uniform', 'frame': 'uniform'},
                cache={'schema': 'robomme_phase_offset16_causal_vae_v1',
                       'manifest_sha256': 'a' * 64})


def _bare_trainer(contract=None):
    trainer = object.__new__(Wan22Trainer)
    cfg = {'robomme_provenance': {'sampling': contract}} if contract is not None else {}
    trainer.cfg = OmegaConf.create(cfg)
    trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 2500, 2, 642
    trainer.batch_size = 16
    return trainer


def test_real_trainer_save_contains_resolved_sampling_snapshot_and_legacy_stays_unchanged(tmp_path):
    contract = _sampling_contract()
    trainer = _bare_trainer(contract)
    trainer.cfg.seed = 42
    trainer.cfg.robomme_provenance.sampling.seed = '${seed}'
    state = tmp_path / 'dense-state'
    state.mkdir()
    trainer._save_trainer_state(str(state))
    payload = json.loads((state / 'trainer_state.json').read_text())
    assert payload == dict(global_step=2500, epoch=2, batch_in_epoch=642, robomme_sampling=contract)
    trainer.cfg.seed = 99
    assert json.loads((state / 'trainer_state.json').read_text())['robomme_sampling']['seed'] == 42
    legacy = tmp_path / 'legacy-state'
    legacy.mkdir()
    _bare_trainer()._save_trainer_state(str(legacy))
    assert json.loads((legacy / 'trainer_state.json').read_text()) == dict(
        global_step=2500, epoch=2, batch_in_epoch=642)


def _record_restore_events(trainer):
    events = []
    trainer.accelerator = SimpleNamespace(
        num_processes=32,
        load_state=lambda **kwargs: events.append(('accelerate', kwargs)),
        wait_for_everyone=lambda: events.append(('barrier', None)))
    trainer.train_sampler = SimpleNamespace(
        set_epoch_offset=lambda value: events.append(('epoch', value)),
        set_resume_batch_offset=lambda value: events.append(('batch', value)),
        clear_resume_batch_offset=lambda: events.append(('clear', None)))
    return events


@pytest.mark.parametrize('corruption', ['legacy_snapshot', 'changed_cache', 'changed_length', 'bad_cursor'])
def test_real_trainer_resume_guard_runs_before_accelerate_or_cursor_mutation(tmp_path, corruption):
    contract = _sampling_contract()
    trainer = _bare_trainer(contract)
    trainer._save_trainer_state(str(tmp_path))
    file = tmp_path / 'trainer_state.json'
    payload = json.loads(file.read_text())
    if corruption == 'legacy_snapshot':
        del payload['robomme_sampling']
    elif corruption == 'changed_cache':
        payload['robomme_sampling']['cache']['manifest_sha256'] = 'b' * 64
    elif corruption == 'changed_length':
        payload['robomme_sampling']['dataset_len'] = 30470
    else:
        payload['batch_in_epoch'] = 930
    file.write_text(json.dumps(payload))
    events = _record_restore_events(trainer)
    trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 7, 0, 0
    with pytest.raises(ValueError):
        trainer.load_training_state(str(tmp_path))
    assert not events
    assert (trainer.global_step, trainer.epoch, trainer.batch_in_epoch) == (7, 0, 0)


def test_real_trainer_resumes_matching_sampling_state_and_consumed_cursor(tmp_path):
    trainer = _bare_trainer(_sampling_contract())
    trainer._save_trainer_state(str(tmp_path))
    trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 0, 0, 0
    events = _record_restore_events(trainer)
    trainer.load_training_state(str(tmp_path))
    assert (trainer.global_step, trainer.epoch, trainer.batch_in_epoch) == (2500, 2, 642)
    assert events == [('accelerate', {'input_dir': str(tmp_path)}), ('epoch', 2),
                      ('batch', 642), ('barrier', None)]
