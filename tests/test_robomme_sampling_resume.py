"""Full-state metadata must prevent silently changing RoboMME sample identity."""
from copy import deepcopy
import json

import pytest

from fastwam.datasets.robomme_sampling_resume import validate_sampling_resume


@pytest.fixture
def contract():
    return dict(schema_version=1, sample_stride=1, dataset_len=475257, seed=42,
                batch_size=16, world_size=32, gradient_accumulation_steps=1,
                continuous_dataloader=True, sampler_version='task_balanced_v1',
                probability={'task': 'uniform', 'episode': 'uniform', 'frame': 'uniform'},
                cache={'schema': '16_offset_causal_v1', 'manifest_sha256': 'abc'},
                data_fingerprint='dataset-identity')


def write_state(tmp_path, sampling=None):
    run = tmp_path / 'run'
    state = run / 'checkpoints' / 'state' / 'step_015000'
    state.mkdir(parents=True)
    payload = dict(global_step=15000, epoch=16, batch_in_epoch=136)
    if sampling is not None:
        payload['robomme_sampling'] = sampling
    (state / 'trainer_state.json').write_text(json.dumps(payload), encoding='utf-8')
    return run, state


def test_identical_state_snapshot_and_completed_epoch_cursor(tmp_path, contract):
    _, state = write_state(tmp_path, contract)
    assert validate_sampling_resume(state, contract) == state / 'trainer_state.json'
    payload = json.loads((state / 'trainer_state.json').read_text())
    payload['batch_in_epoch'] = 929
    (state / 'trainer_state.json').write_text(json.dumps(payload))
    assert validate_sampling_resume(state, contract)


@pytest.mark.parametrize('key,new_value', [
    ('dataset_len', 30470), ('seed', 43), ('batch_size', 32), ('world_size', 16),
    ('gradient_accumulation_steps', 2), ('continuous_dataloader', False),
    ('sampler_version', 'vectorized_v2'), ('probability', {'task': 'proportional'}),
    ('cache', {'schema': 'old_cache', 'manifest_sha256': 'def'}),
    ('data_fingerprint', 'other-data'),
])
def test_any_sampling_or_resume_geometry_change_is_rejected(tmp_path, contract, key, new_value):
    saved = deepcopy(contract)
    saved[key] = new_value
    _, state = write_state(tmp_path, saved)
    with pytest.raises(ValueError, match=key):
        validate_sampling_resume(state, contract)


def test_legacy_state_cannot_silently_resume_dense(tmp_path, contract):
    run, state = write_state(tmp_path)
    with pytest.raises(ValueError, match='legacy stride-16'):
        validate_sampling_resume(state, contract)
    (run / 'robomme_contract.json').write_text(json.dumps({'decisions': 30470}))
    with pytest.raises(ValueError, match='contract must be an object'):
        validate_sampling_resume(state, contract)


def test_parent_dense_contract_fallback(tmp_path, contract):
    run, state = write_state(tmp_path)
    report = run / 'robomme_contract.json'
    report.write_text(json.dumps({'sampling': contract}))
    assert validate_sampling_resume(state, contract) == report


def test_state_snapshot_wins_over_changed_parent_report(tmp_path, contract):
    saved = deepcopy(contract)
    saved['seed'] = 7
    run, state = write_state(tmp_path, saved)
    (run / 'robomme_contract.json').write_text(json.dumps({'sampling': contract}))
    with pytest.raises(ValueError, match='seed'):
        validate_sampling_resume(state, contract)


def test_malformed_snapshot_does_not_fall_back(tmp_path, contract):
    run, state = write_state(tmp_path, {})
    (run / 'robomme_contract.json').write_text(json.dumps({'sampling': contract}))
    with pytest.raises(ValueError, match='incomplete'):
        validate_sampling_resume(state, contract)


def test_legacy_and_fresh_and_portable_paths_keep_existing_behavior(tmp_path, contract):
    assert validate_sampling_resume(tmp_path / 'missing', None) is None
    assert validate_sampling_resume(None, contract) is None
    run, state = write_state(tmp_path)
    (run / 'robomme_contract.json').write_text(json.dumps({'decisions': 30470}))
    assert validate_sampling_resume(state, None) is None
    weights = tmp_path / 'step_015000.pt'
    weights.write_bytes(b'not loaded by this metadata guard')
    assert validate_sampling_resume(weights, contract) is None


@pytest.mark.parametrize('key,value', [('epoch', -1), ('batch_in_epoch', 930),
                                      ('global_step', None)])
def test_bad_consumed_cursor_is_rejected(tmp_path, contract, key, value):
    _, state = write_state(tmp_path, contract)
    path = state / 'trainer_state.json'
    payload = json.loads(path.read_text())
    payload[key] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_sampling_resume(state, contract)


def test_missing_cursor_cannot_be_replaced_by_run_contract(tmp_path, contract):
    run, state = write_state(tmp_path)
    (state / 'trainer_state.json').unlink()
    (run / 'robomme_contract.json').write_text(json.dumps({'sampling': contract}))
    with pytest.raises(ValueError, match='consumed cursor'):
        validate_sampling_resume(state, contract)


def test_old_stride_or_incomplete_current_contract_is_rejected(tmp_path, contract):
    old = dict(contract, sample_stride=16)
    _, state = write_state(tmp_path, old)
    with pytest.raises(ValueError, match='stride-1'):
        validate_sampling_resume(state, contract)
    del contract['cache']
    with pytest.raises(ValueError, match='cache'):
        validate_sampling_resume(None, contract)


def test_changed_global_geometry_is_rejected_even_if_effective_batch_matches(tmp_path, contract):
    saved = dict(contract, batch_size=32, world_size=16)
    _, state = write_state(tmp_path, saved)
    with pytest.raises(ValueError, match='batch_size.*world_size'):
        validate_sampling_resume(state, contract)


def test_dense_state_snapshot_cannot_resume_legacy_recipe(tmp_path, contract):
    _, state = write_state(tmp_path, contract)
    with pytest.raises(ValueError, match='dense.*legacy stride-16'):
        validate_sampling_resume(state, None)


def test_dense_parent_contract_cannot_resume_legacy_recipe(tmp_path, contract):
    run, state = write_state(tmp_path)
    (run / 'robomme_contract.json').write_text(json.dumps({'sampling': contract}))
    with pytest.raises(ValueError, match='dense.*legacy stride-16'):
        validate_sampling_resume(state, None)
