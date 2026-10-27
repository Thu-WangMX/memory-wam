"""Validate dense RoboMME sampling before Accelerate restores a full state.

Call ``validate_sampling_resume(cfg.resume_checkpoint, report.get('sampling'))``
before writing the new run's contract. Legacy stride-16 recipes should pass
``None``. A portable weights file is checked by the model's checkpoint loader.

The complete JSON sampling contract belongs in ``report['sampling']`` and in
``trainer_state.json['robomme_sampling']`` at checkpoint time. Required keys:
``schema_version=1, sample_stride=1, dataset_len, seed, batch_size, world_size,
gradient_accumulation_steps, continuous_dataloader, sampler_version,
probability, cache``. ``probability`` describes the sampling distribution;
``cache`` must identify the offset-cache schema and its content/provenance hash.
Additional fields (for example a model/data fingerprint) are compared too.

This module deliberately has no torch/Accelerate imports: rejection must happen
before optimizer, scheduler, RNG, or data-cursor state is mutated.
"""

from collections.abc import Mapping
import json
from math import ceil
from pathlib import Path


STATE_SAMPLING_KEY = 'robomme_sampling'
_REQUIRED = frozenset({
    'schema_version', 'sample_stride', 'dataset_len', 'seed', 'batch_size',
    'world_size', 'gradient_accumulation_steps', 'continuous_dataloader',
    'sampler_version', 'probability', 'cache',
})


def _contract(value, source):
    if not isinstance(value, Mapping):
        raise ValueError(f'{source}: dense RoboMME sampling contract must be an object')
    missing = _REQUIRED - value.keys()
    if missing:
        raise ValueError(f'{source}: incomplete dense RoboMME sampling contract: {sorted(missing)}')
    try:
        result = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{source}: sampling contract must contain finite JSON values') from exc
    for key in ('schema_version', 'sample_stride', 'dataset_len', 'seed',
                'batch_size', 'world_size', 'gradient_accumulation_steps'):
        if type(result[key]) is not int or result[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'{source}: invalid sampling contract field {key}')
    if result['schema_version'] != 1 or result['sample_stride'] != 1:
        raise ValueError(f'{source}: checkpoint is not the supported dense stride-1 sampling schema')
    if type(result['continuous_dataloader']) is not bool:
        raise ValueError(f'{source}: continuous_dataloader must be boolean')
    if not isinstance(result['sampler_version'], str) or not result['sampler_version']:
        raise ValueError(f'{source}: sampler_version must be a nonempty string')
    for key in ('probability', 'cache'):
        if not isinstance(result[key], (str, dict)) or not result[key]:
            raise ValueError(f'{source}: {key} must identify the sampling/cache contract')
    return result


def _read_object(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read RoboMME resume metadata: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'RoboMME resume metadata must be an object: {path}')
    return value


def _reject_dense_state_for_legacy(resume_path):
    if not resume_path:
        return
    state_dir = Path(resume_path).expanduser()
    if not state_dir.is_dir():
        return
    candidates = [(state_dir / 'trainer_state.json', STATE_SAMPLING_KEY)]
    candidates.extend((parent / 'robomme_contract.json', 'sampling')
                      for parent in list(state_dir.parents)[:3])
    for path, key in candidates:
        if not path.is_file():
            continue
        try:
            metadata = _read_object(path)
        except ValueError:
            # Preserve legacy handling of unrelated/old metadata. Its normal
            # state loader still diagnoses an unreadable trainer_state.json.
            continue
        if metadata.get(key):
            raise ValueError(
                'Cannot exactly resume a dense RoboMME sampling state into a '
                f'legacy stride-16 recipe: {path}')


def validate_sampling_resume(resume_path, current_contract):
    """Return the validated metadata path, or None for fresh/legacy/weights use.

    Full-state resume requires an exact sampling contract and consumed cursor.
    Prefer the state-local immutable snapshot. For a state predating that field,
    accept the existing parent run's ``robomme_contract.json['sampling']`` only
    when it contains the complete matching dense contract. Missing metadata is
    an error, never permission to reinterpret a stride-16 training state.
    """
    if current_contract is None:
        _reject_dense_state_for_legacy(resume_path)
        return None
    current = _contract(current_contract, 'Current configuration')
    if not resume_path:
        return None
    state_dir = Path(resume_path).expanduser()
    if state_dir.is_file():
        return None  # Portable checkpoints have their own model/data guard.
    if not state_dir.is_dir():
        raise ValueError(f'RoboMME resume state directory does not exist: {state_dir}')

    state_path = state_dir / 'trainer_state.json'
    if not state_path.is_file():
        raise ValueError(f'Dense RoboMME exact resume requires the consumed cursor: {state_path}')
    state = _read_object(state_path)
    for key in ('global_step', 'epoch', 'batch_in_epoch'):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError(f'Dense RoboMME exact resume requires a valid {key}: {state_path}')

    source = state_path
    if STATE_SAMPLING_KEY in state:
        saved = _contract(state[STATE_SAMPLING_KEY], str(source))
    else:
        # Standard layout is run/checkpoints/state/step_N. Also allow a run to
        # store states one or two levels closer; never search arbitrary ancestors.
        saved = None
        for parent in list(state_dir.parents)[:3]:
            candidate = parent / 'robomme_contract.json'
            if candidate.is_file():
                source = candidate
                saved = _contract(_read_object(candidate).get('sampling'), str(candidate))
                break
        if saved is None:
            raise ValueError(
                'Dense RoboMME exact resume requires sampling metadata; this may be '
                'a legacy stride-16 state. Start a fresh run or use an explicit '
                'weight-transfer workflow instead of restoring optimizer/data state.')

    if saved != current:
        changed = sorted(key for key in saved.keys() | current.keys()
                         if saved.get(key) != current.get(key))
        raise ValueError(f'Dense RoboMME resume sampling contract differs in {changed}: {source}')
    batch_count = ceil(current['dataset_len'] / (current['batch_size'] * current['world_size']))
    if state['batch_in_epoch'] > batch_count:
        raise ValueError(f'Dense RoboMME resume batch cursor exceeds the epoch length: {state_path}')
    if (current['continuous_dataloader']
            and batch_count % current['gradient_accumulation_steps']):
        raise ValueError('Dense RoboMME continuous epoch length must be divisible by gradient accumulation')
    return source
