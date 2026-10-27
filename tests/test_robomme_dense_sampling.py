"""Dense current-frame sampling keeps a causal, phase-local observed clock."""
from collections import Counter
from itertools import islice
from pathlib import Path
import json

import pytest
import torch

from fastwam.datasets import robomme as dataset_module
from fastwam.datasets.robomme import RoboMMEHeliosDataset
from fastwam.datasets.robomme_past_action import RoboMMEPastActionSparse16x8FlowDataset
from fastwam.datasets.robomme_past_future import RoboMMEPastFutureActionDataset
from fastwam.datasets.robomme_dense_cache import DenseRoboMMECache
from fastwam.datasets.robodojo import sha256
from fastwam.memory.past_action import past_action_sparse_layout
from fastwam.memory.past_future_action import past_future_layout, training_targets
from fastwam.memory.robomme_clock import (
    CACHE_SCHEMA, PHASE_ANCHOR_MODE, decision_frames, phase_sample_indices, select_robomme_anchors,
)
from fastwam.memory.robomme_dense_clock import (
    DENSE_SCHEMA, dense_decision_frames, dense_execution_frames, dense_prefix_frames,
    execution_offset, validate_dense_prefix,
)
from fastwam.utils.samplers import ContinuousEpochSampler


@pytest.mark.parametrize('execution_start', [0, 1, 16, 17, 35, 49])
@pytest.mark.parametrize('remaining', [1, 2, 15, 16, 17, 33, 129])
def test_offset_zero_is_exactly_the_legacy_clock(execution_start, remaining):
    length = execution_start + remaining
    assert dense_decision_frames(length, execution_start, 0) == decision_frames(length, execution_start)
    for current in range(execution_start, length, 16):
        assert dense_prefix_frames(current, execution_start) == decision_frames(current + 1, execution_start)
        assert validate_dense_prefix(dense_prefix_frames(current, execution_start), execution_start) == 0


@pytest.mark.parametrize('execution_start', [0, 1, 17, 35])
def test_all_current_frames_are_covered_once_by_the_sixteen_streams(execution_start):
    length = execution_start + 67
    ownership = Counter()
    for offset in range(16):
        execution = dense_execution_frames(length, execution_start, offset)
        assert execution[0] == execution_start
        expected = ([execution_start] + list(range(execution_start + offset, length, 16))
                    if offset else list(range(execution_start, length, 16)))
        assert execution == expected
        for current in execution:
            if current == execution_start and offset:
                continue  # The shared execution anchor is not an extra training start.
            assert execution_offset(current, execution_start) == offset
            ownership[current] += 1
    assert ownership == Counter({frame: 1 for frame in range(execution_start, length)})


@pytest.mark.parametrize('execution_start', [0, 1, 17, 35, 1100])
@pytest.mark.parametrize('offset', range(16))
def test_dense_prefix_and_vae_rgb_reads_never_use_future_frames(execution_start, offset):
    current = execution_start + 32 + offset
    frames = dense_prefix_frames(current, execution_start)
    assert frames[-1] == current
    assert all(a < b for a, b in zip(frames, frames[1:]))
    assert validate_dense_prefix(frames, execution_start) == offset
    old_demo = [frame for frame in decision_frames(current + 1, execution_start)
                if frame < execution_start]
    demo = [frame for frame in frames if frame < execution_start]
    execution = [frame for frame in frames if frame >= execution_start]
    assert demo == old_demo
    assert execution[0] == execution_start
    if offset:
        assert execution[:2] == [execution_start, execution_start + offset]
    for phase in (demo, execution):
        rgb = phase_sample_indices(phase)
        if phase:
            assert min(rgb) == phase[0] and max(rgb) == phase[-1] <= current
            assert len(rgb) == 1 + 4 * (len(phase) - 1)
    values = (torch.tensor(frames).float() + 1).reshape(-1, 1, 1, 1, 1)
    anchors, valid = select_robomme_anchors(values, frames, execution_start,
                                           mode=PHASE_ANCHOR_MODE)
    assert valid.tolist() == [execution_start > 0, True]
    assert anchors.flatten().tolist() == [1. if execution_start else 0., execution_start + 1.]


@pytest.mark.parametrize('frames, execution_start', [
    ([], 0), ([1, 17], 0), ([0, 16, 15], 0), ([0, 16, 33], 0),
    ([0, 16, 34, 36, 52], 35), ([0, 16, 34, 35, 38, 55], 35),
])
def test_malformed_dense_prefix_is_rejected(frames, execution_start):
    with pytest.raises(ValueError):
        validate_dense_prefix(frames, execution_start)


@pytest.mark.parametrize('offset', [-1, 16, 32])
def test_invalid_stream_offsets_are_rejected(offset):
    with pytest.raises(ValueError):
        dense_execution_frames(100, 17, offset)


@pytest.mark.parametrize('execution_start', [0, 17, 35])
@pytest.mark.parametrize('offset', range(16))
def test_past_and_future_queries_use_actual_dense_frame_destinations(execution_start, offset):
    current = execution_start + 48 + offset
    frames = dense_prefix_frames(current, execution_start)
    past = past_action_sparse_layout(frames, execution_start)
    valid = past['past_action_transition_valid']
    destinations = past['past_action_transition_frames'][valid]
    assert (destinations > 0).all() and (destinations <= current).all()
    if execution_start:
        assert not (destinations == execution_start).any()
    assert past['past_action_frame_indices'][past['past_action_valid']][-1] == current
    assert past['past_action_relative_time'][past['past_action_valid']][-1] == 0
    for start, end, values, occupied in zip(past['past_action_interval_start'],
                                          past['past_action_frame_indices'],
                                          past['past_action_transition_frames'], valid):
        if occupied.any():
            assert ((values[occupied] > start) & (values[occupied] <= end)).all()
            assert (values[occupied][1:] - values[occupied][:-1] == 2).all()
            assert values[occupied][-1] == end
    both = past_future_layout(frames, execution_start)
    assert torch.equal(both['past_action_transition_frames'][:16], past['past_action_transition_frames'])
    assert both['past_action_transition_frames'][-2:].tolist() == [
        list(range(current + 2, current + 17, 2)), list(range(current + 18, current + 33, 2))]
    assert both['past_action_transition_valid'][-2:].all()


class _IdentityNormalizer:
    def forward(self, value):
        return value


class _FrameLabelReader:
    """Synthetic labels expose original destination indices in every dimension."""
    def __init__(self, length):
        self.offsets = {0: (0, length)}

    def targets(self, episode, layout):
        assert episode == 0
        frames = layout['past_action_transition_frames']
        valid = layout['past_action_transition_valid']
        assert (frames[valid] > 0).all() and (frames[valid] < self.offsets[0][1]).all()
        return frames.float().masked_fill(~valid, 0)[..., None].expand(-1, -1, 64).clone()


def _getitem_fixture(dataset_cls, monkeypatch, *, current=38, length=43, execution_start=17):
    """Exercise production __getitem__ without requiring videos or model weights."""
    dataset = object.__new__(dataset_cls)
    row = dict(episode_index=0, length=length, exec_start_idx=execution_start,
               semantic_task='VideoUnmask')
    dataset.samples = [(row, current)]
    dataset.root = Path('/unused-readonly-dataset')
    dataset.prepared_root = Path('/unused-readonly-prepared')
    dataset.info = {}
    dataset.prediction_horizon = 32
    dataset.history_sizes = (96, 2, 1)
    dataset.anchor_frames = 2
    dataset.anchor_mode = PHASE_ANCHOR_MODE
    dataset.sample_stride = 1
    dataset.norm = {key: _IdentityNormalizer() for key in ['action', 'state']}
    dataset.instructions = {0: 'test instruction'}
    dataset._contexts = {'test instruction': (torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))}
    frames = dense_prefix_frames(current, execution_start)
    payload = dict(frame_indices=torch.tensor(frames), execution_start=execution_start,
                   latents=(torch.tensor(frames).float() + 1).reshape(-1, 1, 1, 1, 1)
                   .expand(-1, 48, 1, 2, 2).clone())
    calls = []
    def observed_prefix(request_row, request_current):
        assert request_row is row and request_current == current
        calls.append(request_current)
        return payload
    dataset.observed_prefix = observed_prefix
    def old_cache_is_forbidden(*_):
        raise AssertionError('Dense sample attempted to use the legacy offset-zero clock')
    dataset.load_latents = old_cache_is_forbidden
    rows = torch.arange(length).float()[:, None].expand(-1, 8).clone()
    dataset._table = lambda _: {'action': rows, 'observation.state': rows,
                                 'task_index': torch.zeros(length, dtype=torch.long)}
    video_reads = []
    def read_video(_root, _info, _row, indices):
        video_reads.append(list(indices))
        return torch.tensor(indices).float()[None, :, None, None].expand(3, -1, 2, 2).clone()
    monkeypatch.setattr(dataset_module, 'episode_video', read_video)
    dataset.dreamdojo = _FrameLabelReader(length)
    return dataset, row, calls, video_reads


@pytest.mark.parametrize('dataset_cls', [RoboMMEHeliosDataset,
                                        RoboMMEPastActionSparse16x8FlowDataset,
                                        RoboMMEPastFutureActionDataset])
def test_dense_dataset_action_plus_one_tail_masks_and_shared_observed_prefix(dataset_cls, monkeypatch):
    dataset, row, calls, video_reads = _getitem_fixture(dataset_cls, monkeypatch)
    sample = dataset[0]
    assert calls and set(calls) == {38}
    assert sample['frame_index'] == 38
    assert sample['action'].shape == (32, 8)
    assert sample['action'][:4, 0].tolist() == [39., 40., 41., 42.]
    assert sample['action'][4:, 0].tolist() == [42.] * 28
    assert sample['action_is_pad'].tolist() == [False] * 4 + [True] * 28
    assert sample['image_is_pad'].tolist() == [False, False] + [True] * 7
    assert video_reads == [[38, 42] + [42] * 7]
    assert sample['history_latents'][-1, 0, 0, 0, 0] == 39
    assert sample['anchor_latents'][:, 0, 0, 0, 0].tolist() == [1., 18.]
    if 'past_action_target' in sample:
        past_valid = sample['past_action_transition_valid'][:16]
        assert (sample['past_action_target'][:16, :, 0][past_valid] <= 38).all()
        torch.testing.assert_close(sample['past_action_target'][:16, :, 0][past_valid],
                                   sample['past_action_transition_frames'][:16][past_valid].float())
    if dataset_cls is RoboMMEPastFutureActionDataset:
        assert sample['past_action_transition_valid'][-2:].all()
        assert sample['past_action_valid'][-2:].all()
        assert sample['past_action_loss_valid'][-2:].tolist() == [[True, True] + [False] * 6,
                                                                [False] * 8]
        assert sample['past_action_target'][-2:, :, 0].tolist() == [[40., 42.] + [0.] * 6,
                                                                   [0.] * 8]


def test_final_useful_start_has_one_action_label_and_no_future_label_leak(monkeypatch):
    dataset, _, _, _ = _getitem_fixture(RoboMMEPastFutureActionDataset, monkeypatch,
                                       current=42, length=44, execution_start=17)
    sample = dataset[0]
    assert sample['action'][0, 0] == 43
    assert sample['action_is_pad'].tolist() == [False] + [True] * 31
    assert sample['past_action_valid'][-2:].all()  # Availability cannot leak episode end.
    assert not sample['past_action_loss_valid'][-2:].any()
    assert sample['past_action_target'][-2:].count_nonzero() == 0


def _sampler_dataset():
    dataset = object.__new__(RoboMMEHeliosDataset)
    dataset.task_balanced = True
    dataset.samples, dataset.episode_samples = [], {}
    dataset.task_episodes = {'task_a': [0, 1], 'task_b': [2, 3]}
    for episode, count in enumerate([11, 63, 19, 101]):
        start = len(dataset.samples)
        dataset.samples.extend([(episode, frame) for frame in range(count)])
        dataset.episode_samples[episode] = list(range(start, start + count))
    return dataset


def test_dense_sampling_retains_equal_tasks_and_equal_episodes_not_frame_mass():
    dataset = _sampler_dataset()
    episodes = Counter()
    for seed in range(80):
        sampled = dataset.sample_epoch_indices(seed)
        assert sampled == dataset.sample_epoch_indices(seed)
        assert len(sampled) == len(dataset)
        counts = Counter(dataset.samples[i][0] // 2 for i in sampled)
        assert abs(counts[0] - counts[1]) <= 1
        episodes.update(dataset.samples[i][0] for i in sampled)
    # Episode lengths differ greatly, but their total draw probabilities do not.
    mean = sum(episodes.values()) / 4
    assert all(abs(count - mean) / mean < .08 for count in episodes.values())


@pytest.mark.parametrize('epoch, batch_offset', [(0, 0), (0, 7), (1, 0), (2, 13)])
def test_dense_sampling_resume_matches_consumed_stream_across_epoch_prefetch(epoch, batch_offset):
    dataset = _sampler_dataset()
    sampler = ContinuousEpochSampler(dataset, seed=713, batch_size=2, num_processes=2)
    consumed = epoch * len(sampler) + batch_offset * sampler.global_batch_size
    full = iter(sampler)
    list(islice(full, consumed))
    expected = list(islice(full, len(sampler) + 12))
    # Prefetching extra values must not mutate the saved consumed cursor.
    list(islice(full, 47))
    resumed = ContinuousEpochSampler(dataset, seed=713, batch_size=2, num_processes=2)
    resumed.set_epoch_offset(epoch)
    resumed.set_resume_batch_offset(batch_offset)
    assert list(islice(iter(resumed), len(sampler) + 12)) == expected


def test_dense_sampling_resume_at_legacy_end_of_epoch_cursor():
    dataset = _sampler_dataset()
    old = ContinuousEpochSampler(dataset, seed=42, batch_size=2, num_processes=2)
    expected = list(islice(iter(old), len(old), len(old) * 2))
    resumed = ContinuousEpochSampler(dataset, seed=42, batch_size=2, num_processes=2)
    resumed.set_resume_batch_offset(resumed.batches_per_epoch)
    assert list(islice(iter(resumed), len(resumed))) == expected


@pytest.fixture
def constructor_assets(tmp_path):
    root, base, prepared = [tmp_path / name for name in ['data', 'base', 'prepared']]
    (root / 'meta').mkdir(parents=True)
    base.mkdir()
    prepared.mkdir()
    rows = [dict(episode_index=i, semantic_task=dataset_module.TASKS[i // 100],
                 exec_start_idx=[0, 1, 17, 34, 3][i % 5], length=36 + i % 5)
            for i in range(1600)]
    (root / 'meta/info.json').write_text(json.dumps(dict(total_episodes=1600, fps=10, robot_type='panda')))
    (root / 'meta/episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0, task='test instruction')) + '\n')
    metadata = dict(schema=CACHE_SCHEMA, complete=True, episode_count=1600,
                    semantic_tasks=dataset_module.TASKS, vae_phase_reset=True, replan_stride=16,
                    temporal_sampling='phase_local_endpoint_linspace4',
                    encoding='separate_demo_execution_causal_vae', color_contract='simulator_rgb_preserved',
                    latent_shape=[48, 1, 24, 20], latent_dtype='bfloat16',
                    mosaic='wrists_top_head_bottom_384x320', vae_model_sha256='a' * 64)
    manifest = dict(metadata=metadata, episodes={str(i): f'episode_{i:06d}.pt' for i in range(1600)})
    (base / 'manifest.json').write_text(json.dumps(manifest))
    (prepared / 'dataset_stats.json').write_text(json.dumps({
        key: {'default': {'global_mean': [0.] * 8, 'global_std': [1.] * 8}}
        for key in ['action', 'state']}))
    provenance = {name + '_sha256': sha256(path) for name, path in [
        ('info', root / 'meta/info.json'), ('episodes', root / 'meta/episodes.jsonl'),
        ('tasks', root / 'meta/tasks.jsonl'), ('latents', base / 'manifest.json'),
        ('stats', prepared / 'dataset_stats.json')]}
    (prepared / 'provenance.json').write_text(json.dumps(provenance))
    return dict(root=root, full_kv_cache_path=base, prepared_root=prepared), rows, manifest


def test_constructor_indexes_all_execution_starts_and_keeps_legacy_opt_in(constructor_assets, monkeypatch):
    kwargs, rows, _ = constructor_assets
    constructed = []
    class CompleteDenseCache:
        def __init__(self, *args):
            constructed.append(args)
    monkeypatch.setattr('fastwam.datasets.robomme_dense_cache.DenseRoboMMECache', CompleteDenseCache)
    selected = [0, 1, 2, 3, 1599]
    legacy = RoboMMEHeliosDataset(**kwargs, episode_indices=selected)
    dense = RoboMMEHeliosDataset(**kwargs, episode_indices=selected,
                                sample_stride=1, dense_latents_path='/fake-complete-cache')
    expected = [(row['episode_index'], frame) for row in rows if row['episode_index'] in selected
                for frame in range(row['exec_start_idx'], row['length'] - 1)]
    assert [(row['episode_index'], frame) for row, frame in dense.samples] == expected
    assert len(dense) == len(expected)
    assert [(row['episode_index'], frame) for row, frame in legacy.samples] == [
        (row['episode_index'], frame) for row in rows if row['episode_index'] in selected
        for frame in range(row['exec_start_idx'], row['length'] - 1, 16)]
    assert len(constructed) == 1
    for row, frame in dense.samples:
        assert row['exec_start_idx'] <= frame < row['length'] - 1


def test_dense_constructor_requires_cache_and_rejects_invalid_stride(constructor_assets):
    kwargs, _, _ = constructor_assets
    with pytest.raises(ValueError, match='dense_latents_path'):
        RoboMMEHeliosDataset(**kwargs, sample_stride=1)
    for stride in [0, 2, 8, 32, True]:
        with pytest.raises(ValueError, match='sample_stride'):
            RoboMMEHeliosDataset(**kwargs, sample_stride=stride)


@pytest.fixture
def dense_cache_assets(tmp_path, constructor_assets):
    kwargs, rows, base_manifest = constructor_assets
    # The cache reader is independently tested with one selected source episode.
    row = rows[2]
    root = tmp_path / 'dense'
    root.mkdir()
    source_hash = sha256(kwargs['full_kv_cache_path'] / 'manifest.json')
    metadata = dict(source_manifest_sha256=source_hash)
    for name, suffix in [('info', 'json'), ('episodes', 'jsonl'), ('tasks', 'jsonl')]:
        metadata[name + '_sha256'] = sha256(kwargs['root'] / 'meta' / f'{name}.{suffix}')
    for name in ['vae_model_sha256', 'encoding', 'temporal_sampling', 'latent_shape', 'latent_dtype']:
        metadata[name] = base_manifest['metadata'][name]
    offsets = {}
    shape = (48, 1, 24, 20)
    base_frames = decision_frames(row['length'], row['exec_start_idx'])
    base_payload = dict(frame_indices=torch.tensor(base_frames), execution_start=row['exec_start_idx'],
                        latents=(torch.tensor(base_frames).to(torch.bfloat16) + 1)
                        .reshape(-1, 1, 1, 1, 1).expand(-1, *shape).clone())
    base_file = kwargs['full_kv_cache_path'] / base_manifest['episodes'][str(row['episode_index'])]
    torch.save(base_payload, base_file)
    for offset in range(16):
        execution = dense_execution_frames(row['length'], row['exec_start_idx'], offset)
        if offset == 0:
            offsets['0'] = dict(source='base_cache', frame_count=len(execution),
                                path=base_file.name, sha256=sha256(base_file))
            continue
        frames = execution[1:]
        payload = dict(schema=DENSE_SCHEMA, episode_index=row['episode_index'],
                       execution_start=row['exec_start_idx'], offset=offset,
                       source_manifest_sha256=source_hash, frame_indices=torch.tensor(frames),
                       latents=(torch.tensor(frames).to(torch.bfloat16) + 1)
                       .reshape(-1, 1, 1, 1, 1).expand(-1, *shape).clone())
        path = root / f'offset_{offset:02d}.pt'
        torch.save(payload, path)
        offsets[str(offset)] = dict(source='dense_cache', frame_count=len(frames),
                                    path=path.name, sha256=sha256(path))
    manifest = dict(schema=DENSE_SCHEMA, complete=True, metadata=metadata,
                    episodes={str(row['episode_index']): dict(offsets=offsets)})
    (root / 'manifest.json').write_text(json.dumps(manifest))
    arguments = (root, kwargs['root'], kwargs['full_kv_cache_path'], base_manifest, [row])
    return arguments, manifest, row, base_payload


def test_dense_cache_prefix_matches_exact_current_and_reuses_base_anchor(dense_cache_assets):
    arguments, _, row, base_payload = dense_cache_assets
    cache = DenseRoboMMECache(*arguments)
    for current in range(row['exec_start_idx'], row['length'] - 1):
        prefix = cache.prefix(row, current, base_payload)
        frames = dense_prefix_frames(current, row['exec_start_idx'])
        assert prefix['frame_indices'].tolist() == frames
        assert prefix['latents'].shape[0] == len(frames)
        assert prefix['latents'][:, 0, 0, 0, 0].float().tolist() == [frame + 1 for frame in frames]
        if execution_offset(current, row['exec_start_idx']) == 0:
            assert prefix['latents'].data_ptr() == base_payload['latents'].data_ptr()
    for bad_current in [row['exec_start_idx'] - 1, row['length'] - 1, row['length']]:
        with pytest.raises(ValueError, match='future execution action'):
            cache.prefix(row, bad_current, base_payload)


@pytest.mark.parametrize('corruption', ['incomplete', 'offset_missing', 'source_changed', 'clock_changed', 'escape'])
def test_dense_cache_rejects_incomplete_or_wrong_provenance(dense_cache_assets, corruption):
    arguments, manifest, row, _ = dense_cache_assets
    offsets = manifest['episodes'][str(row['episode_index'])]['offsets']
    if corruption == 'incomplete':
        manifest['complete'] = False
    elif corruption == 'offset_missing':
        del offsets['15']
    elif corruption == 'source_changed':
        manifest['metadata']['source_manifest_sha256'] = '0' * 64
    elif corruption == 'clock_changed':
        offsets['1']['frame_count'] += 1
    else:
        offsets['1']['path'] = '../outside.pt'
    (arguments[0] / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        DenseRoboMMECache(*arguments)


def test_dense_cache_checks_payload_checksum_before_read(dense_cache_assets):
    arguments, manifest, row, base = dense_cache_assets
    entry = manifest['episodes'][str(row['episode_index'])]['offsets']['1']
    with (arguments[0] / entry['path']).open('ab') as stream:
        stream.write(b'corrupted')
    cache = DenseRoboMMECache(*arguments)
    with pytest.raises(ValueError, match='checksum'):
        cache.prefix(row, row['exec_start_idx'] + 1, base)
