"""Shared causal query clock and immutable DreamDojo training-label reader."""
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .robomme_dense_clock import validate_dense_prefix

TARGET_SCHEMA = 'dreamdojo_adjacent_transition_at_history_decisions_v1'
INTERVAL_SCHEMA = 'dreamdojo_ordered_transitions_per_history_interval16_v1'
LABEL_DIM = 64
INTERVAL_STEPS = 16
SPARSE_MODE = 'uniform16_half8'
SPARSE_SCHEMA = 'dreamdojo_uniform16_half8_adjacent_transitions_v1'
SPARSE_QUERIES = 16
SPARSE_TRANSITIONS = 8
CONFIGURABLE_SPARSE_SCHEMA = 'dreamdojo_uniform_half8_adjacent_transitions_v2'


def validate_query_counts(source_history_slots=99, past_query_slots=SPARSE_QUERIES):
    for name, value in (('source_history_slots', source_history_slots),
                        ('past_query_slots', past_query_slots)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if past_query_slots > source_history_slots:
        raise ValueError('past_query_slots cannot exceed source_history_slots')
    return int(source_history_slots), int(past_query_slots)


def sparse_target_schema(source_history_slots=99, past_query_slots=SPARSE_QUERIES):
    counts = validate_query_counts(source_history_slots, past_query_slots)
    return SPARSE_SCHEMA if counts == (99, 16) else CONFIGURABLE_SPARSE_SCHEMA


def sparse_sampling_contract(source_history_slots=99, past_query_slots=SPARSE_QUERIES):
    source_history_slots, past_query_slots = validate_query_counts(
        source_history_slots, past_query_slots)
    contract = dict(
        query_slots=past_query_slots, source_history_slots=source_history_slots,
        query_selection='uniform_valid_interval_rank_rint_linspace_inclusive',
        query_padding='left_pad_invalid_no_repetition',
        transition_selection='endpoint_aligned_stride2_chronological',
        sampled_transition_stride=2, preserve_original_visual_slot=True)
    # Preserve old checkpoint metadata byte-for-byte for the original recipe.
    if (source_history_slots, past_query_slots) != (99, 16):
        contract['past_query_slots'] = past_query_slots
    return contract


def past_action_layout(frame_indices, execution_start, slots=99):
    """One query per visual-history decision, with identical offline/online padding.

    Labels describe the single stored-frame transition ENDING at each decision,
    not the entire 16-frame interval between decisions. Reconstructing the past
    may use all observations available at the current decision.
    """
    frames = np.asarray(frame_indices, dtype=np.int64)
    boundary = int(execution_start)
    if frames.ndim != 1 or len(frames) == 0 or slots < 1:
        raise ValueError('Expected a nonempty original-frame decision clock')
    # Includes the unchanged online offset-zero clock and explicitly prepared
    # training phase offsets. This still rejects missing/future/cross-phase grids.
    validate_dense_prefix(frames.tolist(), boundary)
    destinations = np.full(slots, -1, dtype=np.int64)
    selected = frames[-slots:]
    destinations[-len(selected):] = selected
    phase = np.zeros(slots, dtype=np.int64)
    present = destinations >= 0
    phase[present & (destinations < boundary)] = 1
    phase[present & (destinations >= boundary)] = 3
    if boundary:
        phase[destinations == boundary] = 2
    valid = present & (destinations > 0) & (phase != 2)
    ages = np.zeros(slots, dtype=np.float32)
    ages[present] = (destinations[present] - frames[-1]) / 16.0
    return dict(
        past_action_frame_indices=torch.from_numpy(destinations),
        past_action_phase=torch.from_numpy(phase),
        past_action_valid=torch.from_numpy(valid),
        past_action_relative_time=torch.from_numpy(ages),
    )


def past_action_interval_layout(frame_indices, execution_start, slots=99):
    """Partition observed transitions into ordered blocks ending at decisions.

    A block (s,e] holds E(s,s+1), ..., E(e-1,e), left aligned in 16
    output positions. Short-block tails are padding. The predecessor is found
    BEFORE cropping the history window, so its first block remains complete.
    The demo/execution boundary retains its query slot but no valid transition.
    """
    layout = past_action_layout(frame_indices, execution_start, slots)
    frames = np.asarray(frame_indices, dtype=np.int64)
    previous = np.concatenate((frames[:1], frames[:-1]))
    starts = np.full(slots, -1, dtype=np.int64)
    starts[-min(slots, len(frames)):] = previous[-slots:]
    ends = layout['past_action_frame_indices'].numpy()
    present = ends >= 0
    lengths = np.where(present, ends - starts, 0)
    if np.any(lengths < 0) or np.any(lengths > INTERVAL_STEPS):
        raise ValueError('Observed interval length must be between 0 and 16')
    offsets = np.arange(INTERVAL_STEPS)
    occupied = present[:, None] & (offsets[None] < lengths[:, None])
    destinations = np.where(occupied, starts[:, None] + offsets + 1, -1)
    valid = occupied & (destinations > 0)
    if execution_start:
        valid &= destinations != int(execution_start)
    # Transition blocks may not combine demo and execution.
    for q in np.flatnonzero(valid.any(1)):
        f = destinations[q, valid[q]]
        if np.any(f < execution_start) and np.any(f > execution_start):
            raise ValueError('Past-action interval crosses phases')
    layout.update(
        past_action_interval_start=torch.from_numpy(starts),
        past_action_interval_length=torch.from_numpy(lengths),
        past_action_transition_frames=torch.from_numpy(destinations),
        past_action_transition_valid=torch.from_numpy(valid),
        past_action_valid=torch.from_numpy(valid.any(1)),
    )
    return layout


def past_action_sparse_layout(frame_indices, execution_start, slots=99, *, query_slots=SPARSE_QUERIES):
    """Select uniform observed intervals; retain half their adjacent transitions.

    Selection uses only the observed clock, identically offline and online.
    Each selected label still encodes (frame-1, frame), not a stride-two pair.
    Original visual slot and interval length are preserved after subsampling.
    """
    slots, query_slots = validate_query_counts(slots, query_slots)
    dense = past_action_interval_layout(frame_indices, execution_start, slots)
    candidates = torch.nonzero(dense['past_action_valid'], as_tuple=False).flatten().numpy()
    if len(candidates) > query_slots:
        ranks = np.rint(np.linspace(0, len(candidates)-1, query_slots)).astype(np.int64)
        candidates = candidates[ranks]
    result = {}
    for name in ('past_action_frame_indices', 'past_action_phase',
                 'past_action_relative_time', 'past_action_interval_start',
                 'past_action_interval_length'):
        fill = -1 if name in ('past_action_frame_indices', 'past_action_interval_start') else 0
        result[name] = torch.full((query_slots,), fill, dtype=dense[name].dtype)
        if len(candidates):
            result[name][-len(candidates):] = dense[name][candidates]
    result['past_action_visual_slots'] = torch.full((query_slots,), -1, dtype=torch.long)
    frames = torch.full((query_slots, SPARSE_TRANSITIONS), -1, dtype=torch.long)
    valid = torch.zeros_like(frames, dtype=torch.bool)
    offsets = torch.zeros_like(frames)
    for out_slot, source_slot in enumerate(candidates, start=query_slots-len(candidates)):
        positions = torch.nonzero(dense['past_action_transition_valid'][source_slot],
                                  as_tuple=False).flatten().tolist()
        positions = positions[::-2][::-1]
        count = len(positions)
        frames[out_slot, :count] = dense['past_action_transition_frames'][source_slot, positions]
        offsets[out_slot, :count] = torch.tensor(positions, dtype=torch.long) + 1
        valid[out_slot, :count] = True
        result['past_action_visual_slots'][out_slot] = int(source_slot)
    result.update(past_action_transition_frames=frames, past_action_transition_valid=valid,
                  past_action_transition_offsets=offsets, past_action_valid=valid.any(-1))
    return result


class DreamDojoLabels:
    """Read finalized labels by original episode/frame, with pinned provenance."""
    def __init__(self, root, dataset_root, rows):
        from fastwam.datasets.robodojo import sha256
        self.root = Path(root)
        dataset_root = Path(dataset_root)
        self.config = json.loads((self.root / 'config.json').read_text())
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m, c = self.manifest, self.config
        if (m.get('status') != 'complete' or c.get('schema_version') != 1
                or m.get('fingerprint') != c.get('fingerprint')
                or c.get('latent_dim') != LABEL_DIM
                or c.get('transition_frame_stride') != 1
                or c.get('views') != ['observation.images.cam_high', 'observation.images.cam_left_wrist']):
            raise ValueError('Incompatible/incomplete DreamDojo label cache')
        for name in ('info', 'episodes'):
            suffix = 'jsonl' if name == 'episodes' else 'json'
            if c[name + '_sha256'] != sha256(dataset_root / 'meta' / f'{name}.{suffix}'):
                raise ValueError('DreamDojo labels belong to different source episodes')
        for name, expected in m['files_sha256'].items():
            if sha256(self.root / name) != expected:
                raise ValueError(f'DreamDojo cache checksum mismatch: {name}')
        self.manifest_sha256 = sha256(self.root / 'manifest.json')
        stats = m['normalization']['all']
        self.mean = np.asarray(stats['mean'], dtype=np.float32)
        self.std = np.asarray(stats['std'], dtype=np.float32)
        if (self.mean.shape != (LABEL_DIM,) or self.std.shape != (LABEL_DIM,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 1e-6).any()):
            raise ValueError('Invalid DreamDojo normalization statistics')
        with np.load(self.root / 'frame_index.npz', allow_pickle=False) as f:
            self.valid = f['valid'].copy()
            self.phase = f['phase_id'].copy()
            episode_index = f['episode_index']
            frame_index = f['frame_index']
            if not np.array_equal(f['global_index'], np.arange(len(frame_index))):
                raise ValueError('DreamDojo global index is not contiguous')
        self.offsets = {}
        offset = 0
        for row in rows:
            ep, length = int(row['episode_index']), int(row['length'])
            sl = slice(offset, offset + length)
            if (not np.all(episode_index[sl] == ep)
                    or not np.array_equal(frame_index[sl], np.arange(length))):
                raise ValueError(f'DreamDojo episode/frame mismatch: {ep}')
            self.offsets[ep] = (offset, length)
            offset += length
        if offset != m['total_frames'] or m['total_episodes'] != len(rows):
            raise ValueError('DreamDojo inventory differs from source dataset')
        self._latents = None
        values = np.load(self.root / 'latents.npy', mmap_mode='r', allow_pickle=False)
        if values.shape != (offset, LABEL_DIM) or values.dtype != np.float32:
            raise ValueError('Invalid DreamDojo latent table shape/dtype')

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_latents'] = None
        return state

    def targets(self, episode_index, layout):
        if self._latents is None:
            self._latents = np.load(self.root / 'latents.npy', mmap_mode='r', allow_pickle=False)
        start, length = self.offsets[int(episode_index)]
        interval = 'past_action_transition_frames' in layout
        frames = layout['past_action_transition_frames' if interval else 'past_action_frame_indices'].numpy()
        valid = layout['past_action_transition_valid' if interval else 'past_action_valid'].numpy()
        phase = layout['past_action_phase'].numpy()
        if interval:
            phase = np.broadcast_to(phase[:, None], frames.shape)
        present = frames >= 0
        if np.any(frames[present] >= length):
            raise ValueError('DreamDojo query outside the episode')
        indices = start + np.clip(frames, 0, length - 1)
        if (not np.array_equal(self.valid[indices[present]], valid[present])
                or not np.array_equal(self.phase[indices[present]], phase[present])):
            raise ValueError('DreamDojo boundary mask differs from the shared query clock')
        values = (np.asarray(self._latents[indices]) - self.mean) / self.std
        values[~valid] = 0
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite normalized DreamDojo target')
        return torch.from_numpy(values.copy())

    def contract(self, target_mode='endpoint', *, source_history_slots=99,
                 past_query_slots=SPARSE_QUERIES):
        contract = dict(schema=TARGET_SCHEMA, latent_dim=LABEL_DIM,
                    label_fingerprint=self.manifest['fingerprint'],
                    label_manifest_sha256=self.manifest_sha256,
                    normalization='valid_training_corpus_per_dimension_zscore',
                    normalization_mean=self.mean.tolist(), normalization_std=self.std.tolist(),
                    transition_frame_stride=1, target_alignment='source=i-1,destination=i',
                    boundary_supervision=False, teacher_input=False)
        if target_mode in ('ordered_interval16', SPARSE_MODE):
            contract.update(
                schema=INTERVAL_SCHEMA, query_target_shape=[INTERVAL_STEPS, LABEL_DIM],
                interval_alignment='previous_decision_exclusive_to_current_decision_inclusive',
                interval_padding='chronological_left_aligned_invalid_tail',
                window_predecessor='taken_before_cropping_history',
                loss_reduction='mean_valid_transitions_per_example_then_mean_nonempty_examples')
            if target_mode == SPARSE_MODE:
                contract.update(schema=sparse_target_schema(source_history_slots, past_query_slots),
                                query_target_shape=[SPARSE_TRANSITIONS, LABEL_DIM],
                                **sparse_sampling_contract(source_history_slots, past_query_slots))
        elif target_mode != 'endpoint':
            raise ValueError(f'Unknown past-action target mode: {target_mode}')
        return contract
