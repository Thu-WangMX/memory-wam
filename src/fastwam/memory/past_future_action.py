"""Shared online/offline queries: configurable past blocks and two future blocks.

No trajectory length enters the query layout. End-of-trajectory availability
is a separate training loss mask, never an attention/input mask.
"""
import torch
from .past_action import past_action_sparse_layout, sparse_sampling_contract, validate_query_counts

MODE = 'past16_future2_half8'
SCHEMA = 'dreamdojo_past16_future2_half8_predict32_execute16_v1'
CONFIGURABLE_SCHEMA = 'dreamdojo_past_future2_half8_predict32_execute16_v2'


def target_schema(source_history_slots=99, past_query_slots=16):
    counts = validate_query_counts(source_history_slots, past_query_slots)
    return SCHEMA if counts == (99, 16) else CONFIGURABLE_SCHEMA


def sampling_contract(source_history_slots=99, past_query_slots=16):
    source_history_slots, past_query_slots = validate_query_counts(source_history_slots, past_query_slots)
    return dict(sparse_sampling_contract(source_history_slots, past_query_slots),
                query_slots=past_query_slots+2, past_query_slots=past_query_slots,
                future_query_slots=2, future_horizon=32, future_block_steps=16,
                future_offsets=[2, 4, 6, 8, 10, 12, 14, 16],
                future_query_validity='always_present_no_episode_length',
                future_padding='loss_only_unknown_at_inference',
                future_attention='past_cannot_read_future_future_reads_past_and_future')


def past_future_layout(frame_indices, execution_start, slots=99, *, past_query_slots=16):
    past = past_action_sparse_layout(frame_indices, execution_start, slots, query_slots=past_query_slots)
    current = int(frame_indices[-1])
    starts = torch.tensor([current, current+16], dtype=torch.long)
    offsets = torch.arange(2, 17, 2).expand(2, -1).clone()
    future = dict(
        past_action_frame_indices=starts+16,
        past_action_phase=torch.full((2,), 3, dtype=torch.long),
        past_action_relative_time=torch.tensor([1., 2.]),
        past_action_interval_start=starts,
        past_action_interval_length=torch.full((2,), 16, dtype=torch.long),
        past_action_visual_slots=torch.tensor([slots, slots+1]),
        past_action_transition_frames=starts[:, None]+offsets,
        past_action_transition_valid=torch.ones(2, 8, dtype=torch.bool),
        past_action_transition_offsets=offsets,
        past_action_valid=torch.ones(2, dtype=torch.bool))
    return {k: torch.cat((v, future[k]), 0) for k, v in past.items()}


def training_targets(labels, episode, layout):
    """Read existing adjacent-frame codes; retain fixed future query positions."""
    _, length = labels.offsets[int(episode)]
    valid = layout['past_action_transition_valid'].clone()
    valid &= layout['past_action_transition_frames'] < length
    reader_layout = dict(layout)
    reader_layout['past_action_transition_valid'] = valid
    reader_layout['past_action_transition_frames'] = layout['past_action_transition_frames'].masked_fill(~valid, -1)
    return labels.targets(episode, reader_layout), valid
