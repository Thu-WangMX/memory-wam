"""Causal execution-phase augmentation for every-frame RoboMME supervision.

Inference keeps offset zero. Training at t selects r=(t-execution_start)%16.
Every stream begins at the true execution initial observation; only its first
interval may be shorter than 16. Demo observations and phase reset do not move.
"""
from .robomme_clock import HORIZON, decision_frames


DENSE_SCHEMA = 'robomme_phase_offset16_causal_vae_v1'


def execution_offset(current, execution_start):
    current, execution_start = int(current), int(execution_start)
    if execution_start < 0 or current < execution_start:
        raise ValueError('Dense samples require an observed execution frame')
    return (current - execution_start) % HORIZON


def dense_execution_frames(length, execution_start, offset=0):
    length, execution_start, offset = int(length), int(execution_start), int(offset)
    if not 0 <= execution_start < length or not 0 <= offset < HORIZON:
        raise ValueError('Invalid execution boundary or dense-clock offset')
    if offset == 0:
        return list(range(execution_start, length, HORIZON))
    return [execution_start] + list(range(execution_start + offset, length, HORIZON))


def dense_decision_frames(length, execution_start, offset=0):
    execution = dense_execution_frames(length, execution_start, offset)
    demo = [frame for frame in decision_frames(length, execution_start)
            if frame < execution_start]
    return demo + execution


def dense_prefix_frames(current, execution_start):
    offset = execution_offset(current, execution_start)
    return dense_decision_frames(int(current) + 1, execution_start, offset)


def validate_dense_prefix(frame_indices, execution_start):
    frames = [int(frame) for frame in frame_indices]
    if not frames:
        raise ValueError('Expected a nonempty observed execution prefix')
    offset = execution_offset(frames[-1], execution_start)
    if frames != dense_prefix_frames(frames[-1], execution_start):
        raise ValueError('Observed clock differs from the causal execution-offset stream')
    return offset


def dense_clock_contract():
    return dict(schema=DENSE_SCHEMA, training_sample_stride=1,
                execution_offsets=list(range(HORIZON)), inference_execution_offset=0,
                replan_stride=HORIZON,
                first_interval='execution_start_to_execution_start_plus_offset',
                subsequent_intervals='stride16',
                interval_rgb_sampling='rint_linspace5_drop_first',
                execution_anchor='true_execution_start_shared_from_base_cache',
                demonstration='unchanged_shared_from_base_cache',
                causal_vae='independent_state_per_execution_offset',
                future_observations_in_condition=False)
