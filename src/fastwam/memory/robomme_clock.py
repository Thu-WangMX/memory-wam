"""RoboMME phase clock shared by the dataset and online policy.

Matches Kevin's phase_local_endpoint_linspace4 cache, without his event writer.
The final demo observation is retained even if its interval is shorter than 16.
"""
import numpy as np

CONTRACT = 'robomme_anchor_helios_phase_v1'
CACHE_SCHEMA = 'robomme_multitask_phase_clock_kemo_k8_v2'
HORIZON = 16
LEGACY_ANCHOR_MODE = 'global_first2'
PHASE_ANCHOR_MODE = 'demo_execution_first'


def validate_anchor_mode(mode):
    if mode not in (LEGACY_ANCHOR_MODE, PHASE_ANCHOR_MODE):
        raise ValueError(f'Unknown RoboMME anchor mode: {mode}')
    return mode


def select_robomme_anchors(history, frame_indices=None, execution_start=None,
                           size=2, mode=LEGACY_ANCHOR_MODE):
    """Select causal fixed anchors; preserve legacy checkpoint behavior by default.

    Phase slots are always [demo first, execution first]. An absent phase is
    zero-filled and marked invalid, never replaced by a later observation.
    The caller passes the FULL observed prefix, before rolling-window cropping.
    """
    import torch
    from .helios_history import episode_anchors
    validate_anchor_mode(mode)
    if mode == LEGACY_ANCHOR_MODE:
        return episode_anchors(history, size)
    if size not in (0, 2) or history.ndim != 5 or history.shape[0] < 1:
        raise ValueError('Phase anchors require nonempty [T,C,1,H,W] and size 0 or 2')
    if size == 0:
        return history[:0], torch.zeros(0, device=history.device, dtype=torch.bool)
    if frame_indices is None or execution_start is None:
        raise ValueError('Phase anchors require the original-frame clock and execution_start')
    frames = [int(x) for x in frame_indices]
    if (len(frames) != len(history) or not frames or frames[0] != 0
            or execution_start < 0 or any(b <= a for a, b in zip(frames, frames[1:]))
            or (frames[-1] >= execution_start and execution_start not in frames)):
        raise ValueError('Invalid full observed-prefix clock for phase anchors')
    anchors = history.new_zeros((2, *history.shape[1:]))
    valid = torch.zeros(2, device=history.device, dtype=torch.bool)
    if execution_start > 0:
        anchors[0], valid[0] = history[0], True
    if execution_start in frames:
        anchors[1], valid[1] = history[frames.index(execution_start)], True
    return anchors, valid


def decision_frames(length, execution_start):
    if not 0 <= execution_start < length:
        raise ValueError('Invalid demonstration/execution boundary')
    demo = list(range(0, execution_start, HORIZON))
    if demo and demo[-1] != execution_start - 1:
        demo.append(execution_start - 1)
    return demo + list(range(execution_start, length, HORIZON))


def phase_sample_indices(endpoints):
    if not endpoints:
        return []
    if any(b <= a for a, b in zip(endpoints, endpoints[1:])):
        raise ValueError('Phase endpoints must increase')
    indices = [endpoints[0]]
    for start, end in zip(endpoints, endpoints[1:]):
        indices.extend(np.rint(np.linspace(start, end, 5)).astype(int).tolist()[1:])
    return indices


def inference_contract(anchor_frames=2, anchor_mode=LEGACY_ANCHOR_MODE, view_layout='legacy'):
    validate_anchor_mode(anchor_mode)
    contract = dict(observation_contract=CONTRACT,
                history_vae='separate_demo_execution_causal_vae',
                temporal_sampling='phase_local_endpoint_linspace4',
                vae_phase_reset=True, memory_phase_reset=False,
                anchor_source_indices='first_two_global_decision_latents' if anchor_frames else [],
                action_offset=1, observation_timing='after_recorded_action',
                action_dim=8, proprio_dim=8, action_space='joint_angle',
                gripper_action='minus1_close_plus1_open',
                gripper_state='mean_two_finger_positions',
                camera_order='wrist_wrist_above_front_rgb',
                supervision='execution_only', replan_stride=HORIZON)
    if anchor_mode == PHASE_ANCHOR_MODE:
        contract.update(anchor_mode=anchor_mode,
                        anchor_source_indices='demo_first_and_execution_first' if anchor_frames else [],
                        anchor_slot_roles=['demo', 'execution'] if anchor_frames else [],
                        absent_anchor='zero_latent_masked_key',
                        anchor_position='fixed_role_slots_0_1')
    if view_layout != 'legacy':
        from .robomme_views import view_spec
        contract.update(camera_order='front_left_wrist_right_rgb', view_contract=view_spec(view_layout))
    return contract
