"""Mask or compact fixed-grid Helios startup padding without moving RoPE slots.

Convolutions retain their original right-aligned temporal groups. A patch that
contains any observed latent is retained, including a mixed zero/real boundary
patch. Compaction removes only history columns invalid for the whole batch;
per-example key masks handle the remaining batch padding at every layer.
"""
import math

import torch


HISTORY_PADDING_MODES = ('legacy_zero', 'mask', 'compact')


def validate_history_padding(mode, memory_variant=None):
    if mode not in HISTORY_PADDING_MODES:
        raise ValueError(f'Unknown history_padding: {mode}')
    if mode != 'legacy_zero' and memory_variant == 'full_history_dense':
        raise ValueError('history_padding mask/compact cannot stack with full_history_dense')
    return mode


def history_padding_contract(mode):
    validate_history_padding(mode)
    if mode == 'legacy_zero':
        return {}
    return dict(history_padding=mode,
                history_padding_contract='fixed_grid_patch_any_valid_v1',
                history_padding_rope='preserve_virtual_frame_positions',
                history_padding_anchors='keep_role_slots_separately_masked')


def zero_invalid_history(history, valid):
    """Sanitize missing raw latents before any convolution, including mixed patches."""
    if (history.ndim != 6 or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool or tuple(valid.shape) != tuple(history.shape[:2])):
        raise ValueError('history_valid must be boolean [B,H] for history [B,H,C,1,h,w]')
    valid = valid.to(device=history.device)
    if not valid[:, -1].all():
        raise ValueError('The current observation must be valid')
    return torch.where(valid[:, :, None, None, None, None], history, history.new_zeros(()))


def _segment_valid(valid, kernel, height, width):
    """Mirror pad_for_3d_conv's trailing replicate pad before nonoverlap patches."""
    kt, kh, kw = (int(value) for value in kernel)
    if min(kt, kh, kw) < 1:
        raise ValueError('Helios patch kernels must be positive')
    batch, length = valid.shape
    if not length:
        return valid
    padding = (-length) % kt
    if padding:
        valid = torch.cat((valid, valid[:, -1:].expand(-1, padding)), dim=1)
    groups = valid.reshape(batch, -1, kt).any(-1)
    spatial = math.ceil(height / kh) * math.ceil(width / kw)
    return groups.repeat_interleave(spatial, dim=1)


def apply_history_padding(pre, valid, *, sizes, kernels, height, width,
                          action_length, mode):
    """Apply validity to all history readers and optionally shorten video tokens.

    ``pre`` is a fixed-grid MTM ``pre_dit`` result. Future tokens, future-head
    time embeddings, language context, and virtual frame coordinates stay intact.
    Anchor token columns are always retained; the caller applies anchor_valid.
    """
    validate_history_padding(mode)
    if mode == 'legacy_zero':
        return pre
    batch = pre['tokens'].shape[0]
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or tuple(valid.shape) != (batch, sum(sizes)) or len(sizes) != 3
            or sizes[-1] != 1 or len(kernels) != 3):
        raise ValueError('Fixed-grid history validity does not match the memory profile')
    valid = valid.to(device=pre['tokens'].device)
    if not valid[:, -1].all():
        raise ValueError('The current observation must be valid')
    meta, mtm = pre['meta'], pre['meta']['mtm']
    h, p, anchors = mtm['L_hist'], mtm['L_pred'], mtm['L_anchor']
    segment_valid = [_segment_valid(v, k, height, width)
                     for v, k in zip(valid.split(tuple(sizes), dim=1), kernels)]
    for name, values in zip(('L_long', 'L_mid', 'L_curr'), segment_valid):
        if values.shape[1] != mtm[name]:
            raise ValueError(f'Patch validity length disagrees with MTM {name}')
    history_valid = torch.cat((
        torch.ones((batch, anchors), dtype=torch.bool, device=valid.device),
        *segment_valid), dim=1)
    if history_valid.shape[1] != h or pre['tokens'].shape[1] != h + p:
        raise ValueError('Patch validity differs from the video token layout')
    key_valid = torch.cat((history_valid,
        torch.ones((batch, p + action_length), dtype=torch.bool, device=valid.device)), dim=1)
    existing = meta.get('key_valid')
    if existing is not None:
        if existing.shape != key_valid.shape:
            raise ValueError('Existing key validity differs from the fixed-grid layout')
        key_valid &= existing.to(device=valid.device, dtype=torch.bool)

    # Keep anchors even when their phase is absent. The fixed role layout is
    # consumed by _apply_anchor_validity after this function.
    keep = torch.arange(h, device=valid.device)
    if mode == 'compact':
        keep = torch.nonzero(history_valid.any(0), as_tuple=False).flatten()
    meta['history_keep_indices'] = keep
    if keep.numel() != h:
        video_keep = torch.cat((keep, torch.arange(h, h+p, device=valid.device)))
        pre['tokens'] = pre['tokens'].index_select(1, video_keep)
        pre['t_mod'] = pre['t_mod'].index_select(1, video_keep)
        pre['context_mask'] = pre['context_mask'].index_select(1, video_keep)
        freq_axis = 1 if pre['freqs'].ndim == 4 else 0
        pre['freqs'] = pre['freqs'].index_select(freq_axis, video_keep)
        keys = torch.cat((video_keep, torch.arange(h+p, h+p+action_length, device=valid.device)))
        key_valid = key_valid.index_select(1, keys)
        start = anchors
        for name in ('L_long', 'L_mid', 'L_curr'):
            old_length = mtm[name]
            mtm[name] = int(((keep >= start) & (keep < start + old_length)).sum())
            start += old_length
        if 'L_short' in mtm:
            mtm['L_short'] = mtm['L_curr']
        mtm['L_hist'] = int(keep.numel())
    meta['key_valid'] = key_valid
    meta['history_key_valid'] = key_valid[:, :mtm['L_hist']]
    meta['history_padding'] = mode
    return pre
