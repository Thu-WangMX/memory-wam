"""Shared train/deploy Helios history slicing, independent of data loading."""
import torch


def episode_anchors(history, size=2):
    """Return the earliest causal decision latents as fixed episode anchors.

    Missing startup anchors are zero-filled on the right. This preserves a
    fixed token layout without ever copying a future observation backwards.
    """
    if history.ndim != 5 or history.shape[0] < 1 or size < 0:
        raise ValueError("Expected nonempty history [H,C,1,h,w] and non-negative size")
    if size == 0:
        return history[:0], torch.zeros(0, device=history.device, dtype=torch.bool)
    selected = history[:size]
    valid = torch.arange(size, device=history.device) < len(selected)
    if len(selected) < size:
        selected = torch.cat(
            [selected, history.new_zeros((size - len(selected), *history.shape[1:]))]
        )
    return selected, valid


def fixed_history(history, size=19):
    """Keep the current decision and its predecessors; left-pad startup with zero latents.

    Padding is the original Helios startup convention, not a future observation.
    Both training and inference use this function before patchification.
    """
    if history.ndim != 5 or history.shape[0] < 1 or size < 1:
        raise ValueError("Expected nonempty history [H,C,1,h,w] and positive size")
    selected = history[-size:]
    valid = torch.arange(size, device=history.device) >= size - len(selected)
    if len(selected) < size:
        selected = torch.cat([history.new_zeros((size-len(selected), *history.shape[1:])), selected])
    return selected, valid



# The long99 profile extends the temporal coordinate range with its history.
# The original 19-slot ablations preserve current=20/future=21.
# Uniform scale keeps independent Long/Mid parameters: only compression changes.
MEMORY_PROFILES = {
    'multiscale': ((16,2,1), (4,8,8), (2,4,4), 2, 0),
    'multiscale_long99': ((96,2,1), (4,8,8), (2,4,4), 2, 0),
    'long75_less_temporal': ((72,2,1), (2,8,8), (2,4,4), 2, 0),
    'long75_less_spatial': ((72,2,1), (4,4,4), (2,4,4), 2, 0),
    'multiscale_long75': ((72,2,1), (4,8,8), (2,4,4), 2, 0),
    'long99_less_temporal': ((96,2,1), (2,8,8), (2,4,4), 2, 0),
    'long99_less_spatial': ((96,2,1), (4,4,4), (2,4,4), 2, 0),
    'anchor_only': ((0,0,1), (4,8,8), (2,4,4), 2, 18),
    'recent19_dense': ((16,2,1), (1,2,2), (1,2,2), 2, 0),
    'full_history_dense': ((16,2,1), (1,2,2), (1,2,2), 2, 0),
    'wo_long': ((0,2,1), (4,8,8), (2,4,4), 2, 16),
    'uniform_mid': ((16,2,1), (2,4,4), (2,4,4), 2, 0),
    'reverse_scale': ((2,16,1), (2,4,4), (4,8,8), 2, 0),
    'wo_anchor': ((16,2,1), (4,8,8), (2,4,4), 0, 2),
    'long_less_temporal': ((16,2,1), (2,8,8), (2,4,4), 2, 0),
    'long_less_spatial': ((16,2,1), (4,4,4), (2,4,4), 2, 0),
}
MEMORY_VARIANTS = tuple(MEMORY_PROFILES)


def memory_variant_profile(variant):
    if variant not in MEMORY_PROFILES:
        raise ValueError(f'Unknown Helios memory_variant: {variant}')
    sizes, long, mid, anchors, offset = MEMORY_PROFILES[variant]
    return dict(history_sizes=list(sizes), patch_long=list(long), patch_mid=list(mid),
                anchor_frames=anchors, history_position_offset=offset)


def history_position_offset(variant):
    return memory_variant_profile(variant)['history_position_offset']


def validate_memory_variant(variant, sizes, patch_long, patch_mid, anchor_frames=None):
    if variant not in MEMORY_VARIANTS:
        raise ValueError(f'Unknown Helios memory_variant: {variant}')
    if variant in ('wo_long', 'uniform_mid', 'reverse_scale', 'wo_anchor',
                   'long_less_temporal', 'long_less_spatial', 'multiscale_long99',
                   'long99_less_temporal', 'long99_less_spatial', 'multiscale_long75', 'long75_less_spatial', 'long75_less_temporal'):
        expected = memory_variant_profile(variant)
        if (list(sizes) != expected['history_sizes'] or
                list(patch_long) != expected['patch_long'] or
                list(patch_mid) != expected['patch_mid'] or
                (anchor_frames is not None and anchor_frames != expected['anchor_frames'])):
            raise ValueError(f'{variant} requires its fixed memory profile: {expected}')
    if variant == 'anchor_only' and tuple(sizes) != (0, 0, 1):
        raise ValueError('anchor_only requires history_sizes=[0,0,1] (keep current)')
    if variant in ('recent19_dense', 'full_history_dense'):
        if tuple(sizes) != (16, 2, 1):
            raise ValueError('Dense ablations require base history_sizes=[16,2,1]')
        if tuple(patch_long) != (1, 2, 2) or tuple(patch_mid) != (1, 2, 2):
            raise ValueError('Dense ablations require both history patches=[1,2,2]')


def select_history(history, sizes=(16, 2, 1), variant='multiscale'):
    """One shared train/inference selector; current is included, never future.

    Full history keeps the dense19 startup layout, then extends it without a
    cap. When enabled, anchors remain separate, even if duplicated in history.
    """
    if variant not in MEMORY_VARIANTS:
        raise ValueError(f'Unknown Helios memory_variant: {variant}')
    size = sum(sizes)
    if variant == 'full_history_dense':
        size = max(size, len(history))
    return fixed_history(history, size)
