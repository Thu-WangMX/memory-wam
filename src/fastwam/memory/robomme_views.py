"""Versioned RGB layout shared by RoboMME training, caches, and online policy."""
from .robomme_clock import CACHE_SCHEMA

LEGACY_VIEW_LAYOUT = 'legacy'
TWO_VIEW_HORIZONTAL = 'two_view_horizontal'
TWO_VIEW_BASE_SCHEMA = 'robomme_phase_two_view_base_v1'


def view_spec(layout=LEGACY_VIEW_LAYOUT):
    if layout == LEGACY_VIEW_LAYOUT:
        return dict(view_layout=layout, view_layout_version=1, height=384, width=320,
                    latent_shape=[48, 1, 24, 20],
                    mosaic='wrists_top_head_bottom_384x320',
                    camera_order='wrist_wrist_above_front_rgb',
                    resize='legacy_compose_rgb_two_stage_bilinear_antialias')
    if layout == TWO_VIEW_HORIZONTAL:
        return dict(view_layout=layout, view_layout_version=2, height=256, width=512,
                    latent_shape=[48, 1, 16, 32],
                    mosaic='front_left_wrist_right_256x512',
                    camera_order='front_left_wrist_right_rgb',
                    resize='each_view_256x256_bilinear_antialias_no_crop')
    raise ValueError(f'Unknown RoboMME view layout: {layout}')


def base_cache_schema(layout=LEGACY_VIEW_LAYOUT):
    view_spec(layout)
    return CACHE_SCHEMA if layout == LEGACY_VIEW_LAYOUT else TWO_VIEW_BASE_SCHEMA


def validate_view_metadata(metadata, layout=None, *, dense=False):
    """Infer only a declared new layout; old metadata remains legacy-compatible.

    New caches must carry the complete view contract. A differently shaped cache
    never gets silently reinterpreted as a new view layout.
    """
    inferred = metadata.get('view_layout', LEGACY_VIEW_LAYOUT)
    layout = inferred if layout is None else layout
    spec = view_spec(layout)
    if inferred != layout:
        raise ValueError(f'RoboMME cache view_layout differs: expected {layout}, got {inferred}')
    if metadata.get('latent_shape') != spec['latent_shape']:
        raise ValueError('RoboMME cache latent_shape differs from view layout')
    if layout != LEGACY_VIEW_LAYOUT:
        if metadata.get('view_contract') != spec:
            raise ValueError('RoboMME cache view_contract/layout version differs')
        if metadata.get('view_layout_version') != spec['view_layout_version']:
            raise ValueError('RoboMME cache view_layout_version differs')
    elif 'view_contract' in metadata and metadata['view_contract'] != spec:
        raise ValueError('RoboMME legacy view_contract differs')
    if not dense and metadata.get('mosaic') != spec['mosaic']:
        raise ValueError('RoboMME cache mosaic differs from view layout')
    return spec


def mosaic(front, wrist, view_layout=LEGACY_VIEW_LAYOUT):
    """RGB uint8 HWC -> normalized float32 [3,H,W], without temporal state."""
    import numpy as np
    import torch
    spec = view_spec(view_layout)
    cameras = []
    for image in (front, wrist):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Expected RGB uint8 HWC front/wrist observations')
        cameras.append(torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0))
    if view_layout == LEGACY_VIEW_LAYOUT:
        from fastwam.datasets.robodojo import compose_rgb
        return compose_rgb([cameras[0], cameras[1], cameras[1]])[0]
    from torchvision.transforms.functional import resize
    views = [resize(value.float().div(255), [spec['height'], spec['width']//2],
                    antialias=True) for value in cameras]
    return torch.cat(views, dim=-1)[0].mul(2).sub(1)
