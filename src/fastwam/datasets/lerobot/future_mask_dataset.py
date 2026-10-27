"""Strict Observe-only join of reviewed future mask targets and RGB windows."""
import hashlib
import io
import json
from pathlib import Path

import torch
from .helios_dataset import HeliosRobotVideoDataset
from .full_kv_dataset import _as_int


class FutureMaskHeliosDataset(HeliosRobotVideoDataset):
    def __init__(self, *args, future_mask_cache_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_root = Path(future_mask_cache_path).resolve()
        manifest = json.loads((self.mask_root / 'manifest.json').read_text())
        expected = dict(schema='future_head_mask_local_clip_v1', completed=True,
                        replan_stride=16, offsets=[0, 4, 8, 12, 16])
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f'Incompatible future mask cache: {key}')
        if (self.replan_stride != 16 or self.num_frames != 17
                or self.action_video_freq_ratio != 4):
            raise ValueError('Future mask cache requires 17-frame clips with stride16 / sample4')
        names = self.lerobot_dataset.multi_dataset.ds_names
        if len(names) != 1 or Path(str(names[0])).name != 'observe_and_pickup':
            raise ValueError('This reviewed mask cache belongs exclusively to observe_and_pickup')
        self.mask_rows = {}
        self.mask_lengths = {int(e['episode_index']): int(e['num_frames']) for e in manifest['episodes']}
        for row in manifest['samples']:
            key = (int(row['episode_index']), int(row['frame_index']))
            if key in self.mask_rows:
                raise ValueError(f'Duplicate mask window: {key}')
            self.mask_rows[key] = row
        if len(self.mask_rows) != int(manifest['sample_count']):
            raise ValueError('Mask manifest sample count mismatch')
        self._mask_payloads = {}

    def _get(self, idx):
        sample = super()._get(idx)
        key = (_as_int(sample['episode_index']), _as_int(sample['frame_index']))
        row = self.mask_rows[key]
        payload = self._mask_payloads.get(key)
        if payload is None:
            path = self.mask_root / f'episode_{key[0]:06d}' / f'frame_{key[1]:06d}.pt'
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError(f'Mask cache checksum mismatch: {path}')
            payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            if (_as_int(payload['episode_index']), _as_int(payload['frame_index'])) != key:
                raise ValueError('Mask payload provenance mismatch')
            latent = payload['future_mask_latent']
            if latent.shape != (48, 1, 24, 20) or latent.dtype != torch.bfloat16 or not torch.isfinite(latent).all():
                raise ValueError('Invalid future mask target shape/dtype/value')
            raw_indices = torch.arange(0, 17, 4) + key[1]
            length = self.mask_lengths[key[0]]
            if not torch.equal(payload['frame_indices'], raw_indices.clamp(max=length-1)):
                raise ValueError('Mask frame indices mismatch')
            if not torch.equal(payload['image_is_pad'], raw_indices >= length):
                raise ValueError('Mask padding metadata mismatch')
            self._mask_payloads[key] = payload
        if not torch.equal(sample['image_is_pad'].bool(), payload['image_is_pad']):
            raise ValueError(f'RGB/mask window padding mismatch: {key}')
        sample['future_mask_latent'] = payload['future_mask_latent']
        sample['future_mask_image_is_pad'] = payload['image_is_pad']
        return sample
