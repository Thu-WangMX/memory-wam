"""Immutable supplemental VAE cache for every-frame RoboMME training starts."""
from collections import OrderedDict
import json
from pathlib import Path

import torch

from .robodojo import sha256
from fastwam.memory.robomme_dense_clock import (
    DENSE_SCHEMA, dense_execution_frames, dense_prefix_frames, execution_offset,
)
from fastwam.memory.robomme_views import LEGACY_VIEW_LAYOUT, validate_view_metadata


class DenseRoboMMECache:
    def __init__(self, root, dataset_root, base_root, base_manifest, rows):
        self.root = Path(root).resolve()
        self.base_root = Path(base_root).resolve()
        self.manifest_path = self.root / 'manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if m.get('schema') != DENSE_SCHEMA or m.get('complete') is not True:
            raise ValueError('A complete dense VAE cache is required; run preparation/finalize first')
        metadata = m.get('metadata', {})
        self.view_contract = validate_view_metadata(base_manifest['metadata'])
        self.view_layout = self.view_contract['view_layout']
        validate_view_metadata(metadata, self.view_layout, dense=True)
        self.source_hash = sha256(Path(base_root) / 'manifest.json')
        expected = dict(source_manifest_sha256=self.source_hash)
        for name, suffix in [('info', 'json'), ('episodes', 'jsonl'), ('tasks', 'jsonl')]:
            expected[name + '_sha256'] = sha256(Path(dataset_root) / 'meta' / f'{name}.{suffix}')
        for name in ['vae_model_sha256', 'encoding', 'temporal_sampling', 'latent_shape', 'latent_dtype']:
            expected[name] = base_manifest['metadata'][name]
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f'Dense VAE cache provenance differs: {name}')
        self.latent_shape = tuple(metadata['latent_shape'])
        if self.latent_shape != tuple(self.view_contract['latent_shape']) or metadata['latent_dtype'] != 'bfloat16':
            raise ValueError('Unsupported dense VAE cache layout')
        entries = m.get('episodes', {})
        if set(entries) != {str(row['episode_index']) for row in rows}:
            raise ValueError('Incomplete dense VAE episode inventory')
        for row in rows:
            ep, length, boundary = row['episode_index'], row['length'], row['exec_start_idx']
            offsets = entries[str(ep)].get('offsets', {})
            if set(offsets) != set(map(str, range(16))):
                raise ValueError(f'Incomplete dense VAE offset inventory: {ep}')
            for offset in range(16):
                entry = offsets[str(offset)]
                count = len(dense_execution_frames(length, boundary, offset)) - (offset != 0)
                source = 'dense_cache' if offset else 'base_cache'
                if (entry.get('source') != source or entry.get('frame_count') != count
                        or len(entry.get('sha256', '')) != 64):
                    raise ValueError(f'Invalid dense VAE entry: {ep}/{offset}')
                if offset:
                    self._path(entry['path'])
                elif entry.get('path') != base_manifest['episodes'][str(ep)]:
                    raise ValueError(f'Offset zero must reference the unchanged source cache: {ep}')
        self.manifest_sha256 = sha256(self.manifest_path)
        self._payloads = OrderedDict()
        self._verified = set()
        self._verified_base = set()

    def _path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Dense VAE entry must stay within its cache directory')
        return path

    def _load(self, row, offset):
        key = (row['episode_index'], offset)
        payload = self._payloads.pop(key, None)
        if payload is None:
            entry = self.manifest['episodes'][str(key[0])]['offsets'][str(offset)]
            path = self._path(entry['path'])
            if key not in self._verified:
                if sha256(path) != entry['sha256']:
                    raise ValueError(f'Dense VAE cache checksum mismatch: {key}')
                self._verified.add(key)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if self.view_layout != LEGACY_VIEW_LAYOUT and payload.get('view_contract') != self.view_contract:
                raise ValueError(f'Dense VAE payload view contract differs: {key}')
            frames = dense_execution_frames(row['length'], row['exec_start_idx'], offset)[1:]
            if (payload.get('schema') != DENSE_SCHEMA
                    or payload.get('episode_index') != key[0]
                    or payload.get('execution_start') != row['exec_start_idx']
                    or payload.get('offset') != offset
                    or payload.get('source_manifest_sha256') != self.source_hash
                    or payload['frame_indices'].tolist() != frames
                    or payload['latents'].shape != (len(frames), *self.latent_shape)
                    or payload['latents'].dtype != torch.bfloat16
                    or not torch.isfinite(payload['latents']).all()):
                raise ValueError(f'Dense VAE cache clock/shape mismatch: {key}')
        self._payloads[key] = payload
        while len(self._payloads) > 2:
            self._payloads.popitem(last=False)
        return payload

    def prefix(self, row, current, base):
        if not row['exec_start_idx'] <= current < row['length'] - 1:
            raise ValueError('Training start must have at least one future execution action')
        boundary = row['exec_start_idx']
        episode = row['episode_index']
        if episode not in self._verified_base:
            entry = self.manifest['episodes'][str(episode)]['offsets']['0']
            path = (self.base_root / entry['path']).resolve()
            if not path.is_relative_to(self.base_root) or sha256(path) != entry['sha256']:
                raise ValueError(f'Dense source VAE payload changed: {episode}')
            self._verified_base.add(episode)
        offset = execution_offset(current, boundary)
        frames = dense_prefix_frames(current, boundary)
        if offset == 0:
            stop = base['frame_indices'].tolist().index(current) + 1
            latents = base['latents'][:stop]
        else:
            data = self._load(row, offset)
            head = base['frame_indices'].tolist().index(boundary) + 1
            stop = data['frame_indices'].tolist().index(current) + 1
            latents = torch.cat((base['latents'][:head], data['latents'][:stop]), dim=0)
        if len(latents) != len(frames) or frames[-1] != current:
            raise ValueError('Dense visual prefix is not aligned with the current observation')
        return dict(latents=latents, frame_indices=torch.tensor(frames, dtype=torch.long),
                    execution_start=boundary)

    def contract(self):
        result = dict(schema=DENSE_SCHEMA, manifest_sha256=self.manifest_sha256,
                    sample_stride=1, prediction_start='observation_t_to_actions_t_plus_1',
                    history_clock='true_phase_start_then_current_aligned_stride16',
                    augmentation='execution_clock_offset_0_to_15',
                    inference_clock_offset=0, execution_horizon=16)
        if self.view_layout != LEGACY_VIEW_LAYOUT:
            result['view_contract'] = self.view_contract
        return result
