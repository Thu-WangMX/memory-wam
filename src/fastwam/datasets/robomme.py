"""Read-only adapter for Kevin's complete 16-task RoboMME conversion.

Only observed VAE latents are reused. Event segments are deliberately irrelevant
to Anchor-Helios. Native 8D supervision excludes the conversion's zero padding.
"""
from collections import OrderedDict, Counter
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from fastwam.datasets.robodojo import (
    compose_rgb, read_jsonl, episode_paths, load_context, text_filename, sha256, PROMPT,
)
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from fastwam.memory.helios_history import fixed_history
from fastwam.memory.robomme_clock import (CACHE_SCHEMA, HORIZON, decision_frames,
    LEGACY_ANCHOR_MODE, select_robomme_anchors, validate_anchor_mode)
from fastwam.memory.robomme_views import (
    LEGACY_VIEW_LAYOUT, mosaic, view_spec, base_cache_schema, validate_view_metadata)

TASKS = sorted('BinFill ButtonUnmask ButtonUnmaskSwap InsertPeg MoveCube PatternLock '
               'PickHighlight PickXtimes RouteStick StopCube SwingXtimes VideoPlaceButton '
               'VideoPlaceOrder VideoRepick VideoUnmask VideoUnmaskSwap'.split())
logger = logging.getLogger(__name__)


def episode_video(root, info, row, indices, *, seek=True, view_layout=LEGACY_VIEW_LAYOUT):
    """Read exact frame indices with the original OpenCV/RGB/mosaic contract.

    Following starwam-robodojo's seek-before-window strategy, let FFmpeg seek
    to a preceding keyframe and decode up to the first requested frame. Keep
    integer frame addressing for these per-episode videos (not nearest-PTS
    matching for concatenated shards). ``seek=False`` is the parity oracle.
    """
    wanted, found = set(indices), {}
    if not wanted or min(wanted) < 0:
        raise ValueError('Expected nonnegative frame indices')
    _, paths = episode_paths(root, info, row)
    readers = [cv2.VideoCapture(str(path)) for path in paths[:2]]
    try:
        if not all(reader.isOpened() for reader in readers):
            raise ValueError('Failed to open episode videos')
        first = min(wanted) if seek else 0
        if first and not all(reader.set(cv2.CAP_PROP_POS_FRAMES, first)
                             and abs(reader.get(cv2.CAP_PROP_POS_FRAMES) - first) < 0.5
                             for reader in readers):
            # Reopening is essential: a failed seek may have moved one camera.
            logger.warning('Video seek unavailable for episode %s; decoding from frame zero', row['episode_index'])
            for reader in readers:
                reader.release()
            readers = [cv2.VideoCapture(str(path)) for path in paths[:2]]
            if not all(reader.isOpened() for reader in readers):
                raise ValueError('Failed to reopen episode videos after seek')
            first = 0
        for i in range(first, max(indices) + 1):
            frames = [reader.read() for reader in readers]
            if not all(ok for ok, _ in frames):
                raise ValueError(f'Episode video ended at frame {i}')
            if i in wanted:
                found[i] = mosaic(*(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for _, frame in frames),
                                  view_layout=view_layout)
    finally:
        for reader in readers:
            reader.release()
    return torch.stack([found[i] for i in indices], dim=1)


def normalizers(stats_path):
    raw = json.loads(Path(stats_path).read_text())
    result = {}
    for field in ('action', 'state'):
        values = {k.removeprefix('global_'): torch.tensor(v, dtype=torch.float32)
                  for k, v in raw[field]['default'].items() if k.startswith('global_')}
        if any(v.shape != (8,) or not torch.isfinite(v).all() for v in values.values()):
            raise ValueError('Expected finite native 8D statistics')
        if (values['std'] <= 0).any():
            raise ValueError('Nonpositive normalization standard deviation')
        result[field] = SingleFieldLinearNormalizer(values, mode='z-score')
    return result


class RoboMMEHeliosDataset(torch.utils.data.Dataset):
    # Augmentation/noise lives in the model, not worker-local random state.
    deterministic_samples = True

    def __init__(self, root, prepared_root, full_kv_cache_path, tasks=None,
                 history_sizes=(16, 2, 1), anchor_frames=2, episode_indices=None,
                 task_balanced=True, pretrained_norm_stats=None, prediction_horizon=16,
                 anchor_mode=LEGACY_ANCHOR_MODE, sample_stride=16,
                 dense_latents_path=None, view_layout=LEGACY_VIEW_LAYOUT):
        if prediction_horizon not in (16, 32):
            raise ValueError('Prediction horizon must be 16 or 32; execution stride stays 16')
        self.prediction_horizon = int(prediction_horizon)
        if isinstance(sample_stride, bool) or sample_stride not in (1, 16):
            raise ValueError('sample_stride must be 1 (every execution frame) or 16 (legacy)')
        self.sample_stride = int(sample_stride)
        self.view_layout = view_layout
        self.view_contract = view_spec(view_layout)
        self.latent_shape = tuple(self.view_contract['latent_shape'])
        self.anchor_mode = validate_anchor_mode(anchor_mode)
        self.root, self.prepared_root = Path(root), Path(prepared_root)
        self.cache_root = Path(full_kv_cache_path)
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.rows = read_jsonl(self.root / 'meta/episodes.jsonl')
        self.instructions = {int(r['task_index']): r['task']
                             for r in read_jsonl(self.root / 'meta/tasks.jsonl')}
        if (self.info['total_episodes'], self.info['fps'], self.info['robot_type']) != (1600, 10, 'panda'):
            raise ValueError('Expected audited full16 RoboMME conversion')
        if ([r['episode_index'] for r in self.rows] != list(range(1600))
                or Counter(r['semantic_task'] for r in self.rows) != Counter({t: 100 for t in TASKS})):
            raise ValueError('Expected 100 episodes for each of 16 semantic tasks')
        self.history_sizes, self.anchor_frames = tuple(history_sizes), int(anchor_frames)
        if self.history_sizes not in ((16, 2, 1), (72, 2, 1), (96, 2, 1)) or self.anchor_frames not in (0, 2):
            raise ValueError('Expected Anchor-Helios history=(16,2,1), (72,2,1) or (96,2,1), anchors=0 or 2')
        self.tasks = TASKS if tasks is None or list(tasks) == ['all'] else sorted(tasks)
        if not self.tasks or not set(self.tasks) <= set(TASKS):
            raise ValueError('Unknown/empty semantic tasks')
        self.manifest = json.loads((self.cache_root / 'manifest.json').read_text())
        validate_view_metadata(self.manifest['metadata'], self.view_layout)
        expected = dict(schema=base_cache_schema(self.view_layout), complete=True, episode_count=1600,
                        semantic_tasks=TASKS, vae_phase_reset=True, replan_stride=16,
                        temporal_sampling='phase_local_endpoint_linspace4',
                        encoding='separate_demo_execution_causal_vae',
                        color_contract='simulator_rgb_preserved',
                        latent_shape=list(self.latent_shape), latent_dtype='bfloat16',
                        mosaic=self.view_contract['mosaic'])
        for key, value in expected.items():
            if self.manifest['metadata'].get(key) != value:
                raise ValueError(f'Incompatible latent cache: {key}')
        if set(self.manifest['episodes']) != set(map(str, range(1600))):
            raise ValueError('Incomplete latent cache')
        provenance = json.loads((self.prepared_root / 'provenance.json').read_text())
        if self.view_layout != LEGACY_VIEW_LAYOUT and provenance.get('view_contract') != self.view_contract:
            raise ValueError('Prepared assets must be rebound to the selected view layout')
        for key, path in [('info', self.root/'meta/info.json'),
                          ('episodes', self.root/'meta/episodes.jsonl'),
                          ('tasks', self.root/'meta/tasks.jsonl'),
                          ('latents', self.cache_root/'manifest.json')]:
            if provenance[key+'_sha256'] != sha256(path):
                raise ValueError(f'Dataset/cache provenance changed: {key}')
        stats = Path(pretrained_norm_stats or self.prepared_root/'dataset_stats.json')
        if provenance['stats_sha256'] != sha256(stats):
            raise ValueError('Normalization provenance changed')
        self.norm = normalizers(stats)
        self.dense_cache = None
        if self.sample_stride == 1:
            if not dense_latents_path:
                raise ValueError('Every-frame sampling requires a complete dense_latents_path; '
                                 'prepare the 15 additional causal VAE clock offsets first')
            from .robomme_dense_cache import DenseRoboMMECache
            self.dense_cache = DenseRoboMMECache(
                dense_latents_path, self.root, self.cache_root, self.manifest, self.rows)
        keep = set(range(1600) if episode_indices is None else episode_indices)
        if not keep <= set(range(1600)):
            raise ValueError('Invalid episode selection')
        self.episodes = [r for r in self.rows if r['episode_index'] in keep and r['semantic_task'] in self.tasks]
        if not self.episodes:
            raise ValueError('No selected episodes')
        self.samples, self.task_episodes, self.episode_samples = [], {}, {}
        for row in self.episodes:
            ep, n, start = row['episode_index'], row['length'], row['exec_start_idx']
            decision_frames(n, start)
            indices = []
            for frame in range(start, n-1, self.sample_stride):
                indices.append(len(self.samples))
                self.samples.append((row, frame))
            if not indices:
                raise ValueError(f'Episode {ep} has no execution action')
            self.episode_samples[ep] = indices
            self.task_episodes.setdefault(row['semantic_task'], []).append(ep)
        self.task_balanced = task_balanced
        self._tables, self._latents, self._contexts = OrderedDict(), OrderedDict(), {}
        self._verified_view_payloads = set()

    def __len__(self):
        return len(self.samples)

    def sample_epoch_indices(self, seed):
        rng = np.random.default_rng(seed)
        if not self.task_balanced:
            return rng.permutation(len(self)).tolist()
        # np.choice(list) reconverts the entire list on every draw. Dense starts
        # make those lists 16x larger; cache arrays without changing RNG ordering.
        if not hasattr(self, '_sampling_task_episodes'):
            self._sampling_task_episodes = {k: np.asarray(v, dtype=np.int64)
                                           for k, v in self.task_episodes.items()}
            self._sampling_episode_samples = {k: np.asarray(v, dtype=np.int64)
                                             for k, v in self.episode_samples.items()}
        indices = []
        while len(indices) < len(self):
            for task in rng.permutation(sorted(self.task_episodes)):
                ep = int(rng.choice(self._sampling_task_episodes[task]))
                indices.append(int(rng.choice(self._sampling_episode_samples[ep])))
        return indices[:len(self)]

    def _table(self, row):
        ep = row['episode_index']
        table = self._tables.pop(ep, None)
        if table is None:
            path, _ = episode_paths(self.root, self.info, row)
            raw = pq.read_table(path, columns=['action', 'observation.state', 'task_index',
                                               'frame_index', 'episode_index'])
            table = {k: torch.as_tensor(np.asarray(raw[k].to_pylist())) for k in raw.column_names}
            n = row['length']
            if (not torch.equal(table['frame_index'].reshape(-1), torch.arange(n))
                    or not (table['episode_index'] == ep).all()):
                raise ValueError(f'Frame alignment mismatch: {ep}')
            for key in ('action', 'observation.state'):
                v = table[key]
                if v.shape != (n, 14) or not torch.isfinite(v).all() or (v[:, 8:] != 0).any():
                    raise ValueError(f'Invalid padded 8D source {ep}: {key}')
                table[key] = v[:, :8].float()
        self._tables[ep] = table
        while len(self._tables) > 2:
            self._tables.popitem(last=False)
        return table

    def load_latents(self, row):
        ep = row['episode_index']
        payload = self._latents.pop(ep, None)
        if payload is None:
            path = (self.cache_root / self.manifest['episodes'][str(ep)]).resolve()
            if not path.is_relative_to(self.cache_root.resolve()):
                raise ValueError('Base VAE payload escapes its cache root')
            if self.view_layout != LEGACY_VIEW_LAYOUT and ep not in self._verified_view_payloads:
                expected_hash = self.manifest.get('payload_sha256', {}).get(str(ep))
                if not expected_hash or sha256(path) != expected_hash:
                    raise ValueError(f'View cache payload checksum mismatch: {ep}')
                self._verified_view_payloads.add(ep)
            payload = torch.load(path,
                                 map_location='cpu', weights_only=True)
            frames = decision_frames(row['length'], row['exec_start_idx'])
            if (payload['frame_indices'].tolist() != frames
                    or payload['execution_start'] != row['exec_start_idx']
                    or payload['latents'].shape != (len(frames), *self.latent_shape)
                    or payload['latents'].dtype != torch.bfloat16
                    or not torch.isfinite(payload['latents']).all()):
                raise ValueError(f'Cache temporal/shape mismatch: {ep}')
            if self.view_layout != LEGACY_VIEW_LAYOUT and (
                    payload.get('view_contract') != self.view_contract
                    or payload.get('schema') != base_cache_schema(self.view_layout)
                    or payload.get('source_manifest_sha256') != self.manifest['metadata'].get('source_manifest_sha256')):
                raise ValueError(f'Base cache layout/provenance mismatch: {ep}')
        self._latents[ep] = payload
        while len(self._latents) > 2:
            self._latents.popitem(last=False)
        return payload

    def observed_prefix(self, row, current):
        """Causal visual prefix ending EXACTLY at the sample's observation frame.

        Dense samples use one of 16 phase-offset streams. Each retains the true
        execution start, then follows a 16-frame grid ending at ``current``.
        Offset zero is the original cache, byte for byte. Never floor or ceil
        an unavailable current observation in the old cache.
        """
        payload = self.load_latents(row)
        dense = getattr(self, 'dense_cache', None)
        if dense is not None:
            return dense.prefix(row, current, payload)
        frames = payload['frame_indices'].tolist()
        if current not in frames:
            raise ValueError('Current frame is absent from the legacy VAE cache')
        stop = frames.index(current) + 1
        return dict(latents=payload['latents'][:stop],
                    frame_indices=payload['frame_indices'][:stop],
                    execution_start=row['exec_start_idx'])

    def __getitem__(self, index):
        row, start = self.samples[index]
        n, ep = row['length'], row['episode_index']
        table = self._table(row)
        # Row t contains post-action state: predict action[t+1] from state[t].
        ai = torch.arange(start+1, start+self.prediction_horizon+1)
        si = torch.arange(start, start+self.prediction_horizon)
        vi = torch.arange(start, start+self.prediction_horizon+1, 4)
        instruction = self.instructions[int(table['task_index'][start])]
        if instruction not in self._contexts:
            self._contexts[instruction] = load_context(self.prepared_root/'text_cache', instruction)
        context, mask = self._contexts[instruction]
        payload = self.observed_prefix(row, start)
        decision = len(payload['frame_indices']) - 1
        prefix = payload['latents']
        anchors, av = select_robomme_anchors(
            prefix, payload['frame_indices'][:decision+1].tolist(), row['exec_start_idx'],
            self.anchor_frames, getattr(self, 'anchor_mode', LEGACY_ANCHOR_MODE))
        history, hv = fixed_history(prefix, sum(self.history_sizes))
        video_kwargs = (dict(view_layout=self.view_layout)
                        if getattr(self, 'view_layout', LEGACY_VIEW_LAYOUT) != LEGACY_VIEW_LAYOUT else {})
        future_cache = getattr(self, 'future_clip_cache', None)
        cached_clip = future_cache.get(ep, start) if future_cache is not None else None
        video = (episode_video(self.root, self.info, row, vi.clamp(max=n-1).tolist(), **video_kwargs)
                 if cached_clip is None else torch.empty(0, dtype=torch.float32))
        result = dict(video=video,
                    action=self.norm['action'].forward(table['action'][ai.clamp(max=n-1)]),
                    proprio=self.norm['state'].forward(table['observation.state'][si.clamp(max=n-1)]),
                    action_is_pad=ai >= n, image_is_pad=vi >= n,
                    context=context.clone(), context_mask=mask.clone(), prompt=PROMPT.format(task=instruction),
                    history_latents=history, history_valid=hv, anchor_latents=anchors, anchor_valid=av,
                    episode_index=ep, frame_index=start, decision_index=decision, semantic_task=row['semantic_task'])
        if future_cache is not None:
            result.update(
                _future_clip_latents=(cached_clip if cached_clip is not None
                                      else torch.zeros(future_cache.shape, dtype=future_cache.dtype)),
                _future_clip_namespace=future_cache.namespace,
                _future_clip_video_shape=(3, len(vi), self.view_contract['height'], self.view_contract['width']))
        return result
