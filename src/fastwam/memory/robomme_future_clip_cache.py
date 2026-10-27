"""Exact, opportunistic cache of complete RoboMME future-clip VAE outputs.

This is an execution optimization, deliberately absent from model/data recipes.
Each episode is one sparse, indexed file. A SHA256 commit record protects every
row; interrupted/concurrent writes are misses, never partial tensor hits.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)
SCHEMA = 'robomme_complete_future_clip_v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def execution_identity(device, dtype):
    device = torch.device(device)
    cuda = device.type == 'cuda'
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), dtype=str(dtype),
                gpu=torch.cuda.get_device_name(device) if cuda else 'cpu',
                capability=list(torch.cuda.get_device_capability(device)) if cuda else [],
                autocast=torch.is_autocast_enabled(device.type),
                autocast_dtype=str(torch.get_autocast_dtype(device.type)),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic=torch.are_deterministic_algorithms_enabled())


@contextmanager
def preserve_rng(device):
    cpu, numpy, python = torch.get_rng_state(), np.random.get_state(), random.getstate()
    device = torch.device(device)
    cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        np.random.set_state(numpy)
        random.setstate(python)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)


class FutureClipCache:
    """Lossless fixed-row storage; no pickle, dtype conversion or lossy codec."""
    def __init__(self, root, contract):
        self.contract = json.loads(_json(contract))
        self.namespace = hashlib.sha256(_json(contract).encode()).hexdigest()
        self.root = Path(root) / self.namespace
        self.shape = tuple(contract['latent_shape'])
        self.dtype = getattr(torch, contract['dtype'].removeprefix('torch.'))
        self.nbytes = math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()
        self.record_size = 32 + self.nbytes
        self.rows = {int(r['episode_index']): r for r in contract['episodes']}
        self._warned = set()
        self.disabled = False
        self.root.mkdir(parents=True, exist_ok=True)
        import fcntl
        with (self.root / 'manifest.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'manifest.json'
            if path.exists():
                if json.loads(path.read_text()) != self.contract:
                    raise ValueError('Future clip cache namespace collision')
            else:
                temporary = self.root / f'manifest.{os.getpid()}.tmp'
                temporary.write_text(_json(self.contract) + '\n')
                os.replace(temporary, path)

    def _warning(self, operation, exc):
        if operation not in self._warned:
            logger.warning('Future clip cache %s failed; using online VAE: %s', operation, exc)
            self._warned.add(operation)

    def _position(self, episode, start):
        row = self.rows[int(episode)]
        first, stop = int(row['exec_start_idx']), int(row['length']) - 1
        if not first <= int(start) < stop:
            raise ValueError('Future clip cache start is outside the execution samples')
        return ((int(start) - first) * self.record_size,
                (stop - first) * self.record_size,
                self.root / f'episode_{int(episode):06d}.bin')

    def _digest(self, episode, start, payload):
        key = f'{self.namespace}:{int(episode)}:{int(start)}:'.encode()
        return hashlib.sha256(key + payload).digest()

    def _read_bytes(self, fd, offset, episode, start):
        record = os.pread(fd, self.record_size, offset)
        if len(record) != self.record_size or record[:32] == bytes(32):
            return None
        payload = record[32:]
        return payload if self._digest(episode, start, payload) == record[:32] else None

    def get(self, episode, start):
        try:
            if self.disabled or (self.root / 'disabled').exists():
                return None
            offset, _, path = self._position(episode, start)
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError:
                return None
            try:
                payload = self._read_bytes(fd, offset, episode, start)
            finally:
                os.close(fd)
            if payload is None:
                return None
            return torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(self.dtype).reshape(self.shape)
        except OSError as exc:
            self._warning('read', exc)
            return None

    def put(self, episode, start, tensor):
        # Callers pass the actual encoder output, never a cast to a cache dtype.
        if tuple(tensor.shape) != self.shape or tensor.dtype != self.dtype:
            raise ValueError('Future clip cache must retain the exact VAE output shape/dtype')
        if tensor.device.type != 'cpu':
            raise ValueError('Future clip cache writes require a detached CPU tensor')
        payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        try:
            if self.disabled or (self.root / 'disabled').exists():
                return False
            import fcntl
            offset, size, path = self._position(episode, start)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
            try:
                # Initialization and writes are bounded; OS releases locks on exit.
                fcntl.flock(fd, fcntl.LOCK_EX)
                current_size = os.fstat(fd).st_size
                if current_size == 0:
                    os.ftruncate(fd, size)
                elif current_size != size:
                    self._warning('size', ValueError(str(path)))
                    return False
                previous = self._read_bytes(fd, offset, episode, start)
                if previous is not None:
                    if previous != payload:
                        self.disabled = True
                        (self.root / 'disabled').write_text('Non-identical repeated VAE result\n')
                        logger.error('Future clip cache disabled: repeated encoder outputs differ bitwise')
                    return previous == payload
                os.pwrite(fd, bytes(32), offset)
                # pwrite may legally be short, so finish the full payload before commit.
                written = 0
                while written < len(payload):
                    count = os.pwrite(fd, payload[written:], offset + 32 + written)
                    if count <= 0:
                        raise OSError('Short future clip cache write')
                    written += count
                if os.pwrite(fd, self._digest(episode, start, payload), offset) != 32:
                    raise OSError('Short future clip cache commit')
                return True
            finally:
                os.close(fd)
        except OSError as exc:
            self._warning('write', exc)
            return False


def collate_future_clips(samples):
    """Keep B-sized tensors first for Accelerate; transfer RGB only for misses."""
    exclusions = {'video', '_future_clip_latents', '_future_clip_namespace', '_future_clip_video_shape'}
    plain = [{k: v for k, v in sample.items() if k not in exclusions} for sample in samples]
    batch = default_collate(plain)
    namespace = samples[0]['_future_clip_namespace']
    shape = samples[0]['_future_clip_video_shape']
    if any(s['_future_clip_namespace'] != namespace or s['_future_clip_video_shape'] != shape for s in samples):
        raise ValueError('Mixed future clip cache contracts in one batch')
    misses = tuple(i for i, sample in enumerate(samples) if sample['video'].numel())
    batch['video'] = (torch.stack([samples[i]['video'] for i in misses]) if misses
                      else torch.empty((0, *shape), dtype=torch.float32))
    batch['future_clip_latents'] = torch.stack([s['_future_clip_latents'] for s in samples])
    batch['future_clip_misses'] = misses
    batch['future_clip_keys'] = tuple((int(s['episode_index']), int(s['frame_index'])) for s in samples)
    batch['future_clip_video_shape'] = (len(samples), *shape)
    batch['future_clip_namespace'] = namespace
    return batch


def encode_cached_future_clips(model, sample, tiled=False):
    cache = getattr(model, '_future_clip_cache', None)
    if cache is None or sample['future_clip_namespace'] != cache.namespace:
        raise ValueError('Future clip batch does not match the model execution cache')
    if (tiled or execution_identity(model.device, model.torch_dtype) != cache.contract['execution']
            or cache.disabled or (cache.root / 'disabled').exists()):
        # A training wrapper can change numerical backend flags after loader
        # creation. Reconstruct the original clips rather than use another
        # producer's bytes or abort an otherwise valid training job.
        from fastwam.datasets.robomme import episode_video
        cache._warning('context', ValueError('producer context changed or cache disabled'))
        clips = []
        for episode, start in sample['future_clip_keys']:
            row = cache.rows[int(episode)]
            indices = [min(int(start) + offset, row['length'] - 1)
                       for offset in cache.contract['offsets']]
            clips.append(episode_video(Path(cache.contract['data_root']), cache.contract['source_info'],
                                       row, indices, view_layout=cache.contract['view_contract']['view_layout']))
        video = torch.stack(clips).to(device=model.device, dtype=model.torch_dtype)
        return model._encode_video_latents(video, tiled=tiled)
    latents = sample['future_clip_latents']
    if latents.dtype != cache.dtype or tuple(latents.shape[1:]) != cache.shape:
        raise ValueError('Invalid full future clip latent tensor')
    misses = sample['future_clip_misses']
    video = sample['video']
    if video.shape[0] != len(misses) or len(latents) != len(sample['future_clip_keys']):
        raise ValueError('Future clip cache batch alignment mismatch')
    result = latents.to(device=model.device, non_blocking=True).clone()
    if misses:
        encoded = model._encode_video_latents(video.to(device=model.device, dtype=model.torch_dtype,
                                                       non_blocking=True), tiled=False)
        if encoded.dtype != cache.dtype or tuple(encoded.shape[1:]) != cache.shape:
            raise ValueError('Online VAE output differs from the bound cache shape/dtype')
        for offset, slot in enumerate(misses):
            result[slot] = encoded[offset]
        cpu = encoded.detach().cpu()
        for offset, slot in enumerate(misses):
            cache.put(*sample['future_clip_keys'][slot], cpu[offset])
        if cache.disabled or (cache.root / 'disabled').exists():
            # A concurrent producer conflict invalidates hits in this very
            # batch too; reconstruct the original complete batch immediately.
            return encode_cached_future_clips(model, sample, tiled=tiled)
    return result


def configure_future_clip_cache(model, dataset, cfg):
    """Called under the real Accelerator autocast before workers are created."""
    root = os.environ.get('ROBOMME_FUTURE_CLIP_CACHE_ROOT')
    if not root or getattr(dataset, '_future_clip_cache_configured', False):
        return
    from fastwam.datasets.robomme import RoboMMEHeliosDataset, episode_video
    from fastwam.datasets.robodojo import episode_paths
    if not isinstance(dataset, RoboMMEHeliosDataset) or dataset.prediction_horizon != 32:
        return
    dataset._future_clip_cache_configured = True
    import cv2
    import torchvision
    from fastwam.models.wan22 import wan_video_vae
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.datasets import robomme
    from fastwam.memory import robomme_views
    episodes = []
    for row in dataset.rows:
        _, paths = episode_paths(dataset.root, dataset.info, row)
        sources = []
        for path in paths[:2]:
            path = Path(path)
            stat = path.stat()
            sources.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        episodes.append(dict(episode_index=int(row['episode_index']), length=int(row['length']),
                             exec_start_idx=int(row['exec_start_idx']), videos=sources))
    dtype = model.torch_dtype
    latent_shape = [dataset.latent_shape[0], 3, *dataset.latent_shape[-2:]]
    contract = dict(schema=SCHEMA, dtype=str(dtype), latent_shape=latent_shape,
                    data_root=str(dataset.root.resolve()), source_info=dataset.info,
                    vae_sha256=str(cfg.robomme_provenance.vae_sha256),
                    metadata_sha256={name: _sha(dataset.root / 'meta' / name)
                                     for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl')},
                    view_contract=dataset.view_contract, offsets=list(range(0, 33, 4)),
                    tail_padding='clamp_to_last_frame', encoding='whole_clip_reset_vae_mu',
                    execution=execution_identity(model.device, dtype), episodes=episodes,
                    opencv=cv2.__version__, torchvision=torchvision.__version__,
                    vae_code_sha256=_sha(wan_video_vae.__file__),
                    views_code_sha256=_sha(robomme_views.__file__),
                    video_decoder_sha256=hashlib.sha256(inspect.getsource(robomme.episode_video).encode()).hexdigest(),
                    encode_helper_sha256=hashlib.sha256(inspect.getsource(FastWAM._encode_video_latents).encode()).hexdigest())
    try:
        cache = FutureClipCache(root, contract)
        if (cache.root / 'disabled').exists():
            logger.warning('Future clip cache was marked incompatible; keeping online VAE: %s', cache.root)
            return
        # Verify on this producer, under its actual autocast and without touching
        # the sampler or leaving RNG changes. Never trust a cross-runtime cache.
        row, start = dataset.samples[0]
        frames = [min(start + offset, row['length'] - 1) for offset in contract['offsets']]
        with preserve_rng(model.device), torch.no_grad():
            video = episode_video(dataset.root, dataset.info, row, frames,
                                  view_layout=dataset.view_layout).unsqueeze(0)
            video = video.to(device=model.device, dtype=dtype)
            cpu_rng = torch.get_rng_state()
            cuda_rng = (torch.cuda.get_rng_state(model.device)
                        if torch.device(model.device).type == 'cuda' else None)
            first = model._encode_video_latents(video, tiled=False).detach().cpu()
            second = model._encode_video_latents(video, tiled=False).detach().cpu()
            rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
            if cuda_rng is not None:
                rng_unchanged = rng_unchanged and torch.equal(cuda_rng, torch.cuda.get_rng_state(model.device))
        if (first.dtype != dtype or tuple(first.shape) != (1, *latent_shape)
                or not torch.equal(first, second) or not rng_unchanged):
            logger.warning('Future clip cache disabled: producer repeatability/shape/dtype check failed')
            return
        existing = cache.get(row['episode_index'], start)
        if existing is not None and not torch.equal(first[0], existing):
            (cache.root / 'disabled').write_text('Cached producer probe differs bitwise\n')
            logger.warning('Future clip cache disabled: cached producer probe differs bitwise')
            return
        cache.put(row['episode_index'], start, first[0])
        dataset.future_clip_cache = cache
        dataset.collate_fn = collate_future_clips
        model._future_clip_cache = cache
        logger.info('Future clip cache enabled: %s; complete 9-frame clip, exact %s outputs, lazy fill',
                    cache.root, dtype)
    except (OSError, ValueError) as exc:
        logger.warning('Future clip cache unavailable; keeping online VAE: %s', exc)
