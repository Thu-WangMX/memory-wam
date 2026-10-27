import importlib.util
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).parents[1] / 'src/fastwam/memory/robomme_future_clip_cache.py'
spec = importlib.util.spec_from_file_location('future_clip_cache_under_test', SOURCE)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)


def make_cache(tmp_path, dtype=torch.bfloat16):
    return cache_module.FutureClipCache(tmp_path, dict(
        schema=cache_module.SCHEMA, dtype=str(dtype), latent_shape=[2, 3, 2, 2],
        execution=cache_module.execution_identity('cpu', dtype),
        episodes=[dict(episode_index=0, exec_start_idx=3, length=22)]))


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_raw_dtype_bits_and_every_start_roundtrip(tmp_path, dtype):
    cache = make_cache(tmp_path, dtype)
    for start in range(3, 21):
        assert cache.get(0, start) is None
        value = (torch.arange(24).reshape(cache.shape) + start).to(dtype)
        assert cache.put(0, start, value)
        actual = cache.get(0, start)
        assert actual.dtype == dtype
        assert torch.equal(actual.view(torch.uint8), value.view(torch.uint8))
    assert len(list(cache.root.glob('episode_*.bin'))) == 1
    with pytest.raises(ValueError):
        cache.get(0, 21)


def test_corrupt_and_uncommitted_rows_are_misses(tmp_path):
    cache = make_cache(tmp_path)
    value = torch.arange(24, dtype=torch.bfloat16).reshape(cache.shape)
    cache.put(0, 3, value)
    path = cache.root / 'episode_000000.bin'
    with path.open('r+b') as stream:
        stream.seek(35)
        stream.write(b'\xff')
    assert cache.get(0, 3) is None
    assert cache.put(0, 3, value)
    with path.open('r+b') as stream:
        stream.seek(0)
        stream.write(bytes(32))
    assert cache.get(0, 3) is None
    assert cache.get(0, 4) is None


def test_conflicting_valid_row_disables_namespace(tmp_path):
    cache = make_cache(tmp_path)
    value = torch.ones(cache.shape, dtype=cache.dtype)
    assert cache.put(0, 3, value)
    assert not cache.put(0, 3, value + 1)
    assert (cache.root / 'disabled').exists()
    assert cache.get(0, 3) is None


def test_write_error_is_a_miss_not_a_training_error(tmp_path, monkeypatch):
    cache = make_cache(tmp_path)
    def fail(*args):
        raise OSError('simulated disk full')
    monkeypatch.setattr(os, 'pwrite', fail)
    assert not cache.put(0, 3, torch.ones(cache.shape, dtype=cache.dtype))
    assert cache.get(0, 3) is None


def test_no_silent_dtype_conversion(tmp_path):
    cache = make_cache(tmp_path)
    with pytest.raises(ValueError, match='dtype'):
        cache.put(0, 3, torch.ones(cache.shape, dtype=torch.float32))


@pytest.mark.parametrize('hits', [(), (0,), (1,), (0, 1, 2)])
def test_mixed_batches_keep_order_and_skip_only_cached_encoder_calls(tmp_path, hits):
    cache = make_cache(tmp_path)
    videos = [torch.full((3, 9, 4, 4), float(i + 1)) for i in range(3)]
    calls = []
    def encode(video, tiled=False):
        calls.append(len(video))
        return video[:, :1, :1, :1, :1].expand(-1, *cache.shape).clone().to(cache.dtype)
    expected = encode(torch.stack(videos))
    calls.clear()
    samples = []
    for i in range(3):
        samples.append(dict(video=torch.empty(0) if i in hits else videos[i],
                            action=torch.ones(32, 8), episode_index=0, frame_index=i + 3,
                            _future_clip_namespace=cache.namespace,
                            _future_clip_video_shape=tuple(videos[i].shape),
                            _future_clip_latents=expected[i] if i in hits else torch.zeros_like(expected[i])))
    batch = cache_module.collate_future_clips(samples)
    assert next(iter(batch.values())).shape[0] == 3
    assert batch['video'].shape[0] == 3 - len(hits)
    model = SimpleNamespace(_future_clip_cache=cache, device='cpu', torch_dtype=cache.dtype,
                            _encode_video_latents=encode)
    rng = torch.get_rng_state().clone()
    actual = cache_module.encode_cached_future_clips(model, batch)
    assert torch.equal(actual, expected)
    assert torch.equal(rng, torch.get_rng_state())
    assert calls == ([3 - len(hits)] if len(hits) < 3 else [])
    # The same downstream graph sees the same values and parameter gradients.
    weight1 = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(cache.shape))
    weight2 = torch.nn.Parameter(weight1.detach().clone())
    loss1 = (expected.float() * weight1).square().mean()
    loss2 = (actual.float() * weight2).square().mean()
    loss1.backward(); loss2.backward()
    assert torch.equal(loss1, loss2)
    assert torch.equal(weight1.grad, weight2.grad)


def test_preserve_rng_does_not_change_training_sequence():
    torch.manual_seed(45); np.random.seed(45); random.seed(45)
    cpu = torch.get_rng_state().clone()
    numpy = np.random.get_state()
    python = random.getstate()
    with cache_module.preserve_rng('cpu'):
        torch.randn(20); np.random.rand(20); random.random()
    assert torch.equal(cpu, torch.get_rng_state())
    assert np.array_equal(numpy[1], np.random.get_state()[1])
    assert python == random.getstate()


def test_runtime_identity_separates_producers(tmp_path):
    first = make_cache(tmp_path)
    contract = dict(first.contract)
    contract['execution'] = dict(contract['execution'], gpu='another GPU')
    second = cache_module.FutureClipCache(tmp_path, contract)
    assert first.namespace != second.namespace


def test_concurrent_writers_publish_only_complete_rows(tmp_path):
    import multiprocessing
    cache = make_cache(tmp_path)
    value = torch.arange(24, dtype=cache.dtype).reshape(cache.shape)
    def worker(offset):
        for _ in range(3):
            assert cache.put(0, 3, value)
            assert cache.put(0, 4 + offset, value + offset)
    context = multiprocessing.get_context('fork')
    workers = [context.Process(target=worker, args=(i,)) for i in range(4)]
    for worker_process in workers:
        worker_process.start()
    for worker_process in workers:
        worker_process.join(timeout=15)
        assert worker_process.exitcode == 0
    assert torch.equal(cache.get(0, 3), value)
    for i in range(4):
        assert torch.equal(cache.get(0, 4 + i), value + i)
