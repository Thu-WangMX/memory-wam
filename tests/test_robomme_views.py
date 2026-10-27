"""View geometry, legacy byte parity, and phase/cache identity contracts."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torchvision.transforms.functional import resize

from fastwam.datasets.robodojo import compose_rgb, sha256
from fastwam.datasets.robomme import RoboMMEHeliosDataset
from fastwam.datasets.robomme_dense_cache import DenseRoboMMECache
from fastwam.memory.robomme_clock import decision_frames
from fastwam.memory.robomme_views import (
    LEGACY_VIEW_LAYOUT, TWO_VIEW_HORIZONTAL, base_cache_schema, mosaic,
    validate_view_metadata, view_spec,
)
from scripts.prepare_robomme_two_view_base import (
    encode_base_episode, metadata_for_source, validate_payload,
)
from scripts.prepare_robomme_dense_latents import validate_dense_payload
from test_robomme_dense_sampling import constructor_assets, dense_cache_assets


def test_legacy_rgb_composition_is_bitwise_unchanged():
    rng = np.random.default_rng(9)
    front = rng.integers(0, 256, (231, 317, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (93, 141, 3), dtype=np.uint8)
    cameras = [torch.from_numpy(value.copy()).permute(2, 0, 1)[None]
               for value in (front, wrist, wrist)]
    assert torch.equal(mosaic(front, wrist), compose_rgb(cameras)[0])


def test_two_view_uses_each_camera_once_with_shared_resize_and_normalization():
    rng = np.random.default_rng(17)
    front = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (121, 177, 3), dtype=np.uint8)
    output = mosaic(front, wrist, TWO_VIEW_HORIZONTAL)
    expected = [resize(torch.from_numpy(value.copy()).permute(2, 0, 1).float()/255.,
                       [256, 256], antialias=True).mul(2).sub(1)
                for value in (front, wrist)]
    assert output.shape == (3, 256, 512) and output.dtype == torch.float32
    assert torch.equal(output[:, :, :256], expected[0])
    assert torch.equal(output[:, :, 256:], expected[1])
    assert output.min() >= -1 and output.max() <= 1
    assert view_spec(TWO_VIEW_HORIZONTAL)['latent_shape'] == [48, 1, 16, 32]


@pytest.mark.parametrize('bad', [np.zeros((3, 10, 10), dtype=np.uint8),
                               np.zeros((10, 10, 3), dtype=np.float32)])
def test_invalid_rgb_inputs_are_rejected(bad):
    with pytest.raises(ValueError, match='RGB uint8 HWC'):
        mosaic(bad, bad, TWO_VIEW_HORIZONTAL)


@pytest.mark.parametrize('boundary', [0, 1, 16, 17, 35, 49])
def test_new_base_keeps_demo_endpoints_and_resets_before_execution(boundary):
    row = dict(length=boundary+50, exec_start_idx=boundary)
    calls = []
    def fake_encode(_vae, frame, endpoints, _device):
        calls.append(list(endpoints))
        return torch.tensor(endpoints).reshape(-1, 1, 1, 1, 1)
    frames, values = encode_base_episode(None, None, row, 'cpu', encode_phase=fake_encode)
    assert frames == decision_frames(row['length'], boundary)
    assert values.flatten().tolist() == frames
    assert calls == ([list(filter(lambda x: x < boundary, frames))] if boundary else []) + [
        list(filter(lambda x: x >= boundary, frames))]
    assert calls[-1][0] == boundary
    if boundary:
        assert calls[0][0] == 0 and calls[0][-1] == boundary-1


def new_metadata():
    old = dict(schema=base_cache_schema(), latent_shape=[48, 1, 24, 20],
               mosaic=view_spec()['mosaic'])
    return metadata_for_source({'metadata': old}, 'a'*64, 'b'*64)


@pytest.mark.parametrize('field,value', [
    ('view_layout', LEGACY_VIEW_LAYOUT), ('view_layout_version', 3),
    ('latent_shape', [48, 1, 24, 20]), ('mosaic', view_spec()['mosaic']),
    ('view_contract', {}),
])
def test_new_layout_requires_exact_metadata(field, value):
    metadata = new_metadata()
    metadata[field] = value
    with pytest.raises(ValueError):
        validate_view_metadata(metadata, TWO_VIEW_HORIZONTAL)


def test_layout_inference_accepts_old_metadata_but_never_infers_new_from_shape_only():
    assert validate_view_metadata(dict(latent_shape=[48, 1, 24, 20],
                                       mosaic=view_spec()['mosaic'])) == view_spec()
    assert validate_view_metadata(new_metadata()) == view_spec(TWO_VIEW_HORIZONTAL)
    with pytest.raises(ValueError, match='latent_shape'):
        validate_view_metadata(dict(latent_shape=[48, 1, 16, 32]))


def test_dataset75_requires_reencoded_base_and_rebound_prepared(constructor_assets):
    kwargs, rows, source = constructor_assets
    with pytest.raises(ValueError, match='view_layout'):
        RoboMMEHeliosDataset(**kwargs, history_sizes=[72, 2, 1],
                             view_layout=TWO_VIEW_HORIZONTAL)
    metadata = metadata_for_source(source, sha256(kwargs['full_kv_cache_path']/'manifest.json'),
                                   sha256(kwargs['prepared_root']/'provenance.json'))
    source['metadata'] = metadata
    manifest_path = kwargs['full_kv_cache_path']/'manifest.json'
    manifest_path.write_text(json.dumps(source))
    provenance_path = kwargs['prepared_root']/'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['latents_sha256'] = sha256(manifest_path)
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='rebound'):
        RoboMMEHeliosDataset(**kwargs, history_sizes=[72, 2, 1], view_layout=TWO_VIEW_HORIZONTAL)
    provenance['view_contract'] = metadata['view_contract']
    provenance_path.write_text(json.dumps(provenance))
    dataset = RoboMMEHeliosDataset(**kwargs, history_sizes=[72, 2, 1],
                                  view_layout=TWO_VIEW_HORIZONTAL, episode_indices=[0])
    assert dataset.latent_shape == (48, 1, 16, 32) and sum(dataset.history_sizes) == 75
    with pytest.raises(ValueError, match='view_layout'):
        RoboMMEHeliosDataset(**kwargs)


def test_new_base_payload_requires_clock_shape_and_source_identity():
    row = dict(episode_index=0, length=35, exec_start_idx=17)
    metadata = new_metadata()
    frames = decision_frames(row['length'], row['exec_start_idx'])
    payload = dict(schema=metadata['schema'], episode_index=0, execution_start=17,
                   view_contract=metadata['view_contract'], source_manifest_sha256='a'*64,
                   source_payload_sha256='c'*64, frame_indices=torch.tensor(frames),
                   latents=torch.zeros(len(frames), 48, 1, 16, 32, dtype=torch.bfloat16))
    validate_payload(payload, row, metadata, 'c'*64)
    for key, value in [('view_contract', {}), ('source_payload_sha256', 'x'*64),
                       ('latents', torch.zeros(len(frames), 48, 1, 24, 20, dtype=torch.bfloat16))]:
        with pytest.raises(ValueError):
            validate_payload(dict(payload, **{key: value}), row, metadata, 'c'*64)


def test_dense_cache_new_layout_preserves_current_and_rejects_wrong_payload_contract(dense_cache_assets):
    args, manifest, row, original = dense_cache_assets
    root, dataset_root, base_root, base_manifest, _ = args
    metadata = metadata_for_source(base_manifest, 'd'*64, 'e'*64)
    base_manifest['metadata'] = metadata
    shape = metadata['latent_shape']
    original['latents'] = original['latents'][:, :1, :, :1, :1].expand(-1, *shape).clone()
    original.update(view_contract=metadata['view_contract'], schema=metadata['schema'],
                    source_manifest_sha256='d'*64)
    base_path = base_root/base_manifest['episodes'][str(row['episode_index'])]
    torch.save(original, base_path)
    base_manifest['payload_sha256'] = {str(row['episode_index']): sha256(base_path)}
    (base_root/'manifest.json').write_text(json.dumps(base_manifest))
    source_hash = sha256(base_root/'manifest.json')
    manifest['metadata'].update(view_layout=TWO_VIEW_HORIZONTAL, view_layout_version=2,
                                view_contract=metadata['view_contract'], latent_shape=shape,
                                source_manifest_sha256=source_hash)
    offsets = manifest['episodes'][str(row['episode_index'])]['offsets']
    offsets['0']['sha256'] = sha256(base_path)
    for offset in range(1, 16):
        path = root/offsets[str(offset)]['path']
        payload = torch.load(path, weights_only=True)
        payload['latents'] = payload['latents'][:, :1, :, :1, :1].expand(-1, *shape).clone()
        payload.update(view_contract=metadata['view_contract'], source_manifest_sha256=source_hash)
        validate_dense_payload(payload, row, offset, manifest['metadata'])
        torch.save(payload, path)
        offsets[str(offset)]['sha256'] = sha256(path)
    (root/'manifest.json').write_text(json.dumps(manifest))
    cache = DenseRoboMMECache(*args)
    for current in range(row['exec_start_idx'], row['length']-1):
        result = cache.prefix(row, current, original)
        assert result['latents'].shape[1:] == (48, 1, 16, 32)
        assert result['frame_indices'][-1] == current
    path = root/offsets['1']['path']
    payload = torch.load(path, weights_only=True)
    payload['view_contract'] = view_spec()
    torch.save(payload, path)
    offsets['1']['sha256'] = sha256(path)
    (root/'manifest.json').write_text(json.dumps(manifest))
    cache = DenseRoboMMECache(*args)
    with pytest.raises(ValueError, match='view contract'):
        cache.prefix(row, row['exec_start_idx']+1, original)
