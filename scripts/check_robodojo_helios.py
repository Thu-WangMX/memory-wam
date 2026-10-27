"""Read real RoboDojo samples and optionally verify continuous VAE prefixes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.robodojo import (
    RoboDojoHeliosDataset, decode_rgb, episode_paths, episode_video, observation_image,
    ONLINE_CAMERAS, sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--latents', required=True)
    parser.add_argument('--episode', type=int, default=300)
    parser.add_argument('--task', default='cover_blocks')
    parser.add_argument('--vae', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    ds = RoboDojoHeliosDataset(args.root, args.prepared, args.latents,
                              tasks=[args.task], split='all', episode_indices=[args.episode])
    selected = [0, 1, 2, len(ds)-1]
    samples = [ds[i] for i in selected]
    row = ds.episodes[0]
    for sample in samples:
        t = sample['frame_index']
        assert sample['video'].shape == (3, 5, 384, 320)
        assert sample['action'].shape == (16, 14)
        assert sample['context'].shape == (128, 4096)
        assert sample['history_latents'].shape == (19, 48, 1, 24, 20)
        assert sample['anchor_latents'].shape == (2, 48, 1, 24, 20)
        assert int(sample['history_valid'].sum()) == min(t//16+1, 19)
        assert int(sample['anchor_valid'].sum()) == min(t//16+1, 2)
        expected_pad = torch.arange(t, t+16) >= int(row['length'])
        assert torch.equal(sample['action_is_pad'], expected_pad)
        raw = ds._table(row)['action'][min(t, int(row['length'])-1)].float()
        expected = ds.norm['action'].forward(raw)
        torch.testing.assert_close(sample['action'][0], expected)
    _, paths = episode_paths(ds.root, ds.info, row)
    raw_frames = [decode_rgb(path, [0]) for path in paths]
    vision = {name: {'color': tensor[0].permute(1, 2, 0).numpy()}
              for name, tensor in zip(ONLINE_CAMERAS, raw_frames)}
    online = observation_image(vision)[0]
    torch.testing.assert_close(online, samples[0]['video'][:, 0], rtol=0, atol=0)
    batch = torch.utils.data.default_collate(samples[:3])
    report = {'episode': args.episode, 'decisions': len(ds), 'sample_indices': selected,
              'batch_video_shape': list(batch['video'].shape), 'rgb_train_online_max_abs': 0.,
              'last_action_padding': samples[-1]['action_is_pad'].tolist(), 'data_contract': 'passed'}
    if args.vae:
        from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
        _, _, vae_config, _ = _resolve_configs(
            model_id='Wan-AI/Wan2.2-TI2V-5B', tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B',
            redirect_common_files=True)
        vae_config.download_if_necessary()
        metadata = json.loads((Path(args.latents) / 'manifest.json').read_text())['metadata']
        assert sha256(vae_config.path) == metadata['vae_model_sha256']
        model = _load_registered_model(vae_config.path, 'wan_video_vae',
                                       torch_dtype=torch.bfloat16, device='cuda:0').eval()
        report['prefix_parity'] = []
        for t in (0, 16, 32):
            video = episode_video(ds.root, ds.info, row, list(range(0, t+1, 4)))
            with torch.no_grad():
                latent = model.encode([video.to(device='cuda:0', dtype=torch.bfloat16)], device='cuda:0')[0]
            cached = ds.store.strict_history({'episode_index': args.episode, 'frame_index': t,
                                              'dataset_name': ds.root.name})
            expected = cached.squeeze(2).permute(1, 0, 2, 3).to(latent.device)
            diff = (latent.float() - expected.float()).abs()
            result = {'frame': t, 'max_abs': float(diff.max()), 'mean_abs': float(diff.mean())}
            assert result['max_abs'] <= 0.02 and result['mean_abs'] <= 0.002, result
            report['prefix_parity'].append(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
