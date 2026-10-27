"""Incremental Wan2.2 causal encoding with the original full-prefix projection.

Only encoder feature caches persist between execution decisions. The cheap 1x1
projection and normalization still run on the full encoded prefix, matching the
original VAE's tensor shapes and floating-point operations exactly.
"""
import torch
from fastwam.models.wan22.wan_video_vae import VideoVAE38_, count_conv3d, patchify


class CausalVAEStream:
    def __init__(self, vae):
        if not isinstance(vae.model, VideoVAE38_):
            raise TypeError('Expected the checkpoint Wan2.2 48-channel VAE')
        self.vae = vae
        self.reset()

    def reset(self):
        self.feature_cache = [None] * count_conv3d(self.vae.model.encoder)
        self.encoded_chunks = []
        self.frames = 0

    @torch.no_grad()
    def append(self, video):
        expected = 1 if self.frames == 0 else 4
        if video.ndim != 5 or video.shape[:2] != (1, 3) or video.shape[2] != expected:
            raise ValueError(f'Expected [1,3,{expected},H,W] causal video chunk')
        model = self.vae.model
        x = patchify(video, patch_size=2)
        encoded, self.feature_cache, _ = model.encoder(
            x, feat_cache=self.feature_cache, feat_idx=[0])
        self.encoded_chunks.append(encoded)
        self.frames += expected
        full = torch.cat(self.encoded_chunks, dim=2)
        mu, _ = model.conv1(full).chunk(2, dim=1)
        scale = self.vae.scale
        if isinstance(scale[0], torch.Tensor):
            scale = [value.to(dtype=mu.dtype, device=mu.device) for value in scale]
            mu = (mu - scale[0].view(1, model.z_dim, 1, 1, 1)) * scale[1].view(
                1, model.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu
