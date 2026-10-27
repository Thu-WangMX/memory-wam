"""The existing RMBench data pipeline with a fixed-size Helios history window."""
import torch
from .full_kv_dataset import FullKVRobotVideoDataset


from fastwam.memory.helios_history import episode_anchors, select_history, MEMORY_VARIANTS


class HeliosRobotVideoDataset(FullKVRobotVideoDataset):
    def __init__(self, *args, history_sizes=(16, 2, 1), anchor_frames=2, memory_variant="multiscale", **kwargs):
        self.history_sizes = tuple(int(x) for x in history_sizes)
        self.anchor_frames = int(anchor_frames)
        self.memory_variant = str(memory_variant)
        if self.memory_variant not in MEMORY_VARIANTS:
            raise ValueError(f'Unknown Helios memory_variant: {self.memory_variant}')
        if len(self.history_sizes) != 3 or self.history_sizes[-1] != 1:
            raise ValueError("Helios requires long/mid/current sizes with current=1")
        if self.anchor_frames not in (0, 2):
            raise ValueError("Helios supports either zero (ablation) or two episode anchors")
        super().__init__(*args, **kwargs)

    def _get(self, idx):
        sample = super()._get(idx)
        sample['anchor_latents'], sample['anchor_valid'] = episode_anchors(
            sample['history_latents'], self.anchor_frames)
        sample['history_latents'], sample['history_valid'] = select_history(
            sample['history_latents'], self.history_sizes, self.memory_variant)
        return sample



def collate_full_history(samples):
    """Left-pad variable episode prefixes, distinct from the original startup pad.

    Only extra batch padding is attention-masked; startup zeros retain the
    baseline dense19 behavior. No truncation, resampling or length bucketing.
    """
    from torch.utils.data import default_collate
    sizes = [len(x['history_latents']) for x in samples]
    max_size = max(sizes)
    padded = []
    for sample, size in zip(samples, sizes):
        item = dict(sample)
        h, valid = item['history_latents'], item['history_valid']
        pad = max_size - size
        item['history_latents'] = torch.cat((h.new_zeros((pad, *h.shape[1:])), h))
        item['history_valid'] = torch.cat((valid.new_zeros(pad), valid))
        item['history_batch_pad'] = torch.arange(max_size) < pad
        item['history_unpadded_length'] = size
        padded.append(item)
    return default_collate(padded)
