"""Read frozen labels without loading DreamDojo; no training integration yet."""
import json
from pathlib import Path
import numpy as np


class DreamDojoLabelStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['status'] != 'complete':
            raise ValueError('label cache is not finalized')
        self.latents = np.load(self.root / 'latents.npy', mmap_mode='r', allow_pickle=False)
        with np.load(self.root / 'frame_index.npz', allow_pickle=False) as f:
            self.valid = f['valid'].copy()
            self.episode_index = f['episode_index'].copy()
            self.frame_index = f['frame_index'].copy()
        # The finalizer verifies global rows are exactly index 0..N-1.
        starts = np.flatnonzero(np.r_[True, np.diff(self.episode_index) != 0])
        ends = np.r_[starts[1:], len(self.episode_index)]
        self.offsets = {int(self.episode_index[s]): (int(s), int(e-s)) for s,e in zip(starts,ends)}
        self.mean = np.asarray(self.manifest['normalization']['all']['mean'], dtype=np.float32)
        self.std = np.asarray(self.manifest['normalization']['all']['std'], dtype=np.float32)

    def past_targets(self, episode_index, destination_frames, *, observed_through, normalize=True):
        """Select already-observed forward transitions. Invalid labels are zero.

        `observed_through` and `destination_frames` are ORIGINAL stored frame
        indices (demo + execution), never VAE indices or execution-relative time.
        Callers must use the returned boolean mask in their reconstruction loss.
        """
        start, length = self.offsets[int(episode_index)]
        if not 0 <= observed_through < length:
            raise ValueError('observed_through is outside this episode')
        frames = np.asarray(destination_frames, dtype=np.int64)
        if frames.ndim != 1:
            raise ValueError('destination_frames must be one-dimensional')
        rows = start + np.clip(frames, 0, length-1)
        valid = (frames >= 1) & (frames < length) & (frames <= observed_through) & self.valid[rows]
        labels = np.array(self.latents[rows], copy=True)
        if normalize:
            labels = (labels-self.mean)/self.std
        labels[~valid] = 0
        return labels, valid
