"""RoboDojo v2.1 data boundary for the unchanged Anchor-Helios contract.

All three paths (training, latent preparation, online policy) use the same RGB
composition. Actions at row t are the target issued from observation t; there
is no additional action shift. Historical latents contain observations <= t.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torchvision.transforms.functional import resize

from fastwam.datasets.lerobot.full_kv_dataset import FullKVObservationStore
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from fastwam.memory.helios_history import episode_anchors, fixed_history

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
ONLINE_CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
MEMORY_TASKS = {
    'cover_blocks': 3,
    'imitate_sorting_sequence': 11,
    'match_and_pick_from_conveyor': 16,
    'press_by_number': 25,
    'swap_T': 32,
    'swap_blocks': 33,
}
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
MOSAIC = 'wrists_top_head_bottom_384x320'
HORIZON = 16
RGB_STRIDE = 4


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dataset_metadata(root):
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    if (info['codebase_version'], info['fps'], info['total_episodes'], info['total_frames']) != (
            'v2.1', 25, 3500, 1859602):
        raise ValueError('Expected the audited RoboDojo v2.1 simulation dataset (25 Hz, 3500 episodes)')
    for key in ('action', 'observation.state'):
        if info['features'][key]['shape'] != [14]:
            raise ValueError(f'{key} must have 14 dimensions')
    tasks = {int(row['task_index']): row['task'] for row in read_jsonl(root / 'meta/tasks.jsonl')}
    episodes = read_jsonl(root / 'meta/episodes.jsonl')
    if len(episodes) != info['total_episodes']:
        raise ValueError('Episode metadata count mismatch')
    return info, tasks, episodes


def select_episodes(root, tasks=None, split='all', val_fraction=0.01, split_seed=42):
    """Split the full episode list before task filtering, matching StarWAM."""
    _, instructions, episodes = dataset_metadata(root)
    names = list(MEMORY_TASKS) if tasks is None else list(tasks)
    if names == ['all']:
        allowed = set(instructions.values())
    else:
        unknown = set(names) - MEMORY_TASKS.keys()
        if unknown or not names:
            raise ValueError(f'Choose nonempty Memory tasks or [all]; unknown={sorted(unknown)}')
        allowed = {instructions[MEMORY_TASKS[name]] for name in names}
    if split not in ('all', 'train', 'val') or not 0 <= val_fraction < 1:
        raise ValueError('Invalid episode split or val_fraction')
    positions = np.arange(len(episodes))
    if split != 'all':
        np.random.default_rng(split_seed).shuffle(positions)
        boundary = int(len(positions) * (1 - val_fraction))
        positions = positions[:boundary] if split == 'train' else positions[boundary:]
    selected = [episodes[int(i)] for i in sorted(positions)
                if set(episodes[int(i)]['tasks']) & allowed]
    if not selected:
        raise ValueError(f'No episodes for tasks={names}, split={split}')
    return selected


def episode_paths(root, info, episode):
    root = Path(root)
    index = int(episode['episode_index'])
    fields = dict(episode_index=index, episode_chunk=index // int(info['chunks_size']))
    parquet = root / info['data_path'].format(**fields)
    videos = [root / info['video_path'].format(video_key='observation.images.' + camera, **fields)
              for camera in CAMERAS]
    return parquet, videos


def decode_rgb(path, indices):
    """Decode exact episode-local frame indices to [T,3,H,W] RGB uint8.

    Sequential decoding avoids AV1 seek/timestamp ambiguity. Missing source
    frames are errors; the caller explicitly clamps only known episode tails.
    """
    indices = [int(i) for i in indices]
    if not indices or min(indices) < 0:
        raise ValueError('Frame indices must be nonempty and nonnegative')
    wanted = set(indices)
    found = {}
    with av.open(str(path), timeout=30) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'NONE'
        stream.codec_context.thread_count = 1
        try:
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    found[index] = torch.from_numpy(frame.to_ndarray(format='rgb24').copy()).permute(2, 0, 1)
                if index >= max(wanted):
                    break
        finally:
            stream.codec_context.close()
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'{path}: missing source frames {sorted(missing)}')
    return torch.stack([found[index] for index in indices])


def compose_rgb(cameras):
    """Head/left/right RGB [T,3,H,W] -> wrists-top float [T,3,384,320]."""
    if len(cameras) != 3:
        raise ValueError('Expected three cameras in head/left/right order')
    prepared = []
    for camera in cameras:
        if camera.dtype != torch.uint8 or camera.ndim != 4 or camera.shape[1] != 3:
            raise ValueError(f'Expected RGB uint8 [T,3,H,W], got {camera.dtype}, {camera.shape}')
        prepared.append(resize(camera.float() / 255., [240, 320], antialias=True))
    head = resize(prepared[0], [256, 320], antialias=True)
    wrists = torch.cat([resize(prepared[i], [128, 160], antialias=True) for i in (1, 2)], dim=-1)
    return torch.cat([wrists, head], dim=-2).mul(2).sub(1)


def observation_image(vision):
    cameras = [torch.from_numpy(np.array(vision[key]['color'], copy=True))
               .permute(2, 0, 1).unsqueeze(0) for key in ONLINE_CAMERAS]
    return compose_rgb(cameras)


def episode_video(root, info, episode, indices):
    _, paths = episode_paths(root, info, episode)
    return compose_rgb([decode_rgb(path, indices) for path in paths]).permute(1, 0, 2, 3).contiguous()


def text_filename(instruction):
    digest = hashlib.sha256(PROMPT.format(task=instruction).encode()).hexdigest()
    return f'{digest}.t5_len128.wan22ti2v5b.pt'


def load_context(cache_dir, instruction):
    payload = torch.load(Path(cache_dir) / text_filename(instruction), map_location='cpu', weights_only=True)
    context, mask = payload['context'].clone(), payload['mask'].bool()
    if context.shape != (128, 4096) or mask.shape != (128,):
        raise ValueError('Run prepare_robodojo_helios.py assets to pad the RoboDojo text cache to 128')
    # Match the existing FastWAM RobotVideoDataset, including its text-mask semantics.
    context[~mask] = 0
    return context, torch.ones_like(mask)


def normalizers(stats_path):
    stats = json.loads(Path(stats_path).read_text())
    result = {}
    for field in ('action', 'state'):
        entry = stats[field]['default']
        values = {key.removeprefix('global_'): torch.as_tensor(value, dtype=torch.float32)
                  for key, value in entry.items() if key.startswith('global_')}
        if values['mean'].shape != (14,) or values['std'].shape != (14,):
            raise ValueError(f'Expected 14D {field} statistics')
        if not all(torch.isfinite(value).all() for value in values.values()) or (values['std'] < 0).any():
            raise ValueError(f'Invalid {field} statistics')
        result[field] = SingleFieldLinearNormalizer(values, mode='z-score')
    return result


class RoboDojoHeliosDataset(torch.utils.data.Dataset):
    """Strict causal samples for the existing FastWAMHelios.training_loss."""
    def __init__(self, root, prepared_root, full_kv_cache_path, tasks=None, split='train',
                 val_fraction=0.01, split_seed=42, history_sizes=(16, 2, 1), anchor_frames=2,
                 episode_indices=None, pretrained_norm_stats=None):
        self.root = Path(root).resolve()
        self.prepared_root = Path(prepared_root).resolve()
        self.info, self.instructions, _ = dataset_metadata(self.root)
        self.episodes = select_episodes(self.root, tasks, split, val_fraction, split_seed)
        if episode_indices is not None:
            keep = set(map(int, episode_indices))
            self.episodes = [row for row in self.episodes if int(row['episode_index']) in keep]
            if not self.episodes:
                raise ValueError('episode_indices selects no episodes in this task/split')
        self.history_sizes = tuple(history_sizes)
        self.anchor_frames = int(anchor_frames)
        if self.history_sizes != (16, 2, 1) or self.anchor_frames not in (0, 2):
            raise ValueError('RoboDojo port retains history_sizes=(16,2,1), anchor_frames=0 or 2')
        stats_path = pretrained_norm_stats or self.prepared_root / 'dataset_stats.json'
        self.norm = normalizers(stats_path)
        self.store = FullKVObservationStore(full_kv_cache_path, expected_replan_stride=HORIZON)
        cache_meta = json.loads((Path(full_kv_cache_path) / 'manifest.json').read_text())['metadata']
        if cache_meta.get('dataset_info_sha256') != sha256(self.root / 'meta/info.json'):
            raise ValueError('Latent cache belongs to a different dataset')
        if cache_meta.get('source_format') != 'robodojo_mp4_rgb':
            raise ValueError('RoboDojo requires MP4-derived historical latents')
        self.samples = []
        for row in self.episodes:
            ep = int(row['episode_index'])
            if f'{self.root.name}/{ep}' not in self.store.episodes:
                raise ValueError(f'Latent cache is missing selected episode {ep}; prepare latents first')
            self.samples.extend((row, t) for t in range(0, int(row['length']), HORIZON))
        self._tables = OrderedDict()
        self._contexts = {}

    def __len__(self):
        return len(self.samples)

    def _table(self, row):
        ep = int(row['episode_index'])
        table = self._tables.pop(ep, None)
        if table is None:
            path, _ = episode_paths(self.root, self.info, row)
            raw = pq.read_table(path, columns=['action', 'observation.state', 'task_index',
                                               'frame_index', 'episode_index'])
            table = {name: torch.as_tensor(np.asarray(raw[name].to_pylist())) for name in raw.column_names}
            n = int(row['length'])
            if (len(table['action']) != n or table['action'].shape != (n, 14)
                    or table['observation.state'].shape != (n, 14)
                    or not torch.equal(table['frame_index'].reshape(-1), torch.arange(n))
                    or not (table['episode_index'] == ep).all()):
                raise ValueError(f'Parquet/metadata alignment mismatch: episode {ep}')
            if not torch.isfinite(table['action']).all() or not torch.isfinite(table['observation.state']).all():
                raise ValueError(f'Nonfinite action/state in episode {ep}')
        self._tables[ep] = table
        while len(self._tables) > 2:
            self._tables.popitem(last=False)
        return table

    def __getitem__(self, index):
        row, start = self.samples[index]
        ep, n = int(row['episode_index']), int(row['length'])
        table = self._table(row)
        ai = torch.arange(start, start + HORIZON)
        vi = torch.arange(start, start + HORIZON + 1, RGB_STRIDE)
        instruction = self.instructions[int(table['task_index'][start])]
        if instruction not in self._contexts:
            self._contexts[instruction] = load_context(self.prepared_root / 'text_cache', instruction)
        context, mask = self._contexts[instruction]
        identity = {'episode_index': ep, 'frame_index': start, 'dataset_name': self.root.name}
        history = self.store.strict_history(identity)
        anchors, anchor_valid = episode_anchors(history, self.anchor_frames)
        history, history_valid = fixed_history(history, sum(self.history_sizes))
        return {
            'video': episode_video(self.root, self.info, row, vi.clamp(max=n-1).tolist()),
            'action': self.norm['action'].forward(table['action'][ai.clamp(max=n-1)].float()),
            'proprio': self.norm['state'].forward(table['observation.state'][ai.clamp(max=n-1)].float()),
            'action_is_pad': ai >= n, 'image_is_pad': vi >= n,
            'context': context.clone(), 'context_mask': mask.clone(),
            'prompt': PROMPT.format(task=instruction),
            'history_latents': history, 'history_valid': history_valid,
            'anchor_latents': anchors, 'anchor_valid': anchor_valid,
            'episode_index': ep, 'frame_index': start,
        }
