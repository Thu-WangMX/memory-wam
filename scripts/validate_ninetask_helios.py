import json
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.utils import misc

root = Path(__file__).resolve().parents[1]
dest = root / 'runtime/ninetask_helios'
misc.register_work_dir(str(dest))
config = OmegaConf.create((dest / 'resolved.yaml').read_text().split('\n', 1)[1])
if (dest / 'dataset_stats.json').exists():
    config.data.train.pretrained_norm_stats = str(dest / 'dataset_stats.json')
dataset = instantiate(config.data.train, _recursive_=False)
print('SAMPLES', len(dataset), flush=True)
from collections import Counter
counts = Counter()
selected = {}
multi = dataset.lerobot_dataset.multi_dataset
print('DATASET_NAMES', multi.ds_names, flush=True)
# Episode boundaries allow sampling each source without loading every video.
import bisect
starts = dataset.lerobot_dataset.episode_data_index['from'].tolist()
episode_offset = 0
for record in json.loads((dest / 'tasks.json').read_text()):
    start = starts[episode_offset]
    episode_offset += record['episodes']
    index = bisect.bisect_left(dataset.sample_indices, start)
    sample = dataset[index]
    task = str(sample['dataset_name'])
    if task not in selected:
        selected[task] = sample
        print('CHECK', task, {k: list(v.shape) for k,v in sample.items() if isinstance(v, torch.Tensor)}, flush=True)
    counts[task] += 1
    if len(selected) == 9:
        break
assert len(selected) == 9, selected.keys()
from torch.utils.data import default_collate
batch = default_collate(list(selected.values()))
print('COLLATE_OK', len(selected), flush=True)
(dest / 'validation.json').write_text(json.dumps(dict(samples=len(dataset), tasks=list(selected), batch_size=9, passed=True), indent=2))
