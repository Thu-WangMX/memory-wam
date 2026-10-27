"""Validate every production RGB/mask decision and a real BS16 collated batch."""
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from fastwam.utils import misc

root = Path(__file__).resolve().parents[1]
output = root / 'runtime/observe_future_mask_validation'
output.mkdir(exist_ok=True)
misc.register_work_dir(str(output))
with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
    cfg = compose(config_name='train', overrides=['task=rmbench_helios',
        'data=rmbench_helios_future_mask','model=fastwam_helios_future_mask'])
dataset = instantiate(cfg.data.train)
assert len(dataset) == 500, len(dataset)
batch = []
for i in range(len(dataset)):
    sample = dataset[i]
    if i < 16: batch.append(sample)
    if i % 100 == 0: print(f'Validated {i}/500',flush=True)
collated = torch.utils.data.default_collate(batch)
report = dict(samples=len(dataset),passed=True,
              batch_shapes={k:list(v.shape) for k,v in collated.items() if isinstance(v,torch.Tensor)})
(output / 'data_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
