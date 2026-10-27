"""Real nine-task dataset and pretrained-model batch4 forward/backward audit."""
import json,os,time,sys
from pathlib import Path
import torch
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from torch.utils.data import default_collate
from fastwam.datasets.lerobot.helios_dataset import collate_full_history

R=Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')
out=R/'runtime/memory_ablation_validation_20260929'
torch.set_num_threads(4)
variant=sys.argv[1]
bs=int(os.environ.get('CHECK_BATCH_SIZE','4'))
sizes='[0,0,1]' if variant=='anchor_only' else '[16,2,1]'
patches=['model.helios_config.patch_long=[1,2,2]','model.helios_config.patch_mid=[1,2,2]'] if variant!='anchor_only' else []
datasets='['+','.join(x['dataset'] for x in json.loads((R/'runtime/ninetask_helios/tasks.json').read_text()))+']'
with initialize_config_dir(config_dir=str(R/'configs'),version_base=None):
 cfg=compose(config_name='train',overrides=['task=rmbench_helios',
    'model.helios_config.memory_variant='+variant,'model.helios_config.history_sizes='+sizes,*patches,
    'data.train.dataset_dirs='+datasets,'data.train.pretrained_norm_stats=null',
    'data.train.full_kv_cache_path='+str(R/'runtime/ninetask_helios'),
    'data.train.text_embedding_cache_dir='+str(R/'runtime/ninetask_helios/text_cache')])
print('DATASET_START',flush=True)
dataset=instantiate(cfg.data.train)
starts=dataset.lerobot_dataset.episode_data_index['from'].tolist()
ends=dataset.lerobot_dataset.episode_data_index['to'].tolist()
ep=sorted(range(len(starts)),key=lambda i:ends[i]-starts[i],reverse=True)
indices={source:i for i,source in enumerate(dataset.sample_indices)}
chosen=[indices[starts[e]+((ends[e]-starts[e]-1)//16)*16] for e in ep[:bs]]
samples=[dataset._get(i) for i in chosen]
print('DATASET',len(dataset),'LONGEST_EPISODE_LENGTHS',[ends[i]-starts[i] for i in ep[:bs]],
      'HISTORY',[len(x['history_latents']) for x in samples],flush=True)
batch=collate_full_history(samples) if variant=='full_history_dense' else default_collate(samples)
print('MODEL_START',flush=True)
model=instantiate(cfg.model)
model.eval().requires_grad_(False)
model.dit.train().requires_grad_(True)
model.proprio_encoder.train().requires_grad_(True)
if variant=='anchor_only':
 model.video_expert.patch_long.requires_grad_(False);model.video_expert.patch_mid.requires_grad_(False)
torch.cuda.reset_peak_memory_stats()
start=time.monotonic()
report=dict(variant=variant,batch_size=bs,dataset_samples=len(dataset),
    source_indices=chosen,history_frames=[len(x['history_latents']) for x in samples])
try:
 loss,metrics=model.training_loss(batch)
 # Convert only at this diagnostic output boundary.
 metrics = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
            for key, value in metrics.items()}
 print('FORWARD',float(loss.detach()),metrics,'seconds',time.monotonic()-start,flush=True)
 assert torch.isfinite(loss)
 loss.backward()
 torch.cuda.synchronize()
 report.update(status='passed',loss=float(loss.detach()),metrics=metrics,
    seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,optimizer_included=False)
except Exception as e:
 report.update(status='failed',error=repr(e),seconds=time.monotonic()-start,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
 print(json.dumps(report),flush=True)
 (out/(variant+'_real.json')).write_text(json.dumps(report,indent=2))
 raise
print(json.dumps(report),flush=True)
(out/(variant+'_real.json')).write_text(json.dumps(report,indent=2))
