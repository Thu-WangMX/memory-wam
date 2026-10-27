"""Manual real-assets/GPU smoke. No optimizer steps, rollout or training launch."""
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from experiments.robomme.train import validate_recipe
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
root=Path(__file__).resolve().parents[1]
cache='/mnt/vepfs01/output/spidy.wang/runtime-robomme/two-view-square256-v1'
source=root/'runs/robomme_all16_push75_past32x8_future2_dense1_resume15k_8gpu_b32ga1_s100k_20261025_201820/checkpoints/weights/step_017500.pt'
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(48*1024**3/torch.cuda.get_device_properties(0).total_memory)
configs={}
for kind in ['asv','288']:
    with initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=[f'task=robomme_helios_push75_{kind}',
            'data.train.tasks=[all]', 'data.train.task_balanced=true', 'batch_size=32',
            'gradient_accumulation_steps=1', f'data.train.full_kv_cache_path={cache}/base',
            f'data.train.prepared_root={cache}/prepared',f'data.train.pretrained_norm_stats={cache}/prepared/dataset_stats.json',
            f'data.train.dense_latents_path={cache}/dense'])
    if kind=='asv': cfg.model.helios_config.asv_init_checkpoint=str(source)
    report=validate_recipe(cfg)
    configs[kind]=cfg
    print('PREFLIGHT_OK',kind,report['episodes'],report['decisions'],flush=True)

cfg=configs['asv']
data=instantiate(cfg.data.train)
sample=torch.utils.data.default_collate([data[0]])
mc=OmegaConf.create(OmegaConf.to_container(cfg.model,resolve=True))
mc.skip_dit_load_from_pretrain=True;mc.action_dit_pretrained_path=None;mc.load_text_encoder=False
model=instantiate(mc,device='cuda:0',model_dtype=torch.bfloat16)
assert type(model).__name__=='FastWAMHeliosPastFutureASV'
print('REAL_ASV_INIT_OK',model.helios_config['action_video_attention'],flush=True)
model.train()
with torch.autocast('cuda',dtype=torch.bfloat16):
    loss,metrics=model.training_loss(sample)
    # Convert only at this diagnostic output boundary.
    metrics = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
               for key, value in metrics.items()}
assert torch.isfinite(loss)
loss.backward()
for p in [model.action_expert.head.weight,model.past_action_expert.head.weight,model.video_expert.head.head.weight]:
    assert p.grad is not None and torch.isfinite(p.grad).all()
print('REAL_TRAIN_BACKWARD_OK',json.dumps(metrics),flush=True)
model.zero_grad(set_to_none=True);del loss;torch.cuda.empty_cache()
model.eval().requires_grad_(False)
inputs=model.build_inputs(sample)
kwargs=dict(prompt=None,input_image=None,action_horizon=32,
            proprio=torch.zeros(1,8),context=inputs['video_context'],context_mask=inputs['video_context_mask'],
            num_inference_steps=3,seed=42,observation_latents=torch.randn(1,48,3,16,32),
            observation_frame_indices=[0,16,32],execution_start=0,full_kv_frame_index=2)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    result=model.infer_action(**kwargs)
assert result['action'].shape==(32,8) and torch.isfinite(result['action']).all()
assert result['future_video_prediction'].shape==(1,48,2,16,32)
assert torch.isfinite(result['future_video_prediction']).all()
print('REAL_ASV_INFERENCE_OK',json.dumps(dict(action=list(result['action'].shape),
    future=list(result['future_video_prediction'].shape),steps=result['future_video_flow_steps'],
    peak_gib=torch.cuda.max_memory_allocated()/1024**3)),flush=True)
