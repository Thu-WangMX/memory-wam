"""Pretrained model backward + cached/online VAE parity + checkpoint roundtrip.

Gradients are validated then released to fit alongside existing GPU jobs. Only
small new projections receive an optimizer smoke step; this is NOT a trained
checkpoint or a benchmark success-rate evaluation.
"""
import argparse
import json
import shutil
import time
import types
from pathlib import Path
import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.datasets.robodojo import episode_paths
from experiments.robomme.train import validate_recipe
from experiments.robomme.policy import AnchorHeliosPolicy
from fastwam.memory.helios_history import memory_variant_profile
from fastwam.memory.robomme_clock import decision_frames


def replay_inputs(ds, row, start, end, first):
    _, paths = episode_paths(ds.root, ds.info, row)
    images=[]
    for path in paths[:2]:
        reader=cv2.VideoCapture(str(path)); found=[]
        try:
            for i in range(end):
                ok, frame=reader.read()
                if not ok: raise ValueError(f'Missing replay frame {i}')
                if i>=start: found.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally: reader.release()
        images.append(found)
    table=ds._table(row)
    states=table['observation.state'][start:end].numpy()
    instruction=ds.instructions[int(table['task_index'][end-1])]
    return dict(front_rgb_list=images[0],wrist_rgb_list=images[1],
                joint_state_list=[x[:7] for x in states],
                gripper_state_list=[np.repeat(x[7],2) for x in states],
                task_goal=instruction,is_first_step=first)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--memory-gib',type=float,default=27)
    p.add_argument('--skip-backward',action='store_true')
    p.add_argument('--save-checkpoint',action='store_true')
    p.add_argument('--memory-variant',choices=['multiscale','multiscale_long99'],default='multiscale')
    a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(a.memory_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    root=Path(__file__).resolve().parents[1]; out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    profile=memory_variant_profile(a.memory_variant)
    overrides=[f'model.helios_config.{key}='+json.dumps(profile[key],separators=(',',':'))
               for key in ('history_sizes','patch_long','patch_mid','anchor_frames')]
    overrides.append('model.helios_config.memory_variant='+a.memory_variant)
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=robomme_helios',f'output_dir={out}',*overrides])
    report={'data':validate_recipe(cfg),'trained_checkpoint':False,'memory_variant':a.memory_variant}
    OmegaConf.save(OmegaConf.to_container(cfg,resolve=True),out/'config.yaml')
    shutil.copyfile(cfg.data.train.pretrained_norm_stats,out/'dataset_stats.json')
    ds=instantiate(cfg.data.train)
    print('PREFLIGHT_OK loading pretrained model',flush=True)
    model=instantiate(cfg.model)
    model.eval().requires_grad_(False)
    long_row=max(ds.episodes,key=lambda row:len(decision_frames(row['length'],row['exec_start_idx'])))
    long_index=ds.episode_samples[long_row['episode_index']][-1]
    long_sample=ds[long_index]
    if not a.skip_backward:
        # Choose a real long prefix so both compression branches see observations.
        index=long_index if a.memory_variant=='multiscale_long99' else next(
            i for i,(r,t) in enumerate(ds.samples) if t-r['exec_start_idx']>=320)
        sample=ds[index];batch=torch.utils.data.default_collate([sample])
        model.dit.train().requires_grad_(True);model.proprio_encoder.train().requires_grad_(True)
        kept={n:p for n,p in model.named_parameters() if p.requires_grad and
              any(k in n for k in ('patch_long.','patch_mid.','proprio_encoder.'))}
        optimizer=torch.optim.SGD(kept.values(),lr=1e-5)
        gradients={};hooks=[]
        def check(name):
            def hook(parameter):
                grad=parameter.grad.detach().cpu().float()
                if not torch.isfinite(grad).all(): raise ValueError(f'Nonfinite gradient: {name}')
                gradients[name]=float(grad.norm())
                if name not in kept: parameter.grad=None
            return hook
        for name,param in model.named_parameters():
            if param.requires_grad: hooks.append(param.register_post_accumulate_grad_hook(check(name)))
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        loss,metrics=model.training_loss(batch)
        # Convert only at this diagnostic output boundary.
        metrics = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                   for key, value in metrics.items()}
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        print('FORWARD_OK '+json.dumps(metrics),flush=True)
        loss.backward();torch.cuda.synchronize()
        for hook in hooks:hook.remove()
        for suffix in ('patch_long.weight','patch_mid.weight'):
            values=[v for k,v in gradients.items() if k.endswith(suffix)]
            assert len(values)==1 and values[0]>0,(suffix,values)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        report['backward']=dict(loss=float(loss.detach()),metrics=metrics,seconds=time.monotonic()-started,
                                gradient_tensors=len(gradients),episode=sample['episode_index'],frame=sample['frame_index'],
                                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                                optimizer_scope='small new memory projections and proprio only')
        (out/'backward.json').write_text(json.dumps(report['backward'],indent=2)+'\n')
        del loss,batch,optimizer;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        print('FULL_BACKWARD_OK',flush=True)
    model.eval().requires_grad_(False)
    # Exercise the real long-prefix inference path against the dataset selection.
    cache=ds.observed_prefix(long_row,long_sample['frame_index'])
    observed=cache['latents']
    prefix=observed.squeeze(2).permute(1,0,2,3).unsqueeze(0)
    original_video_pre=model._video_pre
    captured={}
    def capture(self,anchors,history,*args,**kwargs):
        captured['anchors']=anchors.detach().cpu()
        captured['history']=history.detach().cpu()
        return original_video_pre(anchors,history,*args,**kwargs)
    model._video_pre=types.MethodType(capture,model)
    try:
        prediction=model.infer_action(prompt=None,input_image=None,observation_latents=prefix,
            observation_frame_indices=cache['frame_indices'].tolist(),execution_start=long_row['exec_start_idx'],
            action_horizon=16,proprio=long_sample['proprio'][0],context=long_sample['context'],
            context_mask=long_sample['context_mask'],full_kv_frame_index=len(observed)-1,
            num_inference_steps=2,seed=42)
    finally:
        model._video_pre=original_video_pre
    torch.testing.assert_close(captured['anchors'][0],long_sample['anchor_latents'])
    torch.testing.assert_close(captured['history'][0],long_sample['history_latents'])
    assert torch.isfinite(prediction['action']).all()
    report['long_prefix']=dict(episode=long_row['episode_index'],frame=long_sample['frame_index'],
        observed_latents=len(observed),history_slots=sum(profile['history_sizes']),
        valid_history=int(long_sample['history_valid'].sum()),
        video_history_tokens=prediction['full_kv_video_tokens'],
        train_infer_selection_equal=True,action_shape=list(prediction['action'].shape))
    print('LONG_PREFIX_PARITY_OK '+json.dumps(report['long_prefix']),flush=True)
    policy=AnchorHeliosPolicy(model,out/'dataset_stats.json',ds.prepared_root/'text_cache',2,42)
    cases=[ds.rows[1],next(r for r in ds.rows if r['exec_start_idx']==0 and r['length']>33)]
    report['replay']=[]
    for row in cases:
        policy.reset();execution=row['exec_start_idx'];cache=ds.load_latents(row)
        for decision in range(2):
            t=execution+16*decision
            inputs=replay_inputs(ds,row,0 if decision==0 else t-15,t+1,decision==0)
            prediction=policy.infer(inputs)
            j=cache['frame_indices'].tolist().index(t)
            expected=cache['latents'][:j+1].squeeze(2).permute(1,0,2,3).unsqueeze(0)
            error=(expected.float()-policy.last_prefix.float()).abs()
            # Same VAE + decoder + phase clock should reproduce the BF16 cache.
            torch.testing.assert_close(policy.last_prefix,expected,atol=.04,rtol=.01)
            report['replay'].append(dict(episode=row['episode_index'],frame=t,decisions=j+1,
                                         max_latent_error=float(error.max()),mean_latent_error=float(error.mean()),
                                         action_shape=list(prediction['actions'].shape)))
            print('ONLINE_CACHE_PARITY_OK '+json.dumps(report['replay'][-1]),flush=True)
    if a.save_checkpoint:
        checkpoint=out/'step_000000_smoke_only.pt'
        model.save_checkpoint(checkpoint,step=0)
        model.load_checkpoint(checkpoint)
        report['checkpoint']=str(checkpoint)
        report['strict_checkpoint_roundtrip']=True
    (out/'model_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ROBOMME_MODEL_SMOKE_OK '+json.dumps(report),flush=True)


if __name__=='__main__':main()
