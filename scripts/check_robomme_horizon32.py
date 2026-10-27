"""Audit real training samples and all new Hydra recipes without using a GPU."""
import argparse,json
from pathlib import Path
import torch
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from experiments.robomme.train import validate_recipe


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    reports={}
    for suffix in ['past32_flow','pastfuture32_flow','pastfuture32_light_flow']:
        with initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
            cfg=compose(config_name='train',overrides=[f'task=robomme_helios_{suffix}',f'output_dir={out}', 'wandb.enabled=false'])
        provenance=validate_recipe(cfg);ds=instantiate(cfg.data.train)
        checked=[]
        for task,episodes in ds.task_episodes.items():
            ep=episodes[0];indices=ds.episode_samples[ep]
            for i in [indices[0],indices[len(indices)//2],indices[-1]]:
                x=ds[i];row,t=ds.samples[i];n=row['length'];table=ds._table(row)
                assert x['action'].shape==(32,8) and x['video'].shape==(3,9,384,320)
                ai=torch.arange(t+1,t+33)
                torch.testing.assert_close(x['action'],ds.norm['action'].forward(table['action'][ai.clamp(max=n-1)]),atol=0,rtol=0)
                assert torch.equal(x['action_is_pad'],ai>=n)
                assert torch.equal(x['image_is_pad'],torch.arange(t,t+33,4)>=n)
                payload=ds.observed_prefix(row,t)
                frames=payload['frame_indices'].tolist()
                if ds.anchor_mode == 'demo_execution_first':
                    boundary=row['exec_start_idx']
                    assert x['anchor_valid'].tolist()==[boundary>0,True]
                    torch.testing.assert_close(x['anchor_latents'][1],payload['latents'][frames.index(boundary)],atol=0,rtol=0)
                    if boundary:
                        torch.testing.assert_close(x['anchor_latents'][0],payload['latents'][0],atol=0,rtol=0)
                    else:
                        assert x['anchor_latents'][0].count_nonzero()==0
                online=ds.query_layout(frames,row['exec_start_idx'])
                for k,v in online.items():torch.testing.assert_close(x[k],v,atol=0,rtol=0)
                valid=x.get('past_action_loss_valid',x['past_action_transition_valid'])
                assert torch.count_nonzero(x['past_action_target'][~valid])==0
                assert (x['past_action_transition_frames'][valid]<n).all()
                assert (x['past_action_transition_frames'][:16][x['past_action_transition_valid'][:16]]<=t).all()
                if 'past_action_loss_valid' in x:
                    assert x['past_action_valid'][-2:].all() and x['past_action_transition_valid'][-2:].all()
                    assert x['past_action_relative_time'][-2:].tolist()==[1.,2.]
                checked.append(dict(task=task,episode=ep,frame=t,valid_labels=int(valid.sum())))
        reports[suffix]=dict(provenance=provenance,samples=checked,
                             patch_long=list(cfg.model.helios_config.patch_long),anchor_mode=ds.anchor_mode)
        (out/'data_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('RECIPE_DATA_OK',suffix,len(checked),flush=True)

if __name__=='__main__':main()
