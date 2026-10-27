"""Resource stress test with repeated RGB; not a simulator rollout or score."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.robodojo.policy import AnchorHeliosPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=2048)
    args=parser.parse_args()
    if args.steps%16:raise ValueError('Decision step must be a multiple of 16')
    if Path(args.output).exists():raise FileExistsError(args.output)
    cfg=json.loads(Path(args.deployment).read_text())
    torch.set_num_threads(4)
    policy=AnchorHeliosPolicy.from_checkpoint(cfg['config_path'],cfg['checkpoint_path'],prepared_root=cfg['prepared_root'],
              device='cuda:0',num_inference_steps=50,seed=42)
    data=Path(os.environ['ROBODOJO_DATA_ROOT'])
    instruction=json.loads((data/'meta/tasks.jsonl').read_text().splitlines()[0])['task']
    frame=torch.zeros(1,3,384,320)
    results=[]
    for step in [0,args.steps]:
        policy.reset()
        policy.frames=[frame]*(1+step//4)
        policy.last_state=np.zeros(14,dtype=np.float32)
        policy.instruction=instruction
        policy.observation_step=step; policy.last_inferred_step=step-16
        torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
        actions=policy.predict(); torch.cuda.synchronize()
        result={'decision_step':step,'rgb_frames':len(policy.frames),'seconds':time.monotonic()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'finite':bool(np.isfinite(actions).all())}
        print(json.dumps(result),flush=True); results.append(result)
    policy.reset()
    assert len(policy.frames)==0 and policy.observation_step==-1
    Path(args.output).write_text(json.dumps({'type':'synthetic_repeated_rgb_resource_test','benchmark_result':False,'results':results,'reset_passed':True},indent=2))


if __name__=='__main__':main()
