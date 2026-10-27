"""Official RoboMME client protocol; mirrored from Kevin's validated rollout.

Always record the split and step limit. Short smoke rollouts are not official
benchmark scores. status == success is the sole success criterion.
"""
import argparse
from collections import deque
import json
import os
import time
from pathlib import Path
import numpy as np
from challenge_interface.client import PolicyClient
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from experiments.robomme.video_recording import EpisodeVideo


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=BenchmarkEnvBuilder.get_task_list())
    p.add_argument('--split',default='val',choices=['val','test'])
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--episode-start',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=1300,
                   help='MME-VLA evaluation uses 1300; RoboMME Challenge uses 1500')
    p.add_argument('--port',type=int,default=18176)
    p.add_argument('--contract',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--save-video',action=argparse.BooleanOptionalAction,default=True,
                   help='Save every scored rollout, including demo and terminal observations (default: on)')
    p.add_argument('--memory-gib',type=float,help='Limit simulator PyTorch CUDA allocations')
    a=p.parse_args()
    if a.memory_gib is not None:
        import torch
        total=torch.cuda.get_device_properties(0).total_memory
        limit=int(a.memory_gib*2**30)
        if not 0<limit<=total: raise ValueError('Invalid simulator CUDA allocator budget')
        torch.cuda.set_per_process_memory_fraction(limit/total,0)
        print(f'ROBOMME_SIM_ALLOCATOR_LIMIT_GIB={a.memory_gib}',flush=True)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    client=PolicyClient(host='127.0.0.1',port=a.port)
    contract=json.loads(Path(a.contract).read_text())
    if client.get_server_metadata()!=contract or a.task not in contract['semantic_tasks']:
        raise ValueError('Server/checkpoint/task mismatch')
    builder=BenchmarkEnvBuilder(env_id=a.task,dataset=a.split,action_space='joint_angle',
                                gui_render=False,max_steps=a.max_steps)
    if a.episodes<1 or a.episode_start<0 or a.episode_start+a.episodes>builder.get_episode_num():
        raise ValueError('Invalid official episode range')
    results=[]
    for episode in range(a.episode_start,a.episode_start+a.episodes):
        episode_started=time.perf_counter()
        print('ROBOMME_EPISODE_START '+json.dumps(dict(task=a.task,episode=episode,
              split=a.split,max_steps=a.max_steps)),flush=True)
        assert client.reset()['reset_finished']
        env=builder.make_env_for_episode(episode)
        video=EpisodeVideo(output,episode) if a.save_video else None
        try:
            device=env.unwrapped.backend.render_device
            visible=os.getenv('CUDA_VISIBLE_DEVICES','')
            if visible and ',' not in visible and (not device.is_cuda() or device.cuda_id!=0):
                raise RuntimeError(f'Renderer is not using the assigned CUDA device: {device}')
            print(f'ROBOMME_RENDER_DEVICE={device} PCI={device.pci_string}',flush=True)
            obs,info=env.reset();goal=info['task_goal']
            if video is not None: video.append(obs,0,initial=True)
            reset_seconds=time.perf_counter()-episode_started
            print('ROBOMME_RESET '+json.dumps(dict(episode=episode,
                  observations=len(obs['front_rgb_list']),seconds=round(reset_seconds,3))),flush=True)
            inputs=dict(task_goal=goal,is_first_step=True,**{k:list(v) for k,v in obs.items()})
            queue=deque();steps=0;decisions=0;policy_seconds=0.;simulation_seconds=0.
            while True:
                if not queue:
                    tick=time.perf_counter()
                    actions=np.asarray(client.infer(inputs)['actions'],dtype=np.float32)
                    policy_seconds+=time.perf_counter()-tick
                    if actions.shape!=(16,8) or not np.isfinite(actions).all():
                        raise ValueError('Invalid joint action chunk')
                    queue.extend(actions);decisions+=1
                    inputs=dict(task_goal=goal,is_first_step=False,**{k:[] for k in obs})
                tick=time.perf_counter()
                obs,_,terminated,truncated,info=env.step(queue.popleft());steps+=1
                simulation_seconds+=time.perf_counter()-tick
                if video is not None: video.append(obs,steps)
                # FailAwareWrapper returns obs=None on simulator errors.
                if terminated or truncated or (info or {}).get('status')=='error':break
                for key in obs:inputs[key].extend(obs[key])
                if steps%160==0:
                    print('ROBOMME_PROGRESS '+json.dumps(dict(episode=episode,steps=steps,
                          decisions=decisions,elapsed_seconds=round(time.perf_counter()-episode_started,1),
                          policy_seconds=round(policy_seconds,1),simulation_seconds=round(simulation_seconds,1))),flush=True)
            status=str((info or {}).get('status','unknown'))
            row=dict(episode=episode,steps=steps,decisions=decisions,status=status,success=status=='success',
                     elapsed_seconds=round(time.perf_counter()-episode_started,3),
                     reset_seconds=round(reset_seconds,3),policy_seconds=round(policy_seconds,3),
                     simulation_seconds=round(simulation_seconds,3))
            row['task_goal']=goal
            if video is not None: row['video']=video.close(complete=True)
            results.append(row);print('ROBOMME_EPISODE '+json.dumps(row),flush=True)
        finally:
            try:
                if video is not None: video.close()
            finally: env.close()
        result=dict(task=a.task,split=a.split,max_steps=a.max_steps,contract=contract,
                    save_video=a.save_video,episodes=len(results),successes=sum(r['success'] for r in results),results=results)
        temporary=output/'summary.json.tmp'
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(output/'summary.json')
    assert client.reset()['reset_finished']


if __name__=='__main__':main()
