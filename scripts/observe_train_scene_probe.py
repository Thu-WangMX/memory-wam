import os,sys,json,importlib
from pathlib import Path
import types,cv2,h5py,numpy as np
def main():
    a=types.SimpleNamespace(task="observe_and_pickup",output='/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/data_audits/observe_20260920/train_scene_replay')
    root=Path(__file__).resolve().parents[1];sim=root/'eval_runtime/RMBench';out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    os.chdir(sim);sys.path.insert(0,str(sim))
    for k in ['RMBENCH_BATTERY_CONTACT_OFFSET','RMBENCH_BATTERY_TABLE_FRICTION','RMBENCH_EVAL_VIDEO_CAMERA']:os.environ.pop(k,None)
    import yaml
    from envs import CONFIGS_PATH
    cfg=yaml.safe_load(Path('task_config/demo_clean.yml').read_text())
    emb=yaml.safe_load((Path(CONFIGS_PATH)/'_embodiment_config.yml').read_text())
    robot=emb[cfg['embodiment'][0]]['file_path'];rcfg=yaml.safe_load((Path(robot)/'config.yml').read_text())
    cams=yaml.safe_load((Path(CONFIGS_PATH)/'_camera_config.yml').read_text());cam=cams[cfg['camera']['head_camera_type']]
    cfg.update(task_name=a.task,task_config='demo_clean',left_robot_file=robot,right_robot_file=robot,dual_arm_embodied=True,left_embodiment_config=rcfg,right_embodiment_config=rcfg,head_camera_h=cam['h'],head_camera_w=cam['w'],eval_mode=True,save_data=False,eval_video_log=False,render_freq=0,save_path=str(out/'scratch'))
    cls=getattr(importlib.import_module('envs.'+a.task),a.task)
    raw=Path('/mnt/vepfs02/datasets/kevin_wang/code/projects/RMBench/data/observe_and_pickup/demo_clean/data')
    report=[]
    for episode,seed in [(0,0),(1,1),(10,10)]:
        env=cls()
        env.setup_demo(now_ep_num=episode,seed=seed,is_test=True,**cfg)
        obs=env.get_obs();rgb=obs['observation']['head_camera']['rgb']
        with h5py.File(raw/f'episode{episode}.hdf5','r') as f:
            b=f['observation/head_camera/rgb'][0]
            original=cv2.imdecode(np.frombuffer(b,dtype=np.uint8),cv2.IMREAD_COLOR)
        generated=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out/f'episode{episode}_original.png'),original)
        cv2.imwrite(str(out/f'episode{episode}_recreated.png'),generated)
        if generated.shape!=original.shape:generated=cv2.resize(generated,(original.shape[1],original.shape[0]))
        cv2.imwrite(str(out/f'episode{episode}_comparison.png'),np.concatenate([original,generated],axis=1))
        row=dict(episode=episode,seed=seed,mae=float(np.abs(original.astype(float)-generated).mean()),models=env.object_modelnames,model_ids=[int(x) for x in env.object_ids],target=int(env.target_object_idx),poses=[x.get_pose().p.tolist() for x in env.object])
        report.append(row);print('SCENE',json.dumps(row),flush=True)
        (out/'scene_comparison.json').write_text(json.dumps(report,indent=2))
        env.close_env(clear_cache=True)
if __name__=='__main__':main()
