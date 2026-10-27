import os, sys, json, argparse, hashlib, importlib, datetime, traceback
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--output',required=True);ap.add_argument('--count',type=int,default=100);a=ap.parse_args()
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
    files=[sim/'envs/_base_task.py',sim/'envs'/f'{a.task}.py',sim/'envs/robot/robot.py',sim/'task_config/demo_clean.yml',Path(robot)/'config.yml']
    meta={'task':a.task,'count_target':a.count,'start_seed':100000,'physics':'default','method':'expert plan_success and check_success, then rollout initialization','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'config':cfg,'accepted':[],'attempted':0,'status':'running'}
    target=out/(a.task+'.json');assert not target.exists(),'Existing output: choose another directory'
    def save():
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(meta,indent=2,default=str));tmp.replace(target)
    save()
    for seed in range(100000,110000):
        env=cls();ok=False;reason='';phase='expert'
        try:
            env.setup_demo(now_ep_num=len(meta['accepted']),seed=seed,is_test=True,**cfg)
            env.play_once()
            ok=bool(env.plan_success and env.check_success())
            if not ok:reason='expert_failed'
            env.close_env(clear_cache=True)
            if ok:
                phase='rollout_init'
                env.setup_demo(now_ep_num=len(meta['accepted']),seed=seed,is_test=True,**cfg)
        except (UnicodeError, ImportError, FileNotFoundError):
            raise
        except Exception as exc:
            ok=False;reason=phase+': '+type(exc).__name__+': '+str(exc)
        finally:
            try:env.close_env(clear_cache=True)
            except Exception:pass
        meta['attempted']+=1
        if ok:meta['accepted'].append(seed)
        with (out/(a.task+'_attempts.jsonl')).open('a') as f:f.write(json.dumps({'seed':seed,'accepted':ok,'reason':reason})+'\n')
        if len(meta['accepted'])==a.count:meta['status']='complete'
        save();print('SCREEN',a.task,seed,ok,len(meta['accepted']),flush=True)
        if meta['status']=='complete':
            (out/(a.task+'_seeds.json')).write_text(json.dumps(meta['accepted']));return
    raise RuntimeError('Seed search exhausted')

if __name__=='__main__':main()
