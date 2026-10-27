import shutil,subprocess,shlex
from pathlib import Path
import pytest
from test_robomme_flow_launch import sandbox


@pytest.mark.parametrize('variant,compression,task,patch',[
    ('past32','temporal','robomme_helios_past32_flow','[4,8,8]'),
    ('pastfuture32','temporal','robomme_helios_pastfuture32_flow','[4,8,8]'),
    ('pastfuture32_light','temporal','robomme_helios_pastfuture32_light_flow','[2,8,8]'),
    ('pastfuture32_light','spatial','robomme_helios_pastfuture32_light_flow','[4,4,4]')])
def test_all_four_workers_launch_same_variant(sandbox,variant,compression,task,patch):
    source=Path(__file__).resolve().parents[1]/'experiments/robomme/train_horizon32.sh'
    wrapper=sandbox.root/'experiments/robomme/train_horizon32.sh';shutil.copyfile(source,wrapper)
    names=[]
    for rank in range(4):
        env=dict(sandbox.env,MLP_ROLE_INDEX=str(rank),MLP_TASK_ID='horizon32test',
                 ROBOMME_LAUNCH_DRY_RUN='1',HOSTNAME=f'pod-{rank}',ROBOMME_LONG_COMPRESSION=compression)
        proc=subprocess.Popen(['bash',str(wrapper),variant],env=env,cwd=sandbox.root,
                              text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        code,output=sandbox.finish(proc);assert code==0,output
        args=shlex.split(output)
        assert 'task='+task in args and 'model.helios_config.patch_long='+patch in args
        assert args[args.index('--machine_rank')+1]==str(rank)
        assert args[args.index('--num_processes')+1]=='32'
        assert 'batch_size=16' in args and 'gradient_accumulation_steps=1' in args
        names.append(next(s for s in args if s.startswith('wandb.name=')))
    assert len(set(names))==1 and not sandbox.events()
