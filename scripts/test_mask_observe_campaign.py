import tempfile
from pathlib import Path
from mask_observe_eval_campaign import CampaignQueue, command


def complete(queue,g,n,successes):
    for i in range(n):
        lease=queue.call('claim',dict(group=g,owner='test'))
        assert lease['state']=='leased'
        queue.call('finish',dict(group=g,seed=lease['seed'],token=lease['token'],success=i<successes))


with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    groups={f'step_{s:06d}':dict(step=s,seeds=[100000],completed=[]) for s in (2000,1500,1000,500)}
    q=CampaignQueue(root/'db.sqlite',groups)
    for g in groups:complete(q,g,30,15)
    state=q.state()
    assert state['phase']=='extend100'
    assert state['selected']==['step_002000','step_001500','step_001000']
    for g in state['selected']:
        lease=q.call('claim',dict(group=g,owner='test'))
        assert lease['seed']==100030
        q.call('release',dict(group=g,seed=lease['seed'],token=lease['token']))
        complete(q,g,70,20)
    assert q.state()['phase']=='complete'
    result=q.snapshot(root)
    assert len(result['episodes'])==330
    q2=CampaignQueue(root/'db.sqlite',groups)
    assert q2.state()['phase']=='complete'
    for g in groups:
        assert q2.call('claim',dict(group=g,owner='test'))['state']=='complete'
    cmd,env,log=command('step_008000',1,1,root,18965,'test-token')
    assert '/root/kevin_wang' not in env['COMPARISON_VENV']
    assert env['EVAL_STATS_PATH'].endswith('/dataset_stats.json')
    assert env['RMBENCH_QUEUE_GROUP']=='step_008000'
print('PASS: stage30, top3 tie-break, unique seeds 30+70, exact quotas, resume, environment')
