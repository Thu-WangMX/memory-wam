"""Press Button 62.5k, denoise10/replan16, 100 matched baseline seeds."""
import pathlib,json,sqlite3,os,sys,time,subprocess,secrets
import eval_top100_campaign as base
R=base.R
base.NAME='press62500_d10_r16_100_20260928'
base.OUT=R/'eval_launch_logs'/base.NAME
base.URL='http://192.168.52.208:39877'
def prepare():
 out=base.OUT;out.mkdir(exist_ok=True);assert not (out/'config.json').exists()
 prior=R/'eval_launch_logs/ninetask_top100_20260927';old=json.loads((prior/'config.json').read_text());g='press_button_step062500_d50';s=dict(old['groups'][g])
 with sqlite3.connect('file:'+str(prior/'seeds.sqlite.backup')+'?mode=ro',uri=True) as db:
  records=[dict(seed=seed,success=bool(ok)) for seed,ok in db.execute("SELECT s,success FROM jobs WHERE g=? AND state='done' ORDER BY s",(g,))]
 assert len(records)==100 and sum(x['success'] for x in records)==64
 assert [x['seed'] for x in records]==list(range(100000,100100))
 assert pathlib.Path(s['checkpoint']).stat().st_size>10_000_000_000
 s.update(denoise_steps=10,action_horizon=16,replan_steps=16,target=100,seeds=[x['seed'] for x in records],completed=[],invalid_seeds=[],env=dict(s.get('env',{})))
 cfg=dict(root=str(R),output=str(out),db='/tmp/'+base.NAME+'/seeds.sqlite',queue_url=base.URL,port=39877,token=secrets.token_hex(24),groups={'press_button_step062500_d10_r16':s})
 base.atomic(out/'config.json',cfg);os.chown(out/'config.json',0,1397203185);(out/'config.json').chmod(0o640)
 base.atomic(out/'baseline_50denoise_replan16.json',dict(records=records,success=64,total=100,source=str(prior)))
 print('Prepared 62.5k denoise10 replan16, 100 baseline seeds, no completed results copied')
def report():
 while True:
  st=json.loads((base.OUT/'manager_status.json').read_text());v=st['groups']['press_button_step062500_d10_r16'];n=v.get('done',0);ok=v.get('success',0);rate=f'{ok/n:.1%}' if n else 'pending'
  lines=['# Press Button 62.5k denoising comparison','', '| Denoising | Replan | Horizon | Success / completed | Rate |','|---:|---:|---:|---:|---:|','| 50 (baseline) | 16 | 16 | 64/100 | 64% |',f'| 10 | 16 | 16 | {ok}/{n} | {rate} |','','Original executor and physics; same scene seeds 100000..100099.']
  p=base.OUT/'results.md';t=p.with_suffix('.tmp');t.write_text('\n'.join(lines)+'\n');t.replace(p)
  if n>=v['target']:return
  time.sleep(30)
def launch(mode):
 folder=base.OUT if mode in ('coordinator','report') else R/('eval_a100_runtime/campaigns' if mode=='a100' else 'eval_launch_logs')/(base.NAME+'_'+mode)
 folder.mkdir(exist_ok=True)
 with (folder/(mode+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-u',str(R/'scripts/eval_press62500_d10.py'),mode],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (folder/(mode+'.pid')).write_text(str(p.pid));print('START',mode,p.pid,flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='launch':launch(sys.argv[2])
 elif mode=='coordinator':base.coordinator()
 elif mode=='report':report()
 else:base.node(mode)
