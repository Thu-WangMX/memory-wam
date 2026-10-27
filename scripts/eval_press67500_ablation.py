import pathlib,json,sqlite3,os,sys,time,subprocess,secrets
import eval_top100_campaign as base
R=base.R
base.NAME='press67500_ablation100_20260928'
base.OUT=R/'eval_launch_logs'/base.NAME
base.URL='http://192.168.52.208:39876'
def prepare():
 out=base.OUT;out.mkdir(exist_ok=True);assert not (out/'config.json').exists()
 prior=R/'eval_launch_logs/ninetask_top100_20260927';old=json.loads((prior/'config.json').read_text())
 g='press_button_step067500_d50';spec=old['groups'][g]
 with sqlite3.connect('file:'+str(prior/'seeds.sqlite.backup')+'?mode=ro',uri=True) as db:
  records=[dict(seed=s,success=bool(ok)) for s,ok in db.execute("SELECT s,success FROM jobs WHERE g=? AND state='done' ORDER BY s",(g,))]
 assert len(records)==100 and sum(x['success'] for x in records)==65
 assert pathlib.Path(spec['checkpoint']).stat().st_size>10_000_000_000
 groups={}
 for denoise,replan in [(10,16),(50,8)]:
  s=dict(spec);s.update(denoise_steps=denoise,action_horizon=16,replan_steps=replan,target=100,seeds=[x['seed'] for x in records],completed=[],invalid_seeds=[],env=dict(spec.get('env',{})))
  groups[f'press_button_step067500_d{denoise}_r{replan}']=s
 cfg=dict(root=str(R),output=str(out),db='/tmp/'+base.NAME+'/seeds.sqlite',queue_url=base.URL,port=39876,token=secrets.token_hex(24),groups=groups)
 base.atomic(out/'config.json',cfg);os.chown(out/'config.json',0,1397203185);(out/'config.json').chmod(0o640)
 base.atomic(out/'baseline_50denoise_replan16.json',dict(records=records,success=65,total=100,source=str(prior)))
 print('Prepared two independent 100-episode evaluations using baseline seeds 100000..100099')
def report():
 while True:
  st=json.loads((base.OUT/'manager_status.json').read_text());cfg=json.loads((base.OUT/'config.json').read_text())
  lines=['# Press Button 67.5k ablations','', '| Denoising | Replan | Horizon | Success / completed | Rate |','|---:|---:|---:|---:|---:|','| 50 (baseline) | 16 | 16 | 65/100 | 65% |']
  for g,s in cfg['groups'].items():
   v=st['groups'][g];n=v.get('done',0);ok=v.get('success',0);rate=f'{ok/n:.1%}' if n else 'pending'
   lines.append(f'| {s["denoise_steps"]} | {s["replan_steps"]} | 16 | {ok}/{n} | {rate} |')
  lines+=['','Both ablations use the same 100 baseline scene seeds; original executor and physics.','Replan8 uses existing RGB stride2, so latent temporal spacing changes as well.']
  p=base.OUT/'results.md';t=p.with_suffix('.tmp');t.write_text('\n'.join(lines)+'\n');t.replace(p)
  if all(v.get('done',0)>=v['target'] for v in st['groups'].values()):return
  time.sleep(30)
def launch(mode):
 folder=base.OUT if mode in ('coordinator','report') else R/('eval_a100_runtime/campaigns' if mode=='a100' else 'eval_launch_logs')/(base.NAME+'_'+mode)
 folder.mkdir(exist_ok=True)
 with (folder/(mode+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-u',str(R/'scripts/eval_press67500_ablation.py'),mode],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (folder/(mode+'.pid')).write_text(str(p.pid));print('START',mode,p.pid,flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='launch':launch(sys.argv[2])
 elif mode=='coordinator':base.coordinator()
 elif mode=='report':report()
 else:base.node(mode)
