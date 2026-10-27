"""Read local coordinator database and atomically publish recovery records on shared disk."""
import pathlib,sqlite3,json,time,sys
cfg=json.load(open(sys.argv[1]));out=pathlib.Path(cfg['output']);db=sqlite3.connect('file:'+cfg['db']+'?mode=ro',uri=True)
while True:
 rows=db.execute('SELECT g,s,state,success,detail FROM jobs ORDER BY g,s').fetchall()
 records=[dict(group=g,scene_seed=s,state=state,success=success,detail=json.loads(detail) if detail else None) for g,s,state,success,detail in rows]
 tmp=out/'seed_records.tmp';tmp.write_text(json.dumps(records,indent=2));tmp.replace(out/'seed_records.json');time.sleep(30)
