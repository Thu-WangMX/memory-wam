"""Delete only explicitly selected weight files from this completed run."""
import argparse
import json
import os
from pathlib import Path
import re
import stat
from collections import Counter

ROOT=Path('/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison')
RUN=ROOT/'runs/observe_and_pickup_helios_futuremaskaux_robotwininit_bs4_ga1_gpu8_ep500_lr1e-4_20261003_053337'
KEEP={1500,6500}
AUDIT=RUN/'checkpoint_cleanup_20261007.json'

def inspect():
    assert RUN.is_dir() and RUN.resolve()==RUN
    rows=[]
    for folder,pattern in ((RUN/'checkpoints/weights','step_*.pt'),(RUN/'tos_payload','step_*/weights.pt')):
        assert folder.resolve()==folder
        for path in sorted(folder.glob(pattern)):
            assert path.resolve().is_relative_to(RUN) and path.resolve()==path
            st=path.lstat()
            assert stat.S_ISREG(st.st_mode)
            name=path.stem if path.parent==RUN/'checkpoints/weights' else path.parent.name
            match=re.fullmatch(r'step_(\d{6})',name)
            assert match,name
            step=int(match[1])
            assert step in range(500,8001,500)
            rows.append(dict(path=str(path),step=step,delete=step not in KEEP,
                bytes=st.st_size,allocated_bytes=st.st_blocks*512,
                device=st.st_dev,inode=st.st_ino,nlink=st.st_nlink,mtime_ns=st.st_mtime_ns))
    for step in KEEP:
        assert any(r['path']==str(RUN/'checkpoints/weights'/f'step_{step:06d}.pt') for r in rows)
    return rows

def active_references(rows):
    names={f'step_{r["step"]:06d}.pt' for r in rows if r['delete']}
    hits=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:cmd=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,PermissionError):continue
        if str(RUN) in cmd and any(n in cmd for n in names):hits.append(dict(pid=int(entry.name),command=cmd))
    return hits

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
    rows=inspect();delete=[r for r in rows if r['delete']]
    hits=active_references(rows)
    counts=Counter((r['device'],r['inode']) for r in delete)
    unique={(r['device'],r['inode']):r for r in delete}
    estimated=sum(r['allocated_bytes'] for key,r in unique.items() if counts[key]==r['nlink'])
    report=dict(run=str(RUN),keep_steps=sorted(KEEP),files=rows,active_references=hits,
                delete_logical_steps=sorted({r['step'] for r in delete}),
                delete_file_entries=len(delete),estimated_reclaimed_bytes=estimated,applied=False)
    assert not hits, f'Active references found: {hits}'
    if args.apply:
        previous=json.loads(AUDIT.read_text())
        assert not previous['applied'] and previous['files']==rows,'Files changed since audit'
        assert len(delete)==28,'Expected 14 original weights and 14 TOS staging entries'
        # Every target has been resolved/validated. No recursive directory deletion.
        for r in delete:Path(r['path']).unlink()
        for r in rows:
            path=Path(r['path'])
            if r['delete']:assert not path.exists()
            else:
                assert path.is_file() and path.stat().st_size==r['bytes']
        remaining=inspect()
        assert {r['step'] for r in remaining}==KEEP
        report.update(applied=True,remaining_files=remaining)
    AUDIT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','remaining_files')},indent=2))
    print('RETAINED',*[r['path'] for r in rows if not r['delete']],sep='\n')

if __name__=='__main__':main()
