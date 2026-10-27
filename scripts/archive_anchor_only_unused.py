"""Archive the eight explicitly selected anchor-only checkpoints; retain eval weights."""
import argparse
import fcntl
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/ninetask_fastwam-anchor_only-no-robotwininit_histaugtrue_bs4_ga4_gpu8_ep500_save5000_lr0.0001_cosine_20260929_185233'
KEEP = {68500, 60000, 50000, 40000, 30000, 65000}
ARCHIVE = [5000, 10000, 15000, 20000, 25000, 35000, 45000, 55000]
STATE = ROOT / 'runtime/anchor_only_tos_archive_20261005'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def local_links(step):
    if step in KEEP or step not in ARCHIVE:
        raise ValueError('Checkpoint is not authorized for archival')
    candidates = [RUN / f'tos_payload/step_{step:06d}/weights.pt',
                  RUN / f'checkpoints/weights/step_{step:06d}.pt']
    paths = []
    for path in candidates:
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError('Unexpected symlink: ' + str(path))
        if path.exists():
            paths.append(path)
    if not paths:
        return [], None
    st = paths[0].stat()
    if any(identity(p.stat()) != identity(st) for p in paths):
        raise RuntimeError('Payload and primary weight differ')
    if st.st_nlink != len(paths):
        raise RuntimeError('Unaccounted hardlinks; retain checkpoint')
    return paths, st


def ensure_not_used(step):
    if step in KEEP:
        raise RuntimeError('Evaluation checkpoint is protected')
    campaign = ROOT / 'eval_launch_logs/anchor_only_157_20261004/config.json'
    if campaign.exists():
        groups = json.loads(campaign.read_text())['groups']
        for spec in groups.values():
            if Path(spec['checkpoint']) == RUN / f'checkpoints/weights/step_{step:06d}.pt':
                raise RuntimeError('Checkpoint appears in evaluation manifest')
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if str(RUN) in cmd and f'step_{step:06d}.pt' in cmd:
            raise RuntimeError('Checkpoint referenced by process ' + proc.name)


def process(store, uploader, backend, step):
    receipt = STATE / f'step_{step:06d}.json'
    paths, before = local_links(step)
    if not paths:
        if receipt.exists():
            data = json.loads(receipt.read_text())
            if data.get('status') == 'verified_pending_delete':
                data.update(status='deleted', recovered_at=time.time())
                atomic(receipt, data)
            return
        raise RuntimeError('Local weights missing without archival receipt')
    ensure_not_used(step)
    primary = RUN / f'checkpoints/weights/step_{step:06d}.pt'
    payload = RUN / f'tos_payload/step_{step:06d}'
    if primary.exists():
        uploader.stage_payload(RUN, primary)
    logging.info('VERIFY_OR_UPLOAD step=%s bytes=%s', step, before.st_size)
    result = store.upload_checkpoint(payload, run_name=RUN.name, checkpoint_id=step,
                                     source_world_size=8, state_root=STATE / 'upload_state' / str(step))
    verified = store.verify_checkpoint(RUN.name, step)
    if result['manifest_sha256'] != verified['manifest_sha256']:
        raise RuntimeError('Remote manifest changed')
    manifest = store._parse_manifest(store._get_manifest_bytes(RUN.name, step, missing_ok=False))
    if any(not isinstance(entry.get('crc64_ecma'), int) for entry in manifest['files'].values()):
        raise RuntimeError('Remote CRC64 evidence missing')
    # upload_checkpoint compares fresh local SHA256 to the committed remote
    # manifest, then verify_checkpoint checks current remote sizes and CRC64.
    ensure_not_used(step)
    paths, after = local_links(step)
    if identity(before) != identity(after):
        raise RuntimeError('Local weight changed during upload')
    data = dict(step=step, run=str(RUN), verified_at=time.time(), remote=verified,
                remote_manifest=manifest, local_paths=[str(p) for p in paths],
                allocated_bytes=after.st_blocks * 512, status='verified_pending_delete')
    atomic(receipt, data)
    for path in paths:
        if identity(path.stat()) != identity(after):
            raise RuntimeError('Cleanup target changed')
        path.unlink()
        logging.info('UNLINK %s', path)
    data.update(status='deleted', deleted_at=time.time())
    atomic(receipt, data)
    logging.info('ARCHIVED_AND_DELETED step=%s bytes=%s destination=%s',
                 step, data['allocated_bytes'], verified['destination'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    assert RUN.resolve() == RUN and RUN.is_dir()
    STATE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.plan:
        print(json.dumps({'keep': sorted(KEEP), 'archive': [dict(step=s, paths=[str(p) for p in local_links(s)[0]]) for s in ARCHIVE]}, indent=2))
        return
    with (STATE / 'archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend = load_module('anchor_archive_backend', ROOT.parent / 'starwam/starwam/tools/checkpoint_tos/backend.py')
        uploader = load_module('anchor_archive_payload', ROOT / 'scripts/upload_checkpoints.py')
        store = backend.TosCheckpointStore(backend.TosUploadConfig(
            endpoint=os.environ.get('TOS_ENDPOINT', 'https://tos-cn-beijing.ivolces.com'),
            region=os.environ.get('TOS_REGION', 'cn-beijing'), bucket=os.environ['TOS_BUCKET'],
            prefix=os.environ['TOS_PREFIX']))
        store.check_access()
        report = {}
        for step in ARCHIVE:
            try:
                process(store, uploader, backend, step)
                report[str(step)] = {'status': 'deleted'}
            except Exception as exc:
                logging.exception('KEEP_LOCAL step=%s', step)
                report[str(step)] = {'status': 'error', 'error': str(exc)}
            atomic(STATE / 'status.json', {'updated': time.time(), 'checkpoints': report})
        remaining = sorted(int(p.stem[5:]) for p in (RUN / 'checkpoints/weights').glob('step_*.pt'))
        if not KEEP.issubset(remaining):
            raise RuntimeError('Protected evaluation checkpoint missing')
        logging.info('FINISHED remaining_steps=%s', remaining)
        if any(v['status'] == 'error' for v in report.values()):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
