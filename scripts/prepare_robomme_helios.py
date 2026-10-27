"""Copy small audited assets into our namespace; never modify Kevin's data."""
import argparse
import json
from pathlib import Path
import os
import shutil
import torch
from fastwam.datasets.robomme import RoboMMEHeliosDataset, normalizers
from fastwam.datasets.robodojo import sha256, read_jsonl, text_filename, load_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-all', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = Path(os.environ['ROBOMME_KEVIN_DATA']).resolve()
    output = Path(os.environ['ROBOMME_HELIOS_PREPARED']).resolve()
    if output == source or source in output.parents:
        raise ValueError('Prepared output must be outside Kevin data')
    root, cache = Path(os.environ['ROBOMME_DATA_ROOT']), Path(os.environ['ROBOMME_HELIOS_LATENTS'])
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((source/'stats/dataset_stats.json').read_text())
    stats = {f: {'default': {k: v[:8] for k,v in stats[f]['default'].items() if k.startswith('global_')}}
             for f in ('action', 'state')}
    encoded = json.dumps(stats, indent=2)+'\n'
    stats_path = output/'dataset_stats.json'
    if stats_path.exists() and stats_path.read_text() != encoded:
        raise ValueError('Existing prepared normalization differs; choose a fresh output')
    stats_path.write_text(encoded)
    normalizers(stats_path)
    (output/'text_cache').mkdir(exist_ok=True)
    text_hashes = {}
    for row in read_jsonl(root/'meta/tasks.jsonl'):
        filename = text_filename(row['task'])
        src, dst = source/'text_cache'/filename, output/'text_cache'/filename
        if dst.exists() and sha256(dst) != sha256(src):
            raise ValueError('Existing text asset differs')
        if not dst.exists():
            shutil.copyfile(src, dst)
        load_context(output/'text_cache', row['task'])
        text_hashes[filename] = sha256(dst)
    provenance = dict(info_sha256=sha256(root/'meta/info.json'),
                      episodes_sha256=sha256(root/'meta/episodes.jsonl'),
                      tasks_sha256=sha256(root/'meta/tasks.jsonl'),
                      latents_sha256=sha256(cache/'manifest.json'), stats_sha256=sha256(stats_path),
                      text_sha256=text_hashes, source=str(source), action_dimensions=list(range(8)))
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    dataset = RoboMMEHeliosDataset(root, output, cache)
    report = dict(episodes=len(dataset.episodes), decisions=len(dataset), tasks=dataset.tasks)
    if args.audit_all:
        for i,row in enumerate(dataset.episodes):
            dataset._table(row)
            dataset.load_latents(row)
            if i % 100 == 0:
                print(f'AUDIT {i}/1600', flush=True)
        report['all_parquets_and_latents_checked'] = True
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
