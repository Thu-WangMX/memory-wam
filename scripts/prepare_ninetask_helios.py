"""Create read-only references to the nine existing RMBench caches."""
import json
import os
from pathlib import Path
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "runtime/ninetask_helios"
TASKS = "observe_and_pickup rearrange_blocks put_back_block swap_blocks swap_T battery_try blocks_ranking_try cover_blocks press_button".split()
KEYS = ["RMBENCH_COVER_BLOCKS_LEROBOT", "MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS", "MEMORYWAM_COVER_BLOCKS_TEXT_CACHE"]
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "text_cache").mkdir(exist_ok=True)
episodes, records, metadata, text_differences = {}, [], None, []
for task in TASKS:
    env = os.environ.copy()
    for key in KEYS + ["RMBENCH_TASK_ROOT", "MEMORYWAM_COVER_BLOCKS_STATS"]:
        env.pop(key, None)
    env["RMBENCH_TASK"] = task
    import shlex
    query = "import os,json; print(json.dumps([os.environ[k] for k in " + repr(KEYS) + "]))"
    values = json.loads(subprocess.check_output(["bash", "-c", "source scripts/comparison_env.sh && python -c " + shlex.quote(query)], cwd=ROOT, env=env, text=True))
    data, cache, text = map(Path, values)
    manifest = json.loads((cache / "manifest.json").read_text())
    if metadata is None:
        metadata = dict(manifest["metadata"])
    for key in ["schema_version", "replan_stride", "latent_shape", "latent_dtype", "mosaic", "color_contract", "encoding", "vae_model_sha256"]:
        assert manifest["metadata"].get(key) == metadata.get(key), (task, key, manifest["metadata"].get(key), metadata.get(key))
    for key, value in manifest["episodes"].items():
        assert key not in episodes, key
        source = (cache / value).resolve()
        assert source.is_file(), source
        episodes[key] = str(source)
    for source in text.glob("*.pt"):
        target = DEST / "text_cache" / source.name
        if target.exists():
            if target.resolve() != source.resolve():
                a = torch.load(target, map_location="cpu", weights_only=True)
                b = torch.load(source, map_location="cpu", weights_only=True)
                assert a.keys() == b.keys(), source
                if not all(torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k] for k in a):
                    assert torch.equal(a['mask'], b['mask']) and a['context'].shape == b['context'].shape, source
                    difference = (a['context'].float() - b['context'].float()).abs()
                    text_differences.append(dict(prompt_hash=source.name, selected=str(target.resolve()), alternative=str(source), max_abs=float(difference.max()), mean_abs=float(difference.mean())))
        else:
            target.symlink_to(source)
    info = json.loads((data / "meta/info.json").read_text())
    records.append(dict(task=task, dataset=str(data), cache=str(cache), text_cache=str(text), episodes=info["total_episodes"], frames=info["total_frames"]))
    print(task, records[-1]["episodes"], records[-1]["frames"], flush=True)
metadata["episode_count"] = len(episodes)
(DEST / "manifest.json").write_text(json.dumps(dict(metadata=metadata, episodes=episodes), indent=2) + "\n")
(DEST / "tasks.json").write_text(json.dumps(records, indent=2) + "\n")
(DEST / "text_cache_provenance.json").write_text(json.dumps(text_differences, indent=2) + "\n")
print("Prepared", len(episodes), "episodes; no source data modified")
