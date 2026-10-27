# Anchor-Helios on RoboDojo

This adapter runs the existing FastWAM Anchor-Helios model on RoboDojo MP4/Parquet data and the existing RoboDojo IsaacLab/XPolicyLab simulator. The original RMBench and StarWAM model/training files are unchanged.

## Model and data contract

- Original Anchor-Helios 16/2/1 history plus two episode anchors; continuous causal episode VAE encoding.
- Action horizon and replanning interval 16; RGB stride 4. Observe every executed action, including actions inside a chunk.
- Wrists above head, 384x320 mosaic; 14 joint actions and 14 proprio values.
- Original FastWAM normalization and text conditioning semantics. Shared StarWAM action statistics are converted without recomputing them; text is padded to 128 tokens.
- Default scope: cover_blocks, imitate_sorting_sequence, match_and_pick_from_conveyor, press_by_number, swap_T, swap_blocks (600 episodes before split).
- Seed-42 train/validation split is computed on the full dataset before task filtering. This training split seed is separate from simulator layout seeds (installed layouts: 0, 1, 2).
- This is a new Anchor-Helios recipe, not a reproduction of StarWAM's 32-action/24-step recipe. Formal module ablation needs a matched no-memory control; comparing the two repositories alone does not isolate memory.

## Prepare and train

Run from `/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison`:

```bash
source experiments/robodojo/env.sh
python -B scripts/prepare_robodojo_helios.py assets \
  --root "$ROBODOJO_DATA_ROOT" --source-prepared "$ROBODOJO_RUNTIME/prepared" \
  --output "$ROBODOJO_HELIOS_PREPARED"

# Select available GPUs explicitly. One process per selected GPU; resumable shards.
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
  scripts/prepare_robodojo_helios.py latents --root "$ROBODOJO_DATA_ROOT" \
  --output "$ROBODOJO_HELIOS_LATENTS"

# Requires complete caches for the selected episodes; does not load/train the model.
ROBODOJO_PREFLIGHT_ONLY=1 bash experiments/robodojo/train.sh
bash experiments/robodojo/train.sh --config

# Example for a reserved 8-GPU training allocation, not for currently busy GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
  bash experiments/robodojo/train.sh
```

Default budget is 20,000 optimizer steps, per-device batch 1, accumulation 1, save every 2,500. Override Hydra parameters as needed, e.g. `max_steps=1000 save_every=500`. Wan and ActionDiT initialization use the existing FastWAM assets; no RMBench-trained backbone checkpoint is loaded. Generic validation is disabled because it does not supply Helios history. Evaluate saved checkpoints with the simulator below. Full optimizer/distributed training has not been validated by the bounded GPU smoke.

Assets for all 35 tasks are prepared. Only episode 300 has been encoded into the separate `prepared-anchor-helios/latents-smoke-cover` directory during verification. The full six-task cache and formal training have **not** been run. For the verified one-episode CPU preflight:

```bash
ROBODOJO_HELIOS_LATENTS="$ROBODOJO_HELIOS_PREPARED/latents-smoke-cover" \
ROBODOJO_PREFLIGHT_ONLY=1 bash experiments/robodojo/train.sh \
  data.train.tasks=[cover_blocks] data.train.episode_indices=[300] data.train.split=all
```

## Serve and evaluate

Copy `deployment.example.json` to a new file; set `config_path` to the saved training configuration and `checkpoint_path` to its checkpoint. Keep 50 inference steps for normal evaluation; the smoke uses two only to check wiring.

```bash
source experiments/robodojo/env.sh
python -B experiments/robodojo/install_client.py --stack "$ROBODOJO_RUNTIME/eval-stack"
CUDA_VISIBLE_DEVICES=4 bash experiments/robodojo/serve.sh \
  --deployment /absolute/path/deployment.json --host 127.0.0.1 --port 8096
```

In another shell on the same host:

```bash
SIM_GPU_ID=5 POLICY_PORT=8096 EVAL_SEED=0 EVAL_TASK=cover_blocks EVAL_NUM=10 \
  EVAL_RUN_ID=anchor_helios_checkpoint_identity_seed0 \
  bash experiments/robodojo/eval_client.sh
```

Use a fresh run identity per checkpoint/task/seed. Deployment is single-environment only; each concurrent environment needs its own policy state/server. Installer adds `FastWAMAnchorHelios` and refuses conflicting files. Online history is reset at episode boundaries. The server retains sampled RGB history and re-encodes the prefix on each decision to preserve the original inference contract; memory/latency grow with episode length.

## Verification

```bash
source experiments/robodojo/env.sh
PYTHONPATH="$ROBODOJO_RUNTIME/eval-stack:$ROBODOJO_RUNTIME/eval-stack/XPolicyLab:$ROBODOJO_RUNTIME/eval-deps:$PYTHONPATH" \
  python -B -m unittest discover -s tests -p test_robodojo_helios.py -v
```

CPU tests cover camera layout, action/observation sequencing, history reset, missing observations and real WebSocket transport. Real episode 300 checks cover action alignment, terminal padding and cached history. GPU checks on a 4090D verify VAE prefix parity at frames 0/16/32 (max difference 0), a full model backward pass with finite gradients including long/mid history paths, two action decisions and strict checkpoint reload.

The backward smoke releases each parameter gradient after checking it to fit a 24-GiB card; it does not perform an optimizer update. `migration/robodojo_helios_20260922/model-smoke/step_000000_smoke_only.pt` is **untrained** and must not be used to report benchmark performance.

For an isolated 8-control-step simulator wiring check, choose two free GPUs and a fresh output directory. The untrained checkpoint ended the attempted 32-step episode at step 14; that attempt is not a benchmark measurement and did not pass the 32-step check.

```bash
python -B scripts/run_robodojo_rollout_smoke.py \
  --deployment migration/robodojo_helios_20260922/model-smoke/deployment.json \
  --output migration/robodojo_helios_20260922/rollout-smoke-new \
  --policy-gpu 4 --sim-gpu 5 --steps 8
```

This supervisor stops its own server/client after completion or failure. A successful smoke writes `simulator_smoke.json` with `benchmark_result: false` and exits before score aggregation. Reports and logs are retained under `migration/robodojo_helios_20260922/`.
