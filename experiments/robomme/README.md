# FastWAM + Anchor + Helios on RoboMME

Remote repository: `/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison`.
The adapter keeps the existing two episode anchors, default history sizes `[16,2,1]`,
Helios spatial/temporal compression, video auxiliary loss, and action expert.
It does not use Kevin's event segmentation, KEMO writer, or trained weights.

## Data and environments

- Read-only source: `/mnt/vepfs02/output/kevin_wang/memorywam/data/robomme_all16_anchor1_event_k8_v1_full`.
- Parquet/video input: source + `/lerobot/multitask`.
- Observed latent cache: source + `/phase_latents`.
- Our normalization/text/provenance: `/mnt/vepfs01/output/spidy.wang/runtime-robomme/prepared-anchor-helios`.
- Training: `/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv` (Python 3.11, Torch 2.6).
- Simulation: `/mnt/vepfs02/output/kevin_wang/memorywam/robomme_native/.venv`.
- Protocol overlay: `/mnt/vepfs01/output/spidy.wang/runtime-robomme/python311`.

There are **16 semantic tasks, 100 demonstrations each, 1600 episodes**. The
116 entries in `tasks.jsonl` are language variants, not 116 benchmark tasks.
Execution-only sampling yields **30,470 decision samples**. The default is one
multitask model, with balanced task cycles, uniform episode sampling within a
task, and uniform decision sampling within an episode.

`source experiments/robomme/env.sh` sets the audited paths. Environment variables
can override them for another host. `volc-dev-157` has a local CUDA compiler.
The A100 host initially lacked one; a complete copy of volc's CUDA 12.2 toolkit
is now available at `/mnt/vepfs01/output/spidy.wang/runtime-robomme/cuda-12.2`.
The environment script selects it automatically when no local toolkit exists.
Native extension caches are separate by host. Neither system CUDA nor the
existing Python environments were replaced.

## Preparation (already completed on the shared disk)

```bash
cd /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
source experiments/robomme/env.sh
python -B scripts/install_robomme_protocol.py
python -B scripts/prepare_robomme_helios.py --audit-all
python -B scripts/check_robomme_helios.py
ROBOMME_PREFLIGHT_ONLY=1 python -B experiments/robomme/train.py task=robomme_helios
```

Preparation copies only normalization and text assets into our directory. It
does not rewrite Kevin's videos, Parquet files, latents, or environments. All
1600 action/state tables and latent payloads were checked for shape, finiteness,
frame/phase alignment, zero-padded dimensions and task coverage.

## Training

Run on an allocated GPU set. This command launches a fresh 16-task model,
20,000 optimizer steps, effective batch 32 (8 GPUs × microbatch 1 × accumulation 4):

```bash
cd /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
RUN_ID=robomme_anchor_helios_all16_seed42 \
bash experiments/robomme/train.sh \
  batch_size=1 gradient_accumulation_steps=4 max_steps=20000 seed=42
```

This is a starting recipe, not a claim that its learning rate/step count is
optimal. `data.train.tasks=[PatternLock]` selects a single semantic task;
`data.train.task_balanced=false` uses shuffled decision samples. The saved
`config.yaml`, `dataset_stats.json`, and checkpoint must travel together.

Use the provided ZeRO-1 launcher for normal training. Generic FSDP export in the
existing trainer does not carry Helios checkpoint metadata, so the RoboMME entry
rejects FSDP instead of producing an unloadable policy. The existing ZeRO-2 CPU
offload launcher can be used for a memory-constrained smoke test:

```bash
source experiments/robomme/env.sh
CUDA_VISIBLE_DEVICES=7 ROBOMME_GPU_MEMORY_GIB=30 \
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_cpuoffload.yaml \
  --num_processes 1 --main_process_port 29638 \
  experiments/robomme/train.py task=robomme_helios \
  output_dir=runs/robomme_one_step_check max_steps=1 \
  save_every=1 save_training_state=false num_workers=0 log_every=1
```

This smoke needs at least about 30 GiB of free device memory: DeepSpeed briefly
duplicates weights during optimizer initialization. An 18 GiB cap is insufficient,
even though the isolated gradient-release model smoke needs only 14.83 GiB.

For an actual resume, invoke `experiments/robomme/train.py` through the same
Accelerate configuration, with the saved recipe and `resume=<training-state-dir>`.
The fresh-run shell launcher intentionally refuses an existing run directory.

### Volc two-node, 16-GPU training

Run the following **same command on each of the two workers** (or configure it
as the Volc distributed job entry). Each worker needs eight visible GPUs and
the shared repository/data mounts. Choose a fresh `RUN_ID` for every new job;
use exactly the same value on both workers, not independent timestamps.

```bash
set -euo pipefail
cd /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
source runtime/memory_ablation_train.env
export ROBOMME_TRAIN_VENV=/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv

# The credential file also contains OLD single-node topology: clear it first.
unset NODE_RANK MASTER_ADDR OUTPUT_DIR RESUME_CHECKPOINT ROBOTWIN_INIT_CHECKPOINT
unset STARWAM_MEMORY_LIMIT_GIB ROBOMME_GPU_MEMORY_GIB ROBOMME_PREFLIGHT_ONLY
export NNODES=2 GPUS_PER_NODE=8 NUM_GPUS=16
export MASTER_PORT="${MLP_WORKER_0_PORT:-29500}"
export RUN_ID=robomme_all16_anchor_helios_16gpu_b32ga1_s20k_r01
export BATCH_SIZE=32 GRAD_ACCUM=1 MAX_STEPS=20000
export SAVE_EVERY=5000 NUM_WORKERS=16
export LEARNING_RATE=0.0001 LR_SCHEDULER=cosine HISTORY_AUGMENTATION=true
export WANDB_ENTITY=wangmingxin-tsinghua-university WANDB_PROJECT=memorywam
export TOS_UPLOAD=1 TOS_BUCKET=ai-dev TOS_PREFIX=spidy.wang/memorywam_ckpt

bash experiments/robomme/train_volc_all16_16gpu.sh
```

`scripts/distributed_env.sh` resolves node rank from `MLP_ROLE_INDEX`, and the
master from `MLP_WORKER_0_HOST` or Volc `*-worker-0` hostnames. For manual launches
outside that platform, set `NODE_RANK=0` / `NODE_RANK=1` respectively and the
same routable `MASTER_ADDR=<node-0 private IP>` **after** the unsets above.
`MASTER_ADDR=127.0.0.1` is rejected for multi-node runs.

The effective batch is **512** (32 per GPU x 16 GPUs x accumulation 1).
20,000 optimizer steps is approximately **333 epochs** by the trainer's
60-steps-per-epoch estimate. Explicit `max_steps` takes priority over the
credential file's `NUM_EPOCHS=500`; cosine scheduling uses the 20,000-step budget.
The learning rate remains 1e-4, without automatic batch scaling.

The wrapper selects DeepSpeed ZeRO-1 with `standard` multi-node launch, so each
node starts only its eight local processes and requires no SSH between workers.
Node 0 creates the run directory; peer nodes wait for the shared launch marker
and verify matching training arguments/data paths. Each node has a separate
`train.worker-N.log`, while Hydra logs use global rank. Only global rank 0
creates the W&B run and shared config/weight exports; all ranks participate in
saving sharded training states. Only node 0 checks and uploads to TOS after
successful training. TOS uploads weights/stats/YAML, not optimizer states.

For a command-only check, prefix the wrapper with `ROBOMME_LAUNCH_DRY_RUN=1`.
It creates no run directory and does not allocate GPU memory or upload data.
Per-GPU batch 32 and real inter-node NCCL communication still need validation
on the allocated two-node job; the launcher checks do not measure GPU memory.

### Switchable 99-latent history

Both `train.sh` (one or multiple nodes) and `train_volc_all16_16gpu.sh` accept
`ROBOMME_MEMORY_VARIANT`. Set it identically on every worker before launching:

| Selector | Long / mid / current | Long patch | Mid patch | Anchors |
| --- | --- | --- | --- | --- |
| `multiscale` (default) | `[16,2,1]` | `[4,8,8]` | `[2,4,4]` | 2 |
| `multiscale_long99` | `[96,2,1]` | `[4,8,8]` | `[2,4,4]` | 2 |

Use the same Volc command above, changing these two lines before the launch:

```bash
export ROBOMME_MEMORY_VARIANT=multiscale_long99
export RUN_ID=robomme_all16_anchor_helios_long99_16gpu_b32ga1_s20k_r01
bash experiments/robomme/train_volc_all16_16gpu.sh
```

To return to the original recipe, set `ROBOMME_MEMORY_VARIANT=multiscale`
and choose a new `RUN_ID`. The launcher also resolves a Hydra argument such as
`model.helios_config.memory_variant=multiscale_long99`; conflicting environment
and Hydra selectors are rejected. The cross-node launch signature includes the
resolved profile. Dataset sizes/anchors follow the model via Hydra interpolation.

The 99 slots include the current decision and are additional to the two fixed
anchors. The concatenated observed demo + execution prefix is left-zero-padded
or truncated to its latest 99 latents. Both phases keep separate VAE encoding
streams. The same complete phase latent cache is reused; no conversion or cache
regeneration is needed. Sampling, action offset and training step count are unchanged.

On the 24x20 latent grid, history/anchor tokens increase from 426 to 606;
including the future training target, video tokens increase from 546 to 726.
This is a token count, not a measured memory or throughput ratio. Temporal RoPE
coordinates extend with the history rather than reusing the 19-slot positions.

Deployment uses the saved `config.yaml` and matching checkpoint through the
existing serve command. No inference environment switch is needed. Strict
checkpoint metadata rejects a 19/99 layout mismatch; switching the training
variant means starting a separate run, not changing a checkpoint's layout.

Validation commands (GPU smoke uses one allocated GPU):

```bash
source experiments/robomme/env.sh
python -B -m pytest -q tests/test_helios_contract.py tests/test_robomme_helios.py tests/test_robomme_launch.py
python -B scripts/check_robomme_helios.py --memory-variant multiscale_long99
CUDA_VISIBLE_DEVICES=<free_gpu> python -B scripts/smoke_robomme_helios.py \
  --memory-variant multiscale_long99 --output runs/robomme_long99_model_smoke
```

Validated on 2026-10-15 in `runs/robomme_long99_smoke_20261015/`: 36 regression
tests passed; first/tail samples from all 16 tasks passed alignment and history
selection checks. The real pretrained model completed a B=1 backward (1655
finite gradient tensors) and inference on episode 1303 with 90 observed latents;
train/inference selected identical history. Four online/cache VAE comparisons
had zero latent error. The gradient-release smoke updated only the small memory
projections/proprio and is not a full optimizer or batch-32 capacity measurement.
The 8-GPU and both 16-GPU worker commands passed dry-run checks. No production
training was launched by this validation.

The audited 1600 training episodes (900 with demonstrations) contain at most 90
decision latents over demo + execution, so 99 slots cover each training prefix
in full. This does not assume a bound on evaluation episode length: prefixes
longer than 99 still slide, while the two anchors remain fixed.

## Training/inference contract

1. Source observations are **after the recorded action**. At frame `t`, the
   supervised chunk is `action[t+1:t+17]`, with current state at `t`. Training
   starts at `exec_start_idx`; demonstration actions never become targets.
2. Native action dimension is 8: seven absolute Panda joint targets plus gripper
   control (`-1` close, `+1` open). Proprioception is seven joints plus the mean
   of the two finger positions. The conversion's six zero padding dimensions are
   removed. The pretrained action backbone is reused; its input/output heads
   and the proprio encoder are freshly initialized for 8D inputs.
3. Front RGB goes below two copies of wrist RGB in a 384×320 mosaic. The same
   resize, RGB conversion, and normalization are used in both paths. MP4 decoding
   matches Kevin's OpenCV cache generator.
4. Demonstration endpoints are every 16 frames plus its final frame. An
   irregular final interval uses rounded four-way endpoint interpolation.
   Execution replans every 16 actions. Each phase has a separate causal VAE
   stream; memory and anchors retain the demonstration across the transition.
5. Training and inference select the same first two global decision latents as
   anchors and the same last 19 (default) or 99 (`multiscale_long99`) observed latents as Helios history. Startup
   padding is zero. Future video targets are never inserted into observed memory.
6. The five-frame future training clip remains `[t,t+4,t+8,t+12,t+16]`, locally
   VAE encoded as in the current framework. Action/image tails are masked.
7. Normalization, text assets, VAE hash, phase contract, and the data fingerprint
   are checked before training/deployment. Checkpoint loading is strict. The
   default RMBench/RoboDojo contract is unchanged.

The online policy currently re-encodes the visible execution prefix, matching
the cache exactly. This favors a verified reference implementation; incremental
VAE caching can be optimized later without changing the policy contract.

## Evaluation

On the inference host, source the training environment and serve a checkpoint:

```bash
source experiments/robomme/env.sh
CUDA_VISIBLE_DEVICES=2 python -B experiments/robomme/serve.py \
  --config runs/ROBOMME_RUN/config.yaml \
  --checkpoint runs/ROBOMME_RUN/checkpoints/weights/step_020000.pt \
  --steps 50 --port 18176 --output runs/ROBOMME_RUN/eval_server
```

In a second terminal on the same host, use the separate simulator environment:

```bash
CUDA_VISIBLE_DEVICES=3 bash experiments/robomme/eval.sh \
  --task PatternLock --split val --episodes 1 --port 18176 \
  --contract runs/ROBOMME_RUN/eval_server/inference_contract.json \
  --output runs/ROBOMME_RUN/eval_patternlock_val
```

The client uses the official observation/action protocol and checks the server
metadata before rollout. It records split, step limit, episode IDs and outcomes.
Only `status == success` counts as success. The default limit follows the
installed official builder (10,000); keep it consistent when reporting results.
Held-out language variants use the same frozen Wan T5/prompt/length through a
CPU fallback, cached separately under `eval_text_cache`; the server logs their
asset hashes. No evaluation observations are used to change the weights.

## Validation artifacts

- `runtime-robomme/prepared-anchor-helios/audit.json`: all-episode audit.
- `runtime-robomme/prepared-anchor-helios/alignment_check.json`: first/tail samples
  for all 16 tasks, +1 action alignment and task balancing.
- `runs/robomme_helios_smoke_20261014/model_smoke.json`: real pretrained full
  backward, memory-branch gradients, phase cache parity and checkpoint roundtrip.
- `runs/robomme_helios_smoke_20261014/rollout_pickxtimes/summary.json` and
  `rollout_patternlock/summary.json`: official simulation/transport smoke tests.
- `runs/robomme_helios_training_smoke_a100_20261014`: the actual Hydra/Accelerate
  training entry completed a full-model Adam optimizer step with ZeRO-2 CPU
  offload, loss 4.0580 (action 2.3005, video 1.7575), and saved
  `checkpoints/weights/step_000001.pt`. `checkpoint_validation.json` checks its
  weights and compatibility with the validated policy contract.

The model smoke checks 1655 gradient tensors; its peak allocated GPU memory was
14.83 GiB. All four online/cache comparisons (with/without demonstration, two
decisions each) had exactly zero latent error. The small-model regression suite
and RoboMME protocol tests passed (13 tests).

The step-0 smoke checkpoint only exercised updates to small new projections; it
is not a trained policy. The separate step-1 training smoke updated the full
trainable model, but is also not a converged policy. No 20,000-step production
run was started. The two short simulation checks each executed 33 steps
and 3 replans, then timed out as configured. They are wiring checks, not scores.

Reference implementation inspected:
`/mnt/vepfs02/output/kevin_wang/memorywam/eval_code/phasepure_kd020_25k_20261008`,
especially `scripts/cache_videoplaceorder.py`, the phase clock, policy, and client.
