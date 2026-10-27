#!/usr/bin/env bash
# Single-worker 8-GPU run; reuse the audited all35 cache without rewriting it.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBODOJO_FASTWAM_REPO"
[[ "${MLP_WORKER_NUM:-1}" == 1 && "${MLP_ROLE_INDEX:-0}" == 0 ]] || {
  echo 'This entry requires one worker with eight GPUs.' >&2; exit 1;
}
: "${RUN_ID:?Set a fresh RUN_ID}"
: "${WANDB_API_KEY:?Set existing WANDB_API_KEY}"
: "${TOS_ACCESS_KEY:?Set existing TOS_ACCESS_KEY}"
: "${TOS_SECRET_KEY:?Set existing TOS_SECRET_KEY}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]
export NNODES=1 NODE_RANK=0 NUM_GPUS=8 GPUS_PER_NODE=8
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29500}"
export ROBODOJO_HELIOS_LATENTS="$ROBODOJO_HELIOS_PREPARED/latents-all35"
export OUTPUT_DIR="$ROBODOJO_FASTWAM_REPO/runs/$RUN_ID"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-wangmingxin-tsinghua-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-memorywam}" WANDB_MODE=online
export WANDB_INIT_TIMEOUT=120 WANDB_DIR="$OUTPUT_DIR"
export TOS_BUCKET="${TOS_BUCKET:-ai-dev}" TOS_PREFIX="${TOS_PREFIX:-spidy.wang/memorywam_ckpt}"
export TOS_ENDPOINT="${TOS_ENDPOINT:-https://tos-cn-beijing.ivolces.com}" TOS_REGION=cn-beijing
export PYTHONPATH="$PYTHONPATH:/mnt/vepfs01/output/spidy.wang/starwam"
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com"
export no_proxy="$NO_PROXY"
export TRITON_CACHE_DIR="$ROBODOJO_RUNTIME/cache/triton/$RUN_ID/worker-0"
export TORCH_EXTENSIONS_DIR="$ROBODOJO_RUNTIME/cache/torch_extensions/$RUN_ID/worker-0"
unset RESUME_CHECKPOINT NUM_EPOCHS STARWAM_MEMORY_LIMIT_GIB ACCELERATE_USE_CPU ROBODOJO_PREFLIGHT_ONLY
test -f "$ROBODOJO_HELIOS_LATENTS/manifest.json" || { echo 'Missing prepared all35 cache' >&2; exit 1; }
python -c 'import torch; assert torch.cuda.device_count() == 8, "Expose exactly eight GPUs"'
# Refuse to mix a new run with an existing output directory.
mkdir "$OUTPUT_DIR"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
python -B scripts/upload_checkpoints.py --doctor
ROBODOJO_PREFLIGHT_ONLY=1 python -B experiments/robodojo/train.py \
  task=robodojo_helios 'data.train.tasks=[all]' batch_size=16 gradient_accumulation_steps=2 \
  2>&1 | tee "$OUTPUT_DIR/preflight.worker-0.log"

# Preserve the exact normalization bytes required by the post-training uploader.
python -B - <<'PY'
import os, pathlib, hashlib
source=pathlib.Path(os.environ['ROBODOJO_HELIOS_PREPARED'])/'dataset_stats.json'
target=pathlib.Path(os.environ['OUTPUT_DIR'])/'dataset_stats.json'
data=source.read_bytes()
with target.open('xb') as f:
    f.write(data)
assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(data).digest()
PY

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 --num_machines 1 --machine_rank 0 \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  experiments/robodojo/train.py task=robodojo_helios 'data.train.tasks=[all]' \
  "output_dir=$OUTPUT_DIR" batch_size=16 gradient_accumulation_steps=2 num_workers=16 \
  max_steps=20000 save_every=2500 learning_rate=0.0001 lr_scheduler_type=cosine \
  warmup_ratio=0.05 weight_decay=0.01 max_grad_norm=1.0 mixed_precision=bf16 seed=42 \
  model.helios_config.condition_augmentation=true model.helios_config.backbone_init_checkpoint=null \
  resume=null save_training_state=true keep_training_state_checkpoints=1 save_final_checkpoint=true eval_every=0 \
  wandb.enabled=true "wandb.workspace=$WANDB_ENTITY" "wandb.project=$WANDB_PROJECT" \
  "wandb.name=$RUN_ID" wandb.mode=online \
  2>&1 | tee "$OUTPUT_DIR/train.worker-0.log"
python -B scripts/upload_checkpoints.py --run "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/tos-upload.log"
