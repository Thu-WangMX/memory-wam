#!/usr/bin/env bash
# Invoke once per worker of a 2 x 8 GPU Volc job, with the same fresh RUN_ID.
# Load runtime/memory_ablation_train.env BEFORE setting the job topology.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOMME_FASTWAM_REPO"
export NNODES="${NNODES:-2}" GPUS_PER_NODE="${GPUS_PER_NODE:-8}" NUM_GPUS="${NUM_GPUS:-16}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
source scripts/distributed_env.sh
[[ "$NNODES" == 2 && "$GPUS_PER_NODE" == 8 && "$NUM_GPUS" == 16 ]] || {
  echo 'This entry requires NNODES=2 GPUS_PER_NODE=8 NUM_GPUS=16' >&2; exit 1;
}
[[ "${MLP_WORKER_NUM:-2}" == 2 && "${MLP_ROLE_INDEX:-$NODE_RANK}" == "$NODE_RANK" ]] || {
  echo 'Topology disagrees with Volc worker allocation/rank; clear old single-node variables' >&2; exit 1;
}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 1; }
export OUTPUT_DIR="$ROBOMME_FASTWAM_REPO/runs/$RUN_ID"
export BATCH_SIZE="${BATCH_SIZE:-32}" GRAD_ACCUM="${GRAD_ACCUM:-1}" MAX_STEPS="${MAX_STEPS:-20000}"
export SAVE_EVERY="${SAVE_EVERY:-5000}" NUM_WORKERS="${NUM_WORKERS:-16}"
export LEARNING_RATE="${LEARNING_RATE:-0.0001}" LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export HISTORY_AUGMENTATION="${HISTORY_AUGMENTATION:-true}"
export FASTWAM_CHECKPOINT_WHOLE_LAYER=1 FASTWAM_FFN_CHUNK_SIZE=256 FASTWAM_QKV_CHUNK_SIZE=256
export WANDB_ENABLED=true WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-wangmingxin-tsinghua-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-memorywam}" WANDB_NAME="$RUN_ID" WANDB_DIR="$OUTPUT_DIR"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}" WANDB_INIT_TIMEOUT=120
export TOS_UPLOAD="${TOS_UPLOAD:-1}" TOS_BUCKET="${TOS_BUCKET:-ai-dev}"
export TOS_PREFIX="${TOS_PREFIX:-spidy.wang/memorywam_ckpt}"
export TOS_ENDPOINT="${TOS_ENDPOINT:-https://tos-cn-beijing.ivolces.com}" TOS_REGION="${TOS_REGION:-cn-beijing}"
export PYTHONPATH="$PYTHONPATH:/mnt/vepfs01/output/spidy.wang/starwam"
export NO_PROXY="${NO_PROXY:-},$MASTER_ADDR,localhost,127.0.0.1,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com"
export no_proxy="$NO_PROXY"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/cache/triton/$RUN_ID/worker-$NODE_RANK}"
unset RESUME_CHECKPOINT ROBOTWIN_INIT_CHECKPOINT STARWAM_MEMORY_LIMIT_GIB
unset ROBOMME_GPU_MEMORY_GIB ROBOMME_PREFLIGHT_ONLY ACCELERATE_USE_CPU ACCELERATE_USE_FSDP
unset WANDB_RUN_ID WANDB_RESUME

train_args=(
  'data.train.tasks=[all]' data.train.task_balanced=true
  "batch_size=$BATCH_SIZE" "gradient_accumulation_steps=$GRAD_ACCUM" "max_steps=$MAX_STEPS"
  "save_every=$SAVE_EVERY" "num_workers=$NUM_WORKERS"
  "learning_rate=$LEARNING_RATE" "lr_scheduler_type=$LR_SCHEDULER"
  mixed_precision=bf16 seed=42 resume=null
  save_training_state=true save_final_checkpoint=true keep_training_state_checkpoints=1 eval_every=0
  "model.helios_config.condition_augmentation=$HISTORY_AUGMENTATION"
  model.helios_config.backbone_init_checkpoint=null
  wandb.enabled=true "wandb.workspace=$WANDB_ENTITY" "wandb.project=$WANDB_PROJECT"
  "wandb.name=$RUN_ID" wandb.mode=online)

# Dry-run verifies the two workers' commands without touching CUDA or outputs.
if [[ "${ROBOMME_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
  exec bash experiments/robomme/train.sh "${train_args[@]}" "$@"
fi
: "${WANDB_API_KEY:?Source runtime/memory_ablation_train.env first}"
if [[ "$TOS_UPLOAD" == 1 ]]; then
  : "${TOS_ACCESS_KEY:?Source runtime/memory_ablation_train.env first}"
  : "${TOS_SECRET_KEY:?Source runtime/memory_ablation_train.env first}"
fi
python -B - <<'PY'
import os, torch
assert torch.cuda.device_count() == int(os.environ['GPUS_PER_NODE']), 'Expose exactly eight GPUs per worker'
print(f"worker={os.environ['NODE_RANK']} world_size={os.environ['NUM_GPUS']} "
      f"python={os.sys.executable} visible_gpus={torch.cuda.device_count()}", flush=True)
PY
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
if [[ "$TOS_UPLOAD" == 1 && "$NODE_RANK" == 0 ]]; then
  python -B scripts/upload_checkpoints.py --doctor
fi
bash experiments/robomme/train.sh "${train_args[@]}" "$@"
if [[ "$TOS_UPLOAD" == 1 && "$NODE_RANK" == 0 ]]; then
  python -B scripts/upload_checkpoints.py --run "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tos-upload.log"
fi
