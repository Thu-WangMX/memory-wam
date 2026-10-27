#!/usr/bin/env bash
# Observe-only future mask auxiliary experiment. Credentials stay in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
export COMPARISON_VENV="${COMPARISON_VENV:-/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv}"
export RMBENCH_TASK=observe_and_pickup MEMORY_BACKEND=helios
export NNODES=1 NODE_RANK=0 NUM_GPUS=8 GPUS_PER_NODE=8
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29500}"
export BATCH_SIZE=4 GRAD_ACCUM=1 NUM_WORKERS="${NUM_WORKERS:-16}"
export NUM_EPOCHS=500 SAVE_EVERY=500
export WANDB_ENABLED=true WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=wangmingxin-tsinghua-university WANDB_PROJECT=memorywam WANDB_MODE=online
export TOS_UPLOAD=1 TOS_BUCKET=ai-dev TOS_PREFIX=spidy.wang/memorywam_ckpt
export http_proxy="${http_proxy:-http://192.168.39.105:7890}"
export https_proxy="$http_proxy"
export NO_PROXY=localhost,127.0.0.1,ai-dev.tos-cn-beijing.ivolces.com,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com
export no_proxy="$NO_PROXY" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset MAX_STEPS RESUME_CHECKPOINT STARWAM_MEMORY_LIMIT_GIB WANDB_RUN_ID RUN_ID
unset RMBENCH_TASK_ROOT RMBENCH_COVER_BLOCKS_LEROBOT MEMORYWAM_COVER_BLOCKS_STATS
unset MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS MEMORYWAM_COVER_BLOCKS_TEXT_CACHE
export FUTURE_MASK_CACHE="${FUTURE_MASK_CACHE:-$PWD/data_audits/observe_future_mask_localclip_cache_20260929_v1}"
run_name="observe_and_pickup_helios_futuremaskaux_robotwininit_bs4_ga1_gpu8_ep500_lr1e-4_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/runs/$run_name}" WANDB_DIR="${OUTPUT_DIR:-$PWD/runs/$run_name}"
mode=()
if [[ "${1:-}" == --config ]]; then
  mode=(--config)
  shift
else
  : "${WANDB_API_KEY:?Export WANDB_API_KEY before launching}"
  : "${TOS_ACCESS_KEY:?Export TOS_ACCESS_KEY before launching}"
  : "${TOS_SECRET_KEY:?Export TOS_SECRET_KEY before launching}"
fi
bash scripts/train_with_logging.sh "${mode[@]}" \
  "wandb.name=$run_name" resume=null \
  data=rmbench_helios_future_mask model=fastwam_helios_future_mask \
  ++model.helios_config.backbone_init_checkpoint=/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_release/robotwin_uncond_3cam_384.pt \
  model.helios_config.condition_augmentation=true \
  learning_rate=0.0001 lr_scheduler_type=cosine ++warmup_ratio=0.05 \
  weight_decay=0.01 max_grad_norm=1.0 seed=42 save_training_state=false "$@"
