#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export RMBENCH_TASK=observe_and_pickup MEMORY_BACKEND=helios
export COMPARISON_VENV="${COMPARISON_VENV:-/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv}"
export NUM_GPUS="${NUM_GPUS:-8}" BATCH_SIZE="${BATCH_SIZE:-8}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-500}" SAVE_EVERY="${SAVE_EVERY:-500}"
export WANDB_PROJECT="${WANDB_PROJECT:-memorywam}" WANDB_ENTITY="${WANDB_ENTITY:-wangmingxin-tsinghua-university}"
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29500}"
unset MAX_STEPS
lr="${LEARNING_RATE:-0.0001}"
scheduler="${LR_SCHEDULER:-cosine}"
name="observe_and_pickup_fastwam-anchor-helios-nohistnoise-robotwininit_bs${BATCH_SIZE}_ga${GRAD_ACCUM}_gpu${NUM_GPUS}_ep${NUM_EPOCHS}_save${SAVE_EVERY}_lr${lr}_${scheduler}_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/$name}"
init=/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_release/robotwin_uncond_3cam_384.pt
exec bash scripts/train_with_logging.sh "$@" "wandb.name=$name" resume=null \
 "++model.helios_config.backbone_init_checkpoint=$init" \
 model.helios_config.condition_augmentation=false "learning_rate=$lr" "lr_scheduler_type=$scheduler"
