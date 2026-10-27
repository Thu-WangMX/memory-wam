#!/usr/bin/env bash
# One 8-GPU queue per variant/task group; independent single-task training runs.
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:?Usage: train_memory_layout_ablation.sh VARIANT [--task-group press_swapT|swap_blocks|all] [overrides]}"
shift
case "$variant" in wo_long|uniform_mid|reverse_scale|wo_anchor|long_less_temporal|long_less_spatial) ;; *) echo "Invalid layout variant" >&2; exit 1 ;; esac
layout_bs="${BATCH_SIZE:-16}"; layout_ga="${GRAD_ACCUM:-1}"
layout_epochs="${NUM_EPOCHS:-500}"; layout_save="${SAVE_EVERY:-2500}"
layout_workers="${NUM_WORKERS:-16}"; layout_lr="${LEARNING_RATE:-0.0001}"
layout_scheduler="${LR_SCHEDULER:-cosine}"; layout_noise="${HISTORY_AUGMENTATION:-true}"
set -a
source runtime/memory_ablation_train.env
set +a
export NUM_GPUS=8 GPUS_PER_NODE=8 NNODES=1 NODE_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29500}"
export BATCH_SIZE="$layout_bs" GRAD_ACCUM="$layout_ga"
export NUM_EPOCHS="$layout_epochs" SAVE_EVERY="$layout_save" NUM_WORKERS="$layout_workers"
export LEARNING_RATE="$layout_lr" LR_SCHEDULER="$layout_scheduler" HISTORY_AUGMENTATION="$layout_noise"
unset OUTPUT_DIR MAX_STEPS RESUME_CHECKPOINT STARWAM_MEMORY_LIMIT_GIB
exec bash scripts/train_memory_ablation_three_tasks.sh "$variant" "$@"
