#!/usr/bin/env bash
# Submit once with one 8-GPU worker. Credentials/runtime come from the existing launcher.
set -euo pipefail
repo=/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
cd "$repo"
if [[ "${MLP_WORKER_NUM:-1}" != 1 || "${MLP_ROLE_INDEX:-0}" != 0 ]]; then
  echo 'This experiment requires one worker with eight GPUs.' >&2
  exit 1
fi
export ROBOMME_TRAIN_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BATCH_SIZE=32 GRAD_ACCUM=1 MAX_STEPS=100000 SAVE_EVERY=2000
export NUM_WORKERS=16 LEARNING_RATE=0.0001
export PAST_ACTION_LOSS_WEIGHT=0.1 PAST_ACTION_READ=true
export MASTER_PORT="${MASTER_PORT:-29500}"
export ROBOMME_PUSH75_CACHE_ROOT=/mnt/vepfs01/output/spidy.wang/runtime-robomme/two-view-square256-v1
export ROBOMME_HELIOS_LATENTS="$ROBOMME_PUSH75_CACHE_ROOT/base"
export ROBOMME_HELIOS_PREPARED="$ROBOMME_PUSH75_CACHE_ROOT/prepared"
export ROBOMME_DENSE_LATENTS="$ROBOMME_PUSH75_CACHE_ROOT/dense"
export ROBOMME_DREAMDOJO_LABELS=/mnt/vepfs01/output/spidy.wang/runtime-robomme/dreamdojo_labels_v1/cache/head_wrist64_stride1
unset OUTPUT_DIR RESUME_CHECKPOINT WANDB_RUN_ID WANDB_RESUME
export RUN_ID="robomme_all16_push75_action2_wsd_8gpu_b32ga1_s100k_${MLP_TASK_ID:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
exec bash experiments/robomme/train_push75.sh \
  lr_scheduler_type=wsd ++warmup_ratio=0.02 ++wsd_decay_steps=30000 \
  model.loss.lambda_action=2.0 model.loss.lambda_video=1.0 \
  model.helios_config.past_action.loss_weight=0.1 \
  model.helios_config.past_action.future_loss_weight=0.1 \
  "$@"
