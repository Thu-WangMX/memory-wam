#!/usr/bin/env bash
# Submit one worker exposing 8 GPUs. Fresh optimizer/scheduler and step counter.
set -euo pipefail
cd /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
source experiments/robomme/training_speed_ready.sh
[[ "${MLP_WORKER_NUM:-1}" == 1 ]] || { echo 'Requires one worker with 8 GPUs' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROBOMME_TRAIN_GPUS=8 ROBOMME_TRAIN_TASK=robomme_helios_push75_288
export ROBOMME_MEMORY_VARIANT=long75_less_temporal ROBOMME_SAMPLE_STRIDE=1
export ROBOMME_TRAIN_VENV=/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv
export MASTER_PORT="${MASTER_PORT:-29500}"
export BATCH_SIZE=32 GRAD_ACCUM=1 MAX_STEPS=50000 SAVE_EVERY=2500
export NUM_WORKERS=16 LEARNING_RATE=0.0001
export PAST_ACTION_LOSS_WEIGHT=0.1 PAST_ACTION_READ=true
export ROBOMME_PUSH75_CACHE_ROOT=/mnt/vepfs01/output/spidy.wang/runtime-robomme/two-view-square256-v1
export ROBOMME_HELIOS_LATENTS="$ROBOMME_PUSH75_CACHE_ROOT/base"
export ROBOMME_HELIOS_PREPARED="$ROBOMME_PUSH75_CACHE_ROOT/prepared"
export ROBOMME_DENSE_LATENTS="$ROBOMME_PUSH75_CACHE_ROOT/dense"
export RUN_ID="robomme_all16_push75_long288_past32x8_future2_dense1_8gpu_b32ga1_s50000_${MLP_TASK_ID:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
unset OUTPUT_DIR RESUME_CHECKPOINT ROBOTWIN_INIT_CHECKPOINT WANDB_RUN_ID WANDB_RESUME
unset STARWAM_MEMORY_LIMIT_GIB
export ROBOMME_LAUNCH_DRY_RUN="${ROBOMME_LAUNCH_DRY_RUN:-0}"
if [[ "$ROBOMME_LAUNCH_DRY_RUN" != 1 ]]; then
  for file in "$ROBOMME_HELIOS_LATENTS/manifest.json" "$ROBOMME_HELIOS_PREPARED/provenance.json" "$ROBOMME_DENSE_LATENTS/manifest.json"; do
    [[ -s "$file" ]] || { echo "Missing cache: $file" >&2; exit 1; }
  done
fi
exec bash experiments/robomme/train_past_action_8gpu.sh \
  "data.train.full_kv_cache_path=$ROBOMME_HELIOS_LATENTS" \
  "data.train.prepared_root=$ROBOMME_HELIOS_PREPARED" \
  "data.train.pretrained_norm_stats=$ROBOMME_HELIOS_PREPARED/dataset_stats.json" \
  "data.train.dense_latents_path=$ROBOMME_DENSE_LATENTS" \
  data.train.sample_stride=1 model.loss.lambda_action=1.0 model.loss.lambda_video=1.0 \
  model.helios_config.past_action.future_loss_weight=0.1 \
  warmup_ratio=0.05 min_learning_rate=0.000001 resume=null ++weight_only_resume_step=0
