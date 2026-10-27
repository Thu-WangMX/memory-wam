#!/usr/bin/env bash
# Submit once with 2 workers, each exposing 8 GPUs.
set -euo pipefail
cd /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
: "${MLP_WORKER_NUM:?Select 2 workers}"
: "${MLP_ROLE_INDEX:?Missing worker rank}"
: "${MLP_WORKER_0_HOST:?Missing master host}"
: "${MLP_WORKER_0_PORT:?Missing master port}"
: "${MLP_TASK_ID:?Missing shared task ID}"
[[ "$MLP_WORKER_NUM" == 2 ]] || { echo 'Requires 2 workers x 8 GPUs' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2 GPUS_PER_NODE=8 NUM_GPUS=16
export NODE_RANK="$MLP_ROLE_INDEX" MASTER_ADDR="$MLP_WORKER_0_HOST" MASTER_PORT="$MLP_WORKER_0_PORT"
export ROBOMME_TRAIN_TASK=robomme_helios_push75_flow
export ROBOMME_MEMORY_VARIANT=long75_less_spatial ROBOMME_SAMPLE_STRIDE=1
export ROBOMME_TRAIN_VENV=/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv
export BATCH_SIZE=16 GRAD_ACCUM=1 MAX_STEPS=50000 SAVE_EVERY=2500
export NUM_WORKERS=16 LEARNING_RATE=0.0001
export PAST_ACTION_LOSS_WEIGHT=0.1 PAST_ACTION_READ=true
export ROBOMME_PUSH75_CACHE_ROOT=/mnt/vepfs01/output/spidy.wang/runtime-robomme/two-view-square256-v1
export ROBOMME_HELIOS_LATENTS="$ROBOMME_PUSH75_CACHE_ROOT/base"
export ROBOMME_HELIOS_PREPARED="$ROBOMME_PUSH75_CACHE_ROOT/prepared"
export ROBOMME_DENSE_LATENTS="$ROBOMME_PUSH75_CACHE_ROOT/dense"
export RUN_ID="robomme_all16_push75_long444_past32x8_future2_dense1_16gpu_b16ga1_s50k_${MLP_TASK_ID}"
unset OUTPUT_DIR RESUME_CHECKPOINT WANDB_RUN_ID WANDB_RESUME
unset ROBOTWIN_INIT_CHECKPOINT STARWAM_MEMORY_LIMIT_GIB
export ROBOMME_LAUNCH_DRY_RUN="${ROBOMME_LAUNCH_DRY_RUN:-0}"
if [[ "$ROBOMME_LAUNCH_DRY_RUN" != 1 ]]; then
  for file in "$ROBOMME_HELIOS_LATENTS/manifest.json" "$ROBOMME_HELIOS_PREPARED/provenance.json" "$ROBOMME_DENSE_LATENTS/manifest.json"; do
    [[ -s "$file" ]] || { echo "Missing cache: $file" >&2; exit 1; }
  done
fi
exec bash experiments/robomme/train_past_action_flow_16gpu.sh   "data.train.full_kv_cache_path=$ROBOMME_HELIOS_LATENTS"   "data.train.prepared_root=$ROBOMME_HELIOS_PREPARED"   "data.train.pretrained_norm_stats=$ROBOMME_HELIOS_PREPARED/dataset_stats.json"   "data.train.dense_latents_path=$ROBOMME_DENSE_LATENTS"   data.train.sample_stride=1 model.loss.lambda_action=1.0 model.loss.lambda_video=1.0   model.helios_config.past_action.future_loss_weight=0.1   warmup_ratio=0.05 resume=null ++weight_only_resume_step=0
