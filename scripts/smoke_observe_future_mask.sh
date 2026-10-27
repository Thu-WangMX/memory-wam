#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RMBENCH_TASK=observe_and_pickup MEMORY_BACKEND=helios
export NNODES=1 NODE_RANK=0 NUM_GPUS=8 GPUS_PER_NODE=8
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29573
export BATCH_SIZE=16 GRAD_ACCUM=1 NUM_WORKERS=2
export MAX_STEPS=2 SAVE_EVERY=500 WANDB_ENABLED=false TOS_UPLOAD=0
unset NUM_EPOCHS RESUME_CHECKPOINT WANDB_RUN_ID RUN_ID
export FUTURE_MASK_CACHE="$PWD/data_audits/observe_future_mask_localclip_cache_20260929_v1"
export OUTPUT_DIR="$PWD/runs/observe_future_mask_bs16_8gpu_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "$OUTPUT_DIR" > runtime/observe_future_mask_smoke_latest.txt
bash scripts/train_with_logging.sh resume=null \
  data=rmbench_helios_future_mask model=fastwam_helios_future_mask \
  ++model.helios_config.backbone_init_checkpoint=/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_release/robotwin_uncond_3cam_384.pt \
  model.helios_config.condition_augmentation=true \
  learning_rate=0.0001 lr_scheduler_type=cosine ++warmup_ratio=0.05 \
  weight_decay=0.01 max_grad_norm=1.0 seed=42 \
  save_training_state=false save_final_checkpoint=false log_every=1
