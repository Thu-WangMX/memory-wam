#!/usr/bin/env bash
# One 8-GPU worker. Per-GPU batch and all experiment knobs remain overridable.
set -euo pipefail
# Capture experiment knobs before the credentials file supplies legacy defaults.
requested_batch="${BATCH_SIZE:-16}" requested_ga="${GRAD_ACCUM:-4}"
requested_steps="${MAX_STEPS:-20000}" requested_save="${SAVE_EVERY:-2000}"
requested_workers="${NUM_WORKERS:-16}" requested_lr="${LEARNING_RATE:-0.0001}"
requested_port="${MASTER_PORT:-29500}" requested_run="${RUN_ID:-}"
requested_weight="${PAST_ACTION_LOSS_WEIGHT:-0.1}" requested_read="${PAST_ACTION_READ:-true}"
requested_task="${ROBOMME_TRAIN_TASK:-robomme_helios_past_action}"
requested_memory="${ROBOMME_MEMORY_VARIANT:-multiscale_long99}"
case "$requested_task" in
  robomme_helios_push75_asv|robomme_helios_push75_288) variant_name=push75_variant ;;
  robomme_helios_push75_flow) variant_name=push75_past32x8_future2 ;;
  robomme_helios_past32_flow) variant_name=past32 ;;
  robomme_helios_pastfuture32_flow) variant_name=pastfuture32 ;;
  robomme_helios_pastfuture32_light_flow) variant_name=pastfuture32_light ;;

  robomme_helios_past_action) variant_name=pastdet ;;
  robomme_helios_past_action_block16) variant_name=pastblock16 ;;
  robomme_helios_past_action_flow) variant_name=pastflow16 ;;
  robomme_helios_past_action_sparse16x8_flow) variant_name=pastflow_sparse16x8 ;;
  *) echo 'Unknown past-action training task' >&2; exit 1 ;;
esac
if [[ "$requested_memory" == long99_less_spatial ]]; then variant_name="${variant_name}_spatial"; fi
cd /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
source runtime/memory_ablation_train.env
export ROBOMME_TRAIN_VENV=/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv
source experiments/robomme/env.sh
export NUM_GPUS=8 GPUS_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="$requested_port"
export ROBOMME_MEMORY_VARIANT="$requested_memory" ROBOMME_TRAIN_TASK="$requested_task"
export BATCH_SIZE="$requested_batch" GRAD_ACCUM="$requested_ga"
export MAX_STEPS="$requested_steps" SAVE_EVERY="$requested_save"
export NUM_WORKERS="$requested_workers" LEARNING_RATE="$requested_lr"
export PAST_ACTION_LOSS_WEIGHT="$requested_weight" PAST_ACTION_READ="$requested_read"
export RUN_ID="${requested_run:-robomme_all16_long99_${variant_name}_8gpu_b${BATCH_SIZE}ga${GRAD_ACCUM}_s${MAX_STEPS}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="$ROBOMME_FASTWAM_REPO/runs/$RUN_ID"
export FASTWAM_CHECKPOINT_WHOLE_LAYER=1 FASTWAM_FFN_CHUNK_SIZE=256 FASTWAM_QKV_CHUNK_SIZE=256
export WANDB_ENABLED=true WANDB_MODE=online
export WANDB_ENTITY=wangmingxin-tsinghua-university WANDB_PROJECT=memorywam
export WANDB_NAME="$RUN_ID" WANDB_DIR="$OUTPUT_DIR"
export WANDB_BASE_URL=https://api.wandb.ai WANDB_INIT_TIMEOUT=120
export TOS_BUCKET=ai-dev TOS_PREFIX=spidy.wang/memorywam_ckpt
export TOS_ENDPOINT=https://tos-cn-beijing.ivolces.com TOS_REGION=cn-beijing
export PYTHONPATH="$PYTHONPATH:/mnt/vepfs01/output/spidy.wang/starwam"
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com"
export no_proxy="$NO_PROXY"
export TRITON_CACHE_DIR="/mnt/vepfs01/output/spidy.wang/runtime-robomme/cache/triton/$RUN_ID"
unset RESUME_CHECKPOINT ROBOTWIN_INIT_CHECKPOINT STARWAM_MEMORY_LIMIT_GIB
unset ROBOMME_GPU_MEMORY_GIB ROBOMME_PREFLIGHT_ONLY
unset ACCELERATE_USE_CPU ACCELERATE_USE_FSDP WANDB_RUN_ID WANDB_RESUME
train_args=(
  'data.train.tasks=[all]' data.train.task_balanced=true
  "batch_size=$BATCH_SIZE" "gradient_accumulation_steps=$GRAD_ACCUM"
  "max_steps=$MAX_STEPS" "save_every=$SAVE_EVERY" "num_workers=$NUM_WORKERS"
  persistent_workers=true continuous_dataloader=true prefetch_factor=2
  "learning_rate=$LEARNING_RATE" lr_scheduler_type=cosine
  mixed_precision=bf16 seed=42 resume=null
  save_training_state=true save_final_checkpoint=true
  keep_training_state_checkpoints=1 eval_every=0
  model.helios_config.condition_augmentation=true
  model.helios_config.backbone_init_checkpoint=null
  "model.helios_config.past_action.loss_weight=$PAST_ACTION_LOSS_WEIGHT"
  "model.helios_config.past_action.action_reads_past=$PAST_ACTION_READ"
  wandb.enabled=true "wandb.workspace=$WANDB_ENTITY" "wandb.project=$WANDB_PROJECT"
  "wandb.name=$RUN_ID" wandb.mode=online
)
if [[ "${ROBOMME_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
  exec bash experiments/robomme/train.sh "${train_args[@]}" "$@"
fi
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
bash experiments/robomme/train.sh "${train_args[@]}" "$@"
python -B scripts/upload_checkpoints.py --run "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tos-upload.log"
