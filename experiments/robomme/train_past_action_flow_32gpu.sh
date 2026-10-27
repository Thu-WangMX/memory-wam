#!/usr/bin/env bash
# Submit once to Volc with four workers, eight GPUs each. The platform invokes
# this same entry on every worker; distributed_env.sh resolves their ranks.
set -euo pipefail
requested_task="${ROBOMME_TRAIN_TASK:-robomme_helios_past_action_flow}"
requested_memory="${ROBOMME_MEMORY_VARIANT:-multiscale_long99}"
case "$requested_task" in
  robomme_helios_push75_flow) variant_name=push75_past32x8_future2 ;;
  robomme_helios_past32_flow) variant_name=past32 ;;
  robomme_helios_pastfuture32_flow) variant_name=pastfuture32 ;;
  robomme_helios_pastfuture32_light_flow) variant_name=pastfuture32_light ;;

  robomme_helios_past_action_flow) variant_name=pastflow16 ;;
  robomme_helios_past_action_sparse16x8_flow) variant_name=pastflow_sparse16x8 ;;
  *) echo 'Unknown past-action flow training task' >&2; exit 1 ;;
esac
if [[ "$requested_memory" == long99_less_spatial ]]; then variant_name="${variant_name}_spatial"; fi

# Credentials may also contain old single-node experiment defaults. Preserve
# this invocation's recipe and topology before sourcing that environment.
requested_batch="${BATCH_SIZE:-16}" requested_ga="${GRAD_ACCUM:-1}"
requested_steps="${MAX_STEPS:-20000}" requested_save="${SAVE_EVERY:-2500}"
requested_workers="${NUM_WORKERS:-16}" requested_lr="${LEARNING_RATE:-0.0001}"
requested_nodes="${NNODES:-4}" requested_gpus="${NUM_GPUS:-32}"
requested_local_gpus="${GPUS_PER_NODE:-8}"
requested_rank="${NODE_RANK:-${MLP_ROLE_INDEX:-}}" requested_master="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-}}"
requested_port="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
requested_run="${RUN_ID:-}" requested_cache="${TRITON_CACHE_DIR:-}"
requested_weight="${PAST_ACTION_LOSS_WEIGHT:-0.1}" requested_read="${PAST_ACTION_READ:-true}"
requested_upload="${TOS_UPLOAD:-1}"
requested_venv="${ROBOMME_TRAIN_VENV:-/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv}"
requested_cuda="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ -z "$requested_run" && -n "${MLP_TASK_ID:-}" ]]; then
  requested_run="robomme_all16_long99_${variant_name}_32gpu_b${requested_batch}ga${requested_ga}_s${requested_steps}_${MLP_TASK_ID}"
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source runtime/memory_ablation_train.env
export ROBOMME_TRAIN_VENV="$requested_venv"
export CUDA_VISIBLE_DEVICES="$requested_cuda"
source experiments/robomme/env.sh
export NNODES="$requested_nodes" GPUS_PER_NODE="$requested_local_gpus" NUM_GPUS="$requested_gpus"
export NODE_RANK="$requested_rank" MASTER_ADDR="$requested_master" MASTER_PORT="$requested_port"
export RUN_ID="$requested_run"
source scripts/distributed_env.sh
[[ "$NNODES" == 4 && "$GPUS_PER_NODE" == 8 && "$NUM_GPUS" == 32 ]] || {
  echo 'This entry requires four workers with eight GPUs each (32 total)' >&2; exit 1;
}
[[ "${MLP_WORKER_NUM:-4}" == 4 && "${MLP_ROLE_INDEX:-$NODE_RANK}" == "$NODE_RANK" ]] || {
  echo 'Topology disagrees with Volc worker allocation/rank; clear stale single-node variables' >&2; exit 1;
}
export BATCH_SIZE="$requested_batch" GRAD_ACCUM="$requested_ga"
export MAX_STEPS="$requested_steps" SAVE_EVERY="$requested_save"
export NUM_WORKERS="$requested_workers" LEARNING_RATE="$requested_lr"
export PAST_ACTION_LOSS_WEIGHT="$requested_weight" PAST_ACTION_READ="$requested_read"
export ROBOMME_MEMORY_VARIANT="$requested_memory" ROBOMME_TRAIN_TASK="$requested_task"
if [[ -z "$requested_run" ]]; then
  # distributed_env supplies the common job hostname prefix, never a per-node timestamp.
  export RUN_ID="robomme_all16_long99_${variant_name}_32gpu_b${BATCH_SIZE}ga${GRAD_ACCUM}_s${MAX_STEPS}_${RUN_ID}"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 1; }
export OUTPUT_DIR="$ROBOMME_FASTWAM_REPO/runs/$RUN_ID"
export FASTWAM_CHECKPOINT_WHOLE_LAYER=1 FASTWAM_FFN_CHUNK_SIZE=256 FASTWAM_QKV_CHUNK_SIZE=256
export WANDB_ENABLED=true WANDB_MODE=online
export WANDB_ENTITY=wangmingxin-tsinghua-university WANDB_PROJECT=memorywam
export WANDB_NAME="$RUN_ID" WANDB_DIR="$OUTPUT_DIR"
export WANDB_BASE_URL=https://api.wandb.ai WANDB_INIT_TIMEOUT=120
export TOS_UPLOAD="$requested_upload" TOS_BUCKET=ai-dev TOS_PREFIX=spidy.wang/memorywam_ckpt
export TOS_ENDPOINT=https://tos-cn-beijing.ivolces.com TOS_REGION=cn-beijing
export PYTHONPATH="$PYTHONPATH:/mnt/vepfs01/output/spidy.wang/starwam"
export NO_PROXY="${NO_PROXY:-},$MASTER_ADDR,${MLP_WORKER_0_HOST:-},${MLP_WORKER_1_HOST:-},${MLP_WORKER_2_HOST:-},${MLP_WORKER_3_HOST:-},localhost,127.0.0.1,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com"
export no_proxy="$NO_PROXY"
export TRITON_CACHE_DIR="${requested_cache:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/cache/triton/$RUN_ID}/worker-$NODE_RANK"
unset RESUME_CHECKPOINT ROBOTWIN_INIT_CHECKPOINT STARWAM_MEMORY_LIMIT_GIB
unset ROBOMME_GPU_MEMORY_GIB ROBOMME_PREFLIGHT_ONLY ACCELERATE_USE_CPU ACCELERATE_USE_FSDP
unset WANDB_RUN_ID WANDB_RESUME

train_args=(
  'data.train.tasks=[all]' data.train.task_balanced=true
  "batch_size=$BATCH_SIZE" "gradient_accumulation_steps=$GRAD_ACCUM"
  "max_steps=$MAX_STEPS" "save_every=$SAVE_EVERY" "num_workers=$NUM_WORKERS"
  persistent_workers=true continuous_dataloader=true prefetch_factor=2
  "learning_rate=$LEARNING_RATE" lr_scheduler_type=cosine
  mixed_precision=bf16 seed=42 resume=null
  save_training_state=true save_final_checkpoint=true keep_training_state_checkpoints=1 eval_every=0
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
: "${WANDB_API_KEY:?Missing W&B key in runtime/memory_ablation_train.env}"
if [[ "$TOS_UPLOAD" == 1 ]]; then
  : "${TOS_ACCESS_KEY:?Missing TOS key in runtime/memory_ablation_train.env}"
  : "${TOS_SECRET_KEY:?Missing TOS secret in runtime/memory_ablation_train.env}"
fi
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
if [[ "$TOS_UPLOAD" == 1 && "$NODE_RANK" == 0 ]]; then
  python -B scripts/upload_checkpoints.py --doctor
fi
bash experiments/robomme/train.sh "${train_args[@]}" "$@"
if [[ "$TOS_UPLOAD" == 1 && "$NODE_RANK" == 0 ]]; then
  python -B scripts/upload_checkpoints.py --run "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/tos-upload.log"
fi
