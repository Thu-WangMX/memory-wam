#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
group="${1:?Usage: bash scripts/train_anchor_helios_retrain_groups.sh 1|2 [--config]}"
case "$group" in
  1) tasks=(observe_and_pickup swap_blocks) ;;
  2) tasks=(swap_T press_button) ;;
  *) echo "Group must be 1 or 2" >&2; exit 1 ;;
esac
config_only="${2:-}"
[[ -z "$config_only" || "$config_only" == --config ]] || exit 1
export COMPARISON_VENV=/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv
export MEMORY_BACKEND=helios NUM_GPUS=8 BATCH_SIZE=8 GRAD_ACCUM=1 NUM_EPOCHS=500 SAVE_EVERY=500
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_ENABLED=true WANDB_PROJECT=memorywam WANDB_MODE=online
export WANDB_ENTITY=wangmingxin-tsinghua-university WANDB_BASE_URL=https://api.wandb.ai
export TOS_UPLOAD=1 TOS_BUCKET=ai-dev TOS_PREFIX=spidy.wang/memorywam_ckpt
export http_proxy=http://192.168.39.105:7890 https_proxy=http://192.168.39.105:7890
export NO_PROXY=localhost,127.0.0.1,ai-dev.tos-cn-beijing.ivolces.com,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset MAX_STEPS STARWAM_MEMORY_LIMIT_GIB
INIT=/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_release/robotwin_uncond_3cam_384.pt
[[ -r "$INIT" ]] || { echo "Missing official RoboTwin checkpoint" >&2; exit 1; }
if [[ -z "$config_only" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY in the queue environment}"
  : "${TOS_ACCESS_KEY:?Set TOS_ACCESS_KEY in the queue environment}"
  : "${TOS_SECRET_KEY:?Set TOS_SECRET_KEY in the queue environment}"
fi
for task in "${tasks[@]}"; do
  unset PYTHONPATH RMBENCH_TASK_ROOT RMBENCH_COVER_BLOCKS_LEROBOT
  unset MEMORYWAM_COVER_BLOCKS_STATS MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS MEMORYWAM_COVER_BLOCKS_TEXT_CACHE
  unset WANDB_RUN_ID RUN_ID
  export RMBENCH_TASK="$task"
  model_name=fastwam-anchor-helios-robotwininit
  init_args=("++model.helios_config.backbone_init_checkpoint=$INIT")
  if [[ "$task" == press_button ]]; then
    model_name=fastwam-anchor-helios
    init_args=("++model.helios_config.backbone_init_checkpoint=null")
  fi
  name="${task}_${model_name}_bs8_ga1_gpu8_ep500_save500_$(date +%Y%m%d_%H%M%S)"
  export OUTPUT_DIR="$ROOT/runs/$name"
  export WANDB_DIR="$OUTPUT_DIR"
  echo "Starting $task: epochs=500 bs/gpu=8 ga=1 gpu=8 save_every=500 name=$name"
  opts=()
  [[ "$config_only" == --config ]] && opts+=(--config)
  bash scripts/train_with_logging.sh "${opts[@]}" \
    "wandb.name=$name" save_every=500 save_final_checkpoint=true \
    resume=null "${init_args[@]}"
done
