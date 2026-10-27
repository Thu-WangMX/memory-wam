#!/usr/bin/env bash
# Train one RMBench task from scratch; use separate invocations for each task.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
variant="${1:?Usage: train_memory_ablation_task.sh VARIANT TASK [hydra overrides]}"
task="${2:?Specify a single RMBench task}"
shift 2
case "$variant" in
  full_history_dense) default_bs=2; default_ga=4; default_save=2500 ;;
  recent19_dense) default_bs=8; default_ga=2; default_save=5000 ;;
  multiscale|anchor_only) default_bs=8; default_ga=2; default_save=2500 ;;
  wo_long|uniform_mid|reverse_scale|wo_anchor|long_less_temporal|long_less_spatial) default_bs=16; default_ga=1; default_save=2500 ;;
  *) echo 'Unknown memory variant' >&2; exit 1 ;;
esac
case "$task" in
  press_button|swap_blocks|swap_T) ;;
  *) echo 'Expected press_button, swap_blocks or swap_T' >&2; exit 1 ;;
esac
# Do not reuse another task's env-resolved paths or any mixed-training resume.
unset RMBENCH_TASK_ROOT RMBENCH_COVER_BLOCKS_LEROBOT MEMORYWAM_COVER_BLOCKS_STATS
unset MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS MEMORYWAM_COVER_BLOCKS_TEXT_CACHE
unset MAX_STEPS RESUME_CHECKPOINT
export RMBENCH_TASK="$task" MEMORY_BACKEND=helios
export NUM_GPUS="${NUM_GPUS:-8}" BATCH_SIZE="${BATCH_SIZE:-$default_bs}" GRAD_ACCUM="${GRAD_ACCUM:-$default_ga}"
export NUM_EPOCHS="${NUM_EPOCHS:-500}" SAVE_EVERY="${SAVE_EVERY:-$default_save}" NUM_WORKERS="${NUM_WORKERS:-16}"
source scripts/distributed_env.sh
source scripts/comparison_env.sh
# Resolve one canonical profile shared by model tests and all launchers.
readarray -t profile < <(python scripts/memory_variant_args.py "$variant")
[[ "${#profile[@]}" == 4 ]] || { echo 'Invalid memory profile' >&2; exit 1; }
sizes="${profile[0]}"; long="${profile[1]}"; mid="${profile[2]}"; anchors="${profile[3]}"
noise="${HISTORY_AUGMENTATION:-true}"
[[ "$noise" == true || "$noise" == false ]] || { echo 'Invalid HISTORY_AUGMENTATION' >&2; exit 1; }
lr="${LEARNING_RATE:-0.0001}"; scheduler="${LR_SCHEDULER:-cosine}"
init_checkpoint=null
init_tag=no-robotwininit
# Every single-task memory ablation: Swap T starts from official RoboTwin weights.
if [[ "$task" == swap_T ]]; then
  init_checkpoint="${ROBOTWIN_INIT_CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_release/robotwin_uncond_3cam_384.pt}"
  [[ -r "$init_checkpoint" ]] || { echo "Missing RoboTwin initialization checkpoint: $init_checkpoint" >&2; exit 1; }
  init_tag=robotwininit
fi
name="${task}_fastwam-${variant}-${init_tag}_histaug${noise}_bs${BATCH_SIZE}_ga${GRAD_ACCUM}_gpu${NUM_GPUS}_ep${NUM_EPOCHS}_save${SAVE_EVERY}_lr${lr}_${scheduler}_${RUN_ID}"
export OUTPUT_DIR="$ROOT/runs/$name"
export FASTWAM_CHECKPOINT_WHOLE_LAYER="${FASTWAM_CHECKPOINT_WHOLE_LAYER:-1}"
export FASTWAM_FFN_CHUNK_SIZE="${FASTWAM_FFN_CHUNK_SIZE:-256}"
export FASTWAM_QKV_CHUNK_SIZE="${FASTWAM_QKV_CHUNK_SIZE:-256}"
exec bash scripts/train_with_logging.sh "$@" "wandb.name=$name" resume=null \
  "++model.helios_config.backbone_init_checkpoint=$init_checkpoint" \
  "model.helios_config.memory_variant=$variant" "model.helios_config.history_sizes=$sizes" \
  "model.helios_config.patch_long=$long" "model.helios_config.patch_mid=$mid" \
  "model.helios_config.anchor_frames=$anchors" "model.helios_config.condition_augmentation=$noise" \
  "data.train.dataset_dirs=[$RMBENCH_COVER_BLOCKS_LEROBOT]" \
  "data.train.pretrained_norm_stats=$MEMORYWAM_COVER_BLOCKS_STATS" \
  "data.train.full_kv_cache_path=$MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS" \
  "data.train.text_embedding_cache_dir=$MEMORYWAM_COVER_BLOCKS_TEXT_CACHE" \
  "learning_rate=$lr" "lr_scheduler_type=$scheduler"
