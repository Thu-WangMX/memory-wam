#!/usr/bin/env bash
# Controlled Helios ablations: same data, anchors, masks, objectives and optimizer.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
variant="${1:?Usage: bash scripts/train_ninetask_memory_ablation.sh anchor_only|recent19_dense|full_history_dense [hydra overrides]}"
shift
case "$variant" in
  anchor_only) sizes='[0,0,1]'; long='[4,8,8]'; mid='[2,4,4]' ;;
  recent19_dense|full_history_dense) sizes='[16,2,1]'; long='[1,2,2]'; mid='[1,2,2]' ;;
  *) echo "Unknown ablation: $variant" >&2; exit 1 ;;
esac
export RMBENCH_TASK=observe_and_pickup MEMORY_BACKEND=helios
default_bs=4; default_ga=4
if [[ "$variant" == full_history_dense ]]; then default_bs=1; default_ga=16; fi
export NUM_GPUS="${NUM_GPUS:-8}" BATCH_SIZE="${BATCH_SIZE:-$default_bs}" GRAD_ACCUM="${GRAD_ACCUM:-$default_ga}"
export NUM_EPOCHS="${NUM_EPOCHS:-500}" SAVE_EVERY="${SAVE_EVERY:-5000}" NUM_WORKERS="${NUM_WORKERS:-16}"
source scripts/distributed_env.sh
source scripts/comparison_env.sh
noise="${HISTORY_AUGMENTATION:-true}"
[[ "$noise" == true || "$noise" == false ]] || { echo 'Invalid HISTORY_AUGMENTATION' >&2; exit 1; }
[[ -f runtime/ninetask_helios/manifest.json && -f runtime/ninetask_helios/tasks.json ]]
datasets="$(python -c 'import json; print("[" + ",".join(x["dataset"] for x in json.load(open("runtime/ninetask_helios/tasks.json"))) + "]")')"
lr="${LEARNING_RATE:-0.0001}"; scheduler="${LR_SCHEDULER:-cosine}"
name="ninetask_fastwam-${variant}-no-robotwininit_histaug${noise}_bs${BATCH_SIZE}_ga${GRAD_ACCUM}_gpu${NUM_GPUS}_ep${NUM_EPOCHS}_save${SAVE_EVERY}_lr${lr}_${scheduler}_${RUN_ID}"
resume="${RESUME_CHECKPOINT:-}"
if [[ -n "$resume" ]]; then
  [[ -f "$resume" ]] || { echo "Missing resume checkpoint: $resume" >&2; exit 1; }
  name="${name}_resume-$(basename "$resume" .pt)"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/$name}"
# Exact token-wise chunking reduces projection/FFN temporaries without truncating history.
export FASTWAM_CHECKPOINT_WHOLE_LAYER="${FASTWAM_CHECKPOINT_WHOLE_LAYER:-1}"
export FASTWAM_FFN_CHUNK_SIZE="${FASTWAM_FFN_CHUNK_SIZE:-256}"
export FASTWAM_QKV_CHUNK_SIZE="${FASTWAM_QKV_CHUNK_SIZE:-256}"
exec bash scripts/train_with_logging.sh "$@" "wandb.name=$name" "resume=${resume:-null}" \
  ++model.helios_config.backbone_init_checkpoint=null \
  "model.helios_config.memory_variant=$variant" "model.helios_config.history_sizes=$sizes" \
  "model.helios_config.patch_long=$long" "model.helios_config.patch_mid=$mid" \
  model.helios_config.anchor_frames=2 "model.helios_config.condition_augmentation=$noise" \
  "data.train.dataset_dirs=$datasets" data.train.pretrained_norm_stats=null \
  "data.train.full_kv_cache_path=$ROOT/runtime/ninetask_helios" \
  "data.train.text_embedding_cache_dir=$ROOT/runtime/ninetask_helios/text_cache" \
  "learning_rate=$lr" "lr_scheduler_type=$scheduler"
