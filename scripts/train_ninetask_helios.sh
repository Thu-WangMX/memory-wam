#!/usr/bin/env bash
# One mixed dataset, one shared normalization and one model initialization.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export RMBENCH_TASK=observe_and_pickup MEMORY_BACKEND=helios
export NUM_GPUS="${NUM_GPUS:-8}" BATCH_SIZE="${BATCH_SIZE:-8}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-500}" SAVE_EVERY="${SAVE_EVERY:-1000}"
export WANDB_PROJECT="${WANDB_PROJECT:-memorywam}" WANDB_ENTITY="${WANDB_ENTITY:-wangmingxin-tsinghua-university}"
source scripts/distributed_env.sh
unset MAX_STEPS
source scripts/comparison_env.sh
noise="${HISTORY_AUGMENTATION:?Set HISTORY_AUGMENTATION=true or false explicitly}"
[[ "$noise" == true || "$noise" == false ]] || { echo 'Invalid HISTORY_AUGMENTATION' >&2; exit 1; }
[[ -f runtime/ninetask_helios/manifest.json && -f runtime/ninetask_helios/tasks.json ]] || { echo 'Run scripts/prepare_ninetask_helios.py first' >&2; exit 1; }
datasets="$(python -c 'import json; print("[" + ",".join(x["dataset"] for x in json.load(open("runtime/ninetask_helios/tasks.json"))) + "]")')"
lr="${LEARNING_RATE:-0.0001}"
scheduler="${LR_SCHEDULER:-cosine}"
name="ninetask_fastwam-anchor-helios-no-robotwininit_histaug${noise}_bs${BATCH_SIZE}_ga${GRAD_ACCUM}_gpu${NUM_GPUS}_ep${NUM_EPOCHS}_save${SAVE_EVERY}_lr${lr}_${scheduler}_${RUN_ID}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
if [[ -n "$resume_checkpoint" ]]; then
  [[ -e "$resume_checkpoint" ]] || { echo "Resume checkpoint does not exist: $resume_checkpoint" >&2; exit 1; }
  resume_tag="$(basename "$resume_checkpoint")"
  name="${name}_resume-${resume_tag%.pt}"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/$name}"
exec bash scripts/train_with_logging.sh "$@" "wandb.name=$name" "resume=${resume_checkpoint:-null}" \
 ++model.helios_config.backbone_init_checkpoint=null \
 "model.helios_config.condition_augmentation=$noise" \
 "data.train.dataset_dirs=$datasets" data.train.pretrained_norm_stats=null \
 "data.train.full_kv_cache_path=$ROOT/runtime/ninetask_helios" \
 "data.train.text_embedding_cache_dir=$ROOT/runtime/ninetask_helios/text_cache" \
 "learning_rate=$lr" "lr_scheduler_type=$scheduler"
