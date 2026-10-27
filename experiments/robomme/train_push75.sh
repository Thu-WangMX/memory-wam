#!/usr/bin/env bash
# Submit once: platform workers invoke this entry with the injected MLP rank.
set -euo pipefail
export ROBOMME_TRAIN_TASK=robomme_helios_push75_flow
export ROBOMME_MEMORY_VARIANT=multiscale_long75
export ROBOMME_SAMPLE_STRIDE=1
cache_root="${ROBOMME_PUSH75_CACHE_ROOT:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/two-view-square256-v1}"
export ROBOMME_HELIOS_LATENTS="${ROBOMME_HELIOS_LATENTS:-$cache_root/base}"
export ROBOMME_HELIOS_PREPARED="${ROBOMME_HELIOS_PREPARED:-$cache_root/prepared}"
export ROBOMME_DENSE_LATENTS="${ROBOMME_DENSE_LATENTS:-$cache_root/dense}"
export BATCH_SIZE="${BATCH_SIZE:-16}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MAX_STEPS="${MAX_STEPS:-20000}" SAVE_EVERY="${SAVE_EVERY:-2500}"
if [[ "${ROBOMME_LAUNCH_DRY_RUN:-0}" != 1 ]]; then
  for file in "$ROBOMME_HELIOS_LATENTS/manifest.json" "$ROBOMME_HELIOS_PREPARED/provenance.json" "$ROBOMME_DENSE_LATENTS/manifest.json"; do
    [[ -f "$file" ]] || { echo "Missing finalized camera cache: $file; run prepare_push75_latents.sh first" >&2; exit 1; }
  done
fi
if [[ -z "${RUN_ID:-}" ]]; then
  if [[ -n "${MLP_TASK_ID:-}" ]]; then
    export RUN_ID="robomme_all16_push75_past32x8_future2_dense1_${MLP_TASK_ID}"
  elif [[ "${ROBOMME_TRAIN_GPUS:-32}" == 8 ]]; then
    export RUN_ID="robomme_all16_push75_past32x8_future2_dense1_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  fi
fi
args=("data.train.full_kv_cache_path=$ROBOMME_HELIOS_LATENTS"
      "data.train.prepared_root=$ROBOMME_HELIOS_PREPARED"
      "data.train.pretrained_norm_stats=$ROBOMME_HELIOS_PREPARED/dataset_stats.json"
      "data.train.dense_latents_path=$ROBOMME_DENSE_LATENTS" data.train.sample_stride=1)
case "${ROBOMME_TRAIN_GPUS:-32}" in
  32) exec bash "$(dirname "$0")/train_past_action_flow_32gpu.sh" "${args[@]}" "$@" ;;
  8) exec bash "$(dirname "$0")/train_past_action_8gpu.sh" "${args[@]}" "$@" ;;
  *) echo 'ROBOMME_TRAIN_GPUS must be 8 or 32' >&2; exit 1 ;;
esac
