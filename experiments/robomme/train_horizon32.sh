#!/usr/bin/env bash
# Submit once; Volc invokes the same entry on each allocated worker.
set -euo pipefail
variant="${1:-past32}"
if (( $# )); then shift; fi
case "$variant" in
  past32)
    export ROBOMME_TRAIN_TASK=robomme_helios_past32_flow
    export ROBOMME_MEMORY_VARIANT=multiscale_long99 ;;
  pastfuture32)
    export ROBOMME_TRAIN_TASK=robomme_helios_pastfuture32_flow
    export ROBOMME_MEMORY_VARIANT=multiscale_long99 ;;
  pastfuture32_light)
    export ROBOMME_TRAIN_TASK=robomme_helios_pastfuture32_light_flow
    case "${ROBOMME_LONG_COMPRESSION:-temporal}" in
      temporal) export ROBOMME_MEMORY_VARIANT=long99_less_temporal ;;
      spatial) export ROBOMME_MEMORY_VARIANT=long99_less_spatial ;;
      *) echo 'ROBOMME_LONG_COMPRESSION must be temporal or spatial' >&2; exit 1 ;;
    esac ;;
  *) echo 'Choose past32, pastfuture32, or pastfuture32_light' >&2; exit 1 ;;
esac
export BATCH_SIZE="${BATCH_SIZE:-16}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MAX_STEPS="${MAX_STEPS:-20000}" SAVE_EVERY="${SAVE_EVERY:-2500}"
case "${ROBOMME_TRAIN_GPUS:-32}" in
  32) exec bash "$(dirname "$0")/train_past_action_flow_32gpu.sh" "$@" ;;
  8) exec bash "$(dirname "$0")/train_past_action_8gpu.sh" "$@" ;;
  *) echo 'ROBOMME_TRAIN_GPUS must be 8 or 32' >&2; exit 1 ;;
esac
