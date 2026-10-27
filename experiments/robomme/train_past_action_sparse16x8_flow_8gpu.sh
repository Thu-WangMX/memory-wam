#!/usr/bin/env bash
set -euo pipefail
export ROBOMME_TRAIN_TASK=robomme_helios_past_action_sparse16x8_flow
export BATCH_SIZE="${BATCH_SIZE:-16}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MAX_STEPS="${MAX_STEPS:-20000}" SAVE_EVERY="${SAVE_EVERY:-2500}"
exec bash "$(dirname "$0")/train_past_action_8gpu.sh" "$@"
