#!/usr/bin/env bash
set -euo pipefail
export ROBOMME_TRAIN_TASK=robomme_helios_past_action_sparse16x8_flow
exec bash "$(dirname "$0")/train_past_action_flow_32gpu.sh" "$@"
