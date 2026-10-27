#!/usr/bin/env bash
set -euo pipefail
export ROBOMME_TRAIN_TASK=robomme_helios_past_action_flow
exec bash "$(dirname "$0")/train_past_action_8gpu.sh" "$@"
