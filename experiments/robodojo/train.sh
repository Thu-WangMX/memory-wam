#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBODOJO_FASTWAM_REPO"
if [[ "${1:-}" == --config ]]; then
  shift
  exec python -B experiments/robodojo/train.py task=robodojo_helios "$@" --cfg job --resolve
fi
run_name="robodojo_anchor_helios_$(date +%Y%m%d_%H%M%S)"
args=(task=robodojo_helios "output_dir=${OUTPUT_DIR:-$ROBODOJO_FASTWAM_REPO/runs/$run_name}" "wandb.name=$run_name" "$@")
if [[ "${ROBODOJO_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exec python -B experiments/robodojo/train.py "${args[@]}"
fi
exec accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NUM_GPUS:-8}" --num_machines 1 --machine_rank 0 \
  --main_process_port "${MASTER_PORT:-29646}" \
  experiments/robodojo/train.py "${args[@]}"
