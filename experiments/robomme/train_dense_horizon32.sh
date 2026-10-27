#!/usr/bin/env bash
# Same model/optimizer recipe as train_horizon32.sh, with every-frame starts.
set -euo pipefail
variant="${1:-past32}"
if (( $# )); then shift; fi
export ROBOMME_SAMPLE_STRIDE="${ROBOMME_SAMPLE_STRIDE:-1}"
export ROBOMME_DENSE_LATENTS="${ROBOMME_DENSE_LATENTS:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/phase-offset16-latents-v1}"
case "$ROBOMME_SAMPLE_STRIDE" in
  1|16) ;;
  *) echo 'ROBOMME_SAMPLE_STRIDE must be 1 or 16' >&2; exit 1 ;;
esac
if [[ "$ROBOMME_SAMPLE_STRIDE" == 1 && "${ROBOMME_LAUNCH_DRY_RUN:-0}" != 1 \
      && ! -f "$ROBOMME_DENSE_LATENTS/manifest.json" ]]; then
  echo "Dense VAE cache is not finalized: $ROBOMME_DENSE_LATENTS/manifest.json" >&2
  echo 'Run experiments/robomme/prepare_dense_latents.sh before training.' >&2
  exit 1
fi
# Make the sampling variant visible in run names on all workers.
if [[ -z "${RUN_ID:-}" && -n "${MLP_TASK_ID:-}" ]]; then
  export RUN_ID="robomme_all16_dense${ROBOMME_SAMPLE_STRIDE}_${variant}_${MLP_TASK_ID}"
fi
exec bash "$(dirname "$0")/train_horizon32.sh" "$variant" \
  "data.train.sample_stride=$ROBOMME_SAMPLE_STRIDE" \
  "data.train.dense_latents_path=$ROBOMME_DENSE_LATENTS" "$@"
