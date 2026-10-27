#!/usr/bin/env bash
# Performance-only runtime setup; no Hydra/model/sampling contract changes.
export ROBOMME_FUTURE_CLIP_CACHE_ROOT="${ROBOMME_FUTURE_CLIP_CACHE_ROOT:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/future-clips-exact-v1}"
if [[ "${ROBOMME_LAUNCH_DRY_RUN:-0}" != 1 ]]; then
  _speed_ready=/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/runtime/speed_optimization_20261026/READY.json
  if [[ ! -s "$_speed_ready" ]]; then
    echo 'Waiting for validated RoboMME training-speed patch; this submitted job will continue automatically.'
    until [[ -s "$_speed_ready" ]]; do sleep 5; done
  fi
  unset _speed_ready
fi
