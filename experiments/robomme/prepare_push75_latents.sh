#!/usr/bin/env bash
# One explicit invocation prepares fresh two-view base + 15 offset caches.
# ROBOMME_PREP_STAGE=base or dense can resume the two stages independently.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOMME_FASTWAM_REPO"
source_base="${ROBOMME_SOURCE_BASE_CACHE:-$ROBOMME_HELIOS_LATENTS}"
source_prepared="${ROBOMME_SOURCE_PREPARED:-$ROBOMME_HELIOS_PREPARED}"
cache_root="${ROBOMME_PUSH75_CACHE_ROOT:-${ROBOMME_TWO_VIEW_CACHE_ROOT:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/two-view-square256-v1}}"
base="${ROBOMME_TWO_VIEW_BASE:-$cache_root/base}"
prepared="${ROBOMME_TWO_VIEW_PREPARED:-$cache_root/prepared}"
dense="${ROBOMME_TWO_VIEW_DENSE:-$cache_root/dense}"
stage="${ROBOMME_PREP_STAGE:-all}"
[[ "$stage" == all || "$stage" == base || "$stage" == dense ]] || { echo 'ROBOMME_PREP_STAGE must be all, base or dense' >&2; exit 1; }
python - "$base" "$prepared" "$dense" "$ROBOMME_DATA_ROOT" "$source_base" "$source_prepared" <<'PY'
import sys
from scripts.prepare_robomme_two_view_base import runtime_output
paths = [runtime_output(p, *sys.argv[4:]) for p in sys.argv[1:4]]
for i, left in enumerate(paths):
    for right in paths[i+1:]:
        if left == right or left in right.parents or right in left.parents:
            raise ValueError('Base/prepared/dense outputs must be separate directories')
PY
IFS=',' read -r -a gpus <<< "${ROBOMME_PREP_GPUS:-0}"
(( ${#gpus[@]} > 0 )) || exit 1
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU IDs must be integers' >&2; exit 1; }
done
pids=()
stop_children() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap stop_children INT TERM
if [[ "$stage" != dense ]]; then
  mkdir -p "$base/logs"
  for index in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" python -B scripts/prepare_robomme_two_view_base.py \
      --dataset-root "$ROBOMME_DATA_ROOT" --source-base-cache "$source_base" \
      --source-prepared "$source_prepared" --output "$base" --prepared-output "$prepared" \
      --device cuda:0 --memory-gib "${ROBOMME_PREP_MEMORY_GIB:-8}" \
      --num-shards "${#gpus[@]}" --shard-index "$index" \
      > "$base/logs/shard-$index.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "Base preparation failed; inspect $base/logs. Rerun verifies completed episodes." >&2; exit 1; }
  pids=()
  python -B scripts/prepare_robomme_two_view_base.py \
    --dataset-root "$ROBOMME_DATA_ROOT" --source-base-cache "$source_base" \
    --source-prepared "$source_prepared" --output "$base" --prepared-output "$prepared" --finalize
fi
if [[ "$stage" != base ]]; then
  export ROBOMME_VIEW_LAYOUT=two_view_horizontal
  export ROBOMME_HELIOS_LATENTS="$base" ROBOMME_HELIOS_PREPARED="$prepared"
  export ROBOMME_DENSE_LATENTS="$dense"
  bash experiments/robomme/prepare_dense_latents.sh
fi
printf 'Two-view base: %s\nPrepared assets: %s\nDense offsets: %s\n' "$base" "$prepared" "$dense"
