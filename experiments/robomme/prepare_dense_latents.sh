#!/usr/bin/env bash
# Run once on one machine; choose GPUs explicitly when parallel preparation is wanted.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOMME_FASTWAM_REPO"
output="${ROBOMME_DENSE_LATENTS:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/phase-offset16-latents-v1}"
view_layout="${ROBOMME_VIEW_LAYOUT:-legacy}"
output="$(python - "$output" "$ROBOMME_DATA_ROOT" "$ROBOMME_HELIOS_LATENTS" <<'PY'
import sys
from scripts.prepare_robomme_dense_latents import safe_output
print(safe_output(*sys.argv[1:]))
PY
)"
IFS=',' read -r -a gpus <<< "${ROBOMME_PREP_GPUS:-0}"
(( ${#gpus[@]} > 0 )) || exit 1
mkdir -p "$output/logs"
pids=()
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU IDs must be integers' >&2; exit 1; }
done
stop_children() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap stop_children INT TERM
for index in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" python -B scripts/prepare_robomme_dense_latents.py \
    --output "$output" --base-cache "$ROBOMME_HELIOS_LATENTS" --view-layout "$view_layout" \
    --device cuda:0 --memory-gib "${ROBOMME_PREP_MEMORY_GIB:-6}" \
    --num-shards "${#gpus[@]}" --shard-index "$index" \
    > "$output/logs/shard-$index.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "Preparation failed; inspect $output/logs. Rerunning verifies and resumes completed offsets." >&2; exit 1; }
python -B scripts/prepare_robomme_dense_latents.py --output "$output" \
  --base-cache "$ROBOMME_HELIOS_LATENTS" --view-layout "$view_layout" --finalize
