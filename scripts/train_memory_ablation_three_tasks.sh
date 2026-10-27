#!/usr/bin/env bash
# Sequential independent single-task runs. Caller supplies credentials and GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:?Usage: train_memory_ablation_three_tasks.sh VARIANT [--task-group press_swapT|swap_blocks|all] [hydra overrides]}"
shift
case "$variant" in
  wo_long|uniform_mid|reverse_scale|wo_anchor|long_less_temporal|long_less_spatial|multiscale|anchor_only|recent19_dense|full_history_dense) ;;
  *) echo "Unknown memory variant: $variant" >&2; exit 1 ;;
esac
group=all
if [[ "${1:-}" == --task-group ]]; then
  group="${2:?--task-group requires press_swapT, swap_blocks or all}"
  shift 2
fi
case "$group" in
  press_swapT) tasks=(press_button swap_T) ;;
  swap_blocks) tasks=(swap_blocks) ;;
  all) tasks=(press_button swap_T swap_blocks) ;;
  *) echo "Invalid task group: $group" >&2; exit 1 ;;
esac
for task in "${tasks[@]}"; do
  bash scripts/train_memory_ablation_task.sh "$variant" "$task" "$@"
done
