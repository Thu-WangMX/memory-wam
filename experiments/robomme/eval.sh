#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SIM="${ROBOMME_SIM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/robomme_native}"
export PYTHONPATH="$REPO:$SIM/src:$SIM/src/src:$SIM/maniskill${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export XDG_CACHE_HOME="${ROBOMME_USER_CACHE:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$REPO"
exec "$SIM/.venv/bin/python" -B experiments/robomme/eval.py "$@"
