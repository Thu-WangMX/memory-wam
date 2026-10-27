#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export PYTHONPATH="$PYTHONPATH:$ROBODOJO_RUNTIME/eval-stack:$ROBODOJO_RUNTIME/eval-stack/XPolicyLab:$ROBODOJO_RUNTIME/eval-deps"
cd "$ROBODOJO_FASTWAM_REPO"
exec python -B -m experiments.robodojo.serve "$@"
