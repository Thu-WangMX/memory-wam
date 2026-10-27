#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBODOJO_FASTWAM_REPO"
export PYTHONPATH="$PYTHONPATH:$ROBODOJO_RUNTIME/eval-stack:$ROBODOJO_RUNTIME/eval-stack/XPolicyLab:$ROBODOJO_RUNTIME/eval-deps"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
exec python -u -B -m experiments.robodojo.eval_campaign
