#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
STACK="$ROBODOJO_RUNTIME/eval-stack"
EVAL_SEED="${EVAL_SEED:-0}"
test -d "$STACK/Assets/Eval_Layout/RoboDojo/arx_x5/$EVAL_SEED" || { echo "Missing RoboDojo layout seed: $EVAL_SEED" >&2; exit 1; }
SIM_ENV="${ROBODOJO_SIM_VENV:-/mnt/vepfs01/output/royce.luo/dojo4090/runtime/isaaclab-venv}"
export PATH="$SIM_ENV/bin:$PATH"
export PYTHONPATH="$STACK:$STACK/XPolicyLab:$ROBODOJO_RUNTIME/eval-deps"
export XDG_CACHE_HOME="$ROBODOJO_RUNTIME/eval-client-cache"
export TMPDIR="$ROBODOJO_RUNTIME/tmp"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=Y
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,${POLICY_HOST:-127.0.0.1}"
export no_proxy="$NO_PROXY"
export ROBODOJO_RUN_ID="${EVAL_RUN_ID:?Set a fresh EVAL_RUN_ID containing checkpoint identity}"
export EVAL_NUM="${EVAL_NUM:-1}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ROBODOJO_RUNTIME/eval-client-logs"
cd "$STACK"
bash scripts/eval_policy.sh \
  --root_dir "$STACK" --task_name "${EVAL_TASK:-cover_blocks}" \
  --env_cfg_type arx_x5 --device_id "${SIM_GPU_ID:?Select a free simulator GPU}" \
  --policy_name FastWAMAnchorHelios --port "${POLICY_PORT:-8096}" \
  --host "${POLICY_HOST:-127.0.0.1}" --protocol ws \
  --seed "$EVAL_SEED" --additional_info "ckpt_name=$ROBODOJO_RUN_ID,action_type=joint" \
  2>&1 | tee "$ROBODOJO_RUNTIME/eval-client-logs/$ROBODOJO_RUN_ID.log"
