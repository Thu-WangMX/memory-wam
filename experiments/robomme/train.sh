#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOMME_FASTWAM_REPO"
: "${RUN_ID:?Set a fresh RUN_ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29631}}"
source scripts/distributed_env.sh
export OUTPUT_DIR="$ROBOMME_FASTWAM_REPO/runs/$RUN_ID"
train_task="${ROBOMME_TRAIN_TASK:-robomme_helios}"
[[ "$train_task" == robomme_helios || "$train_task" == robomme_helios_past_action \
   || "$train_task" == robomme_helios_past_action_block16 \
   || "$train_task" == robomme_helios_past_action_flow \
   || "$train_task" == robomme_helios_past_action_sparse16x8_flow \
   || "$train_task" == robomme_helios_past32_flow \
   || "$train_task" == robomme_helios_pastfuture32_flow \
   || "$train_task" == robomme_helios_push75_asv \
   || "$train_task" == robomme_helios_push75_288 \
   || "$train_task" == robomme_helios_push75_flow \
   || "$train_task" == robomme_helios_pastfuture32_light_flow ]] || {
  echo 'Unknown RoboMME training task' >&2; exit 1;
}

# Resolve a named profile once for both the model and its interpolated dataset.
# An explicit Hydra variant remains supported; conflicting selectors fail early.
variant="${ROBOMME_MEMORY_VARIANT:-}"
for arg in "$@"; do
  if [[ "$arg" == model.helios_config.memory_variant=* ]]; then
    cli_variant="${arg#*=}"
    [[ -z "$variant" || "$variant" == "$cli_variant" ]] || {
      echo 'ROBOMME_MEMORY_VARIANT disagrees with the Hydra memory_variant' >&2; exit 1;
    }
    variant="$cli_variant"
  fi
done
default_variant=multiscale
case "$train_task" in
  robomme_helios_push75_288) default_variant=long75_less_temporal ;;
  robomme_helios_push75_asv|robomme_helios_push75_flow) default_variant=multiscale_long75 ;;
  robomme_helios_past32_flow|robomme_helios_pastfuture32_flow) default_variant=multiscale_long99 ;;
  robomme_helios_pastfuture32_light_flow) default_variant=long99_less_temporal ;;
esac
variant="${variant:-$default_variant}"
profile_text="$(python -B scripts/memory_variant_args.py "$variant")"
readarray -t profile <<< "$profile_text"
[[ "${#profile[@]}" == 4 ]] || { echo 'Invalid memory profile' >&2; exit 1; }
memory_args=("model.helios_config.memory_variant=$variant"
  "model.helios_config.history_sizes=${profile[0]}"
  "model.helios_config.patch_long=${profile[1]}" "model.helios_config.patch_mid=${profile[2]}"
  "model.helios_config.anchor_frames=${profile[3]}")

# "standard" runs torchrun on this worker only. Each node invokes this script;
# DeepSpeed must not also launch remote workers through pdsh/SSH.
launch=(accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml
  --num_processes "$NUM_GPUS" --num_machines "$NNODES" --machine_rank "$NODE_RANK"
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT"
  --deepspeed_multinode_launcher standard --rdzv_backend static
  --rdzv_conf "timeout=${ROBOMME_LAUNCH_TIMEOUT:-600}" --max_restarts 0
  experiments/robomme/train.py "task=$train_task" "output_dir=$OUTPUT_DIR" "${memory_args[@]}" "$@")
if [[ "${ROBOMME_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi

# Node 0 exclusively creates the fresh run. Other nodes wait for an atomic
# marker on the shared filesystem and reject inconsistent launch parameters.
timeout="${ROBOMME_LAUNCH_TIMEOUT:-600}"
[[ "$timeout" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid launch timeout' >&2; exit 1; }
signature="$(printf '%s\0' "$NUM_GPUS" "$NNODES" "$MASTER_ADDR" "$MASTER_PORT" "$train_task" \
  "$ROBOMME_DATA_ROOT" "$ROBOMME_HELIOS_LATENTS" "$ROBOMME_HELIOS_PREPARED" \
  "${ROBOMME_SAMPLE_STRIDE:-16}" "${ROBOMME_DENSE_LATENTS:-}" \
  "${ROBOMME_DREAMDOJO_LABELS:-}" \
  "$ROBOMME_TRAIN_VENV" "${DIFFSYNTH_MODEL_BASE_PATH}" "${FASTWAM_ACTION_DIT_INIT}" \
  "${memory_args[@]}" "$@" | sha256sum | cut -d' ' -f1)"
if (( NODE_RANK == 0 )); then
  mkdir "$OUTPUT_DIR"
  mkdir "$OUTPUT_DIR/launch"
  printf '%s\n' "$signature" > "$OUTPUT_DIR/launch/signature.tmp"
  mv "$OUTPUT_DIR/launch/signature.tmp" "$OUTPUT_DIR/launch/signature"
else
  deadline=$((SECONDS + timeout))
  until [[ -f "$OUTPUT_DIR/launch/signature" ]]; do
    (( SECONDS < deadline )) || { echo 'Timed out waiting for node 0 run directory' >&2; exit 1; }
    sleep 1
  done
  [[ "$(cat "$OUTPUT_DIR/launch/signature")" == "$signature" ]] || {
    echo 'Workers disagree on topology, data paths or training arguments' >&2; exit 1;
  }
fi
# Refuse a second launcher for the same node/run, including accidental retries.
mkdir "$OUTPUT_DIR/launch/worker-$NODE_RANK"
log="$OUTPUT_DIR/train.worker-$NODE_RANK.log"
if (( NNODES == 1 )); then log="$OUTPUT_DIR/train.log"; fi
"${launch[@]}" 2>&1 | tee "$log"
