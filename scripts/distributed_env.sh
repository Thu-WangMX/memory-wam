#!/usr/bin/env bash
# NUM_GPUS is global; GPUS_PER_NODE is local. No GPU mask is overwritten.
export NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
host_short="${HOSTNAME:-$(hostname)}"; host_short="${host_short%%.*}"
if [[ -z "${NODE_RANK:-}" ]]; then
  if [[ -n "${MLP_ROLE_INDEX:-}" ]]; then NODE_RANK="$MLP_ROLE_INDEX"
  elif [[ "$host_short" =~ -worker-([0-9]+)$ ]]; then NODE_RANK="${BASH_REMATCH[1]}"
  elif [[ "$NNODES" == 1 ]]; then NODE_RANK=0
  else echo 'Set NODE_RANK on each worker' >&2; return 1; fi
fi
export NODE_RANK
export NUM_GPUS="${NUM_GPUS:-$((NNODES * ${GPUS_PER_NODE:-8}))}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-$((NUM_GPUS / NNODES))}"
[[ "$NNODES" =~ ^[1-9][0-9]*$ && "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ && "$NODE_RANK" =~ ^[0-9]+$ ]] || return 1
(( NODE_RANK < NNODES && NUM_GPUS == NNODES * GPUS_PER_NODE )) || { echo 'Inconsistent distributed topology' >&2; return 1; }
if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ "$NNODES" == 1 ]]; then MASTER_ADDR=127.0.0.1
  elif [[ -n "${MLP_WORKER_0_HOST:-}" ]]; then MASTER_ADDR="$MLP_WORKER_0_HOST"
  elif [[ "$host_short" =~ -worker-[0-9]+$ ]]; then
    base="${host_short%-worker-*}"
    fqdn="$(hostname -f)"
    if [[ "$fqdn" == "$host_short".* ]]; then MASTER_ADDR="${base}-worker-0.${fqdn#*.}"
    else
      search_domain="$(awk '/^search / {print $2; exit}' /etc/resolv.conf)"
      if [[ "$search_domain" == *.svc.cluster.local ]]; then MASTER_ADDR="${base}-worker-0.${base}-worker.${search_domain}"
      else MASTER_ADDR="${base}-worker-0"; fi
    fi
  else echo 'Set MASTER_ADDR to worker-0 address' >&2; return 1; fi
fi
export MASTER_ADDR MASTER_PORT="${MASTER_PORT:-29500}"
if (( NNODES > 1 )); then
  [[ "$MASTER_ADDR" != localhost && "$MASTER_ADDR" != 127.0.0.1 ]] || { echo 'Multi-node MASTER_ADDR cannot be loopback' >&2; return 1; }
  if [[ -z "${RUN_ID:-}" ]]; then
    [[ "$host_short" =~ -worker-[0-9]+$ ]] || { echo 'Set the same RUN_ID on both workers' >&2; return 1; }
    export RUN_ID="${host_short%-worker-*}"
  fi
else export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"; fi
