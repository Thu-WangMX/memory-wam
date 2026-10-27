#!/usr/bin/env bash
# One invocation on each of two Volc workers, eight allocated GPUs per worker.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBODOJO_FASTWAM_REPO"
: "${MLP_WORKER_NUM:?Missing MLP_WORKER_NUM}"
: "${MLP_ROLE_INDEX:?Missing MLP_ROLE_INDEX}"
: "${MLP_WORKER_0_HOST:?Missing MLP_WORKER_0_HOST}"
: "${MLP_WORKER_0_PORT:?Missing MLP_WORKER_0_PORT}"
: "${RUN_ID:?Set a fresh RUN_ID identical on both workers}"
: "${WANDB_API_KEY:?Set existing WANDB_API_KEY in job environment}"
: "${TOS_ACCESS_KEY:?Set existing TOS_ACCESS_KEY in job environment}"
: "${TOS_SECRET_KEY:?Set existing TOS_SECRET_KEY in job environment}"
[[ "$MLP_WORKER_NUM" == 2 && "$MLP_ROLE_INDEX" =~ ^[01]$ ]]
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]
export NNODES=2 NODE_RANK="$MLP_ROLE_INDEX" NUM_GPUS=16 GPUS_PER_NODE=8
export MASTER_ADDR="$MLP_WORKER_0_HOST" MASTER_PORT="$MLP_WORKER_0_PORT"
export ROBODOJO_HELIOS_LATENTS="$ROBODOJO_HELIOS_PREPARED/latents-all35"
export OUTPUT_DIR="$ROBODOJO_FASTWAM_REPO/runs/$RUN_ID"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-wangmingxin-tsinghua-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-memorywam}" WANDB_MODE=online
export WANDB_INIT_TIMEOUT=120 WANDB_DIR="$OUTPUT_DIR"
export TOS_BUCKET="${TOS_BUCKET:-ai-dev}" TOS_PREFIX="${TOS_PREFIX:-spidy.wang/memorywam_ckpt}"
export TOS_ENDPOINT="${TOS_ENDPOINT:-https://tos-cn-beijing.ivolces.com}" TOS_REGION=cn-beijing
export PYTHONPATH="$PYTHONPATH:/mnt/vepfs01/output/spidy.wang/starwam"
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,$MASTER_ADDR,.tos-cn-beijing.ivolces.com,tos-cn-beijing.ivolces.com"
export no_proxy="$NO_PROXY"
export TRITON_CACHE_DIR="$ROBODOJO_RUNTIME/cache/triton/$RUN_ID/worker-$NODE_RANK"
export TORCH_EXTENSIONS_DIR="$ROBODOJO_RUNTIME/cache/torch_extensions/$RUN_ID/worker-$NODE_RANK"
unset RESUME_CHECKPOINT NUM_EPOCHS STARWAM_MEMORY_LIMIT_GIB ACCELERATE_USE_CPU
mkdir -p "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
python -c 'import torch; assert torch.cuda.device_count() == 8, "Each worker must expose exactly 8 GPUs"'

if [[ "$NODE_RANK" == 0 ]]; then
  python -B scripts/upload_checkpoints.py --doctor
  python -B scripts/prepare_robodojo_helios.py assets \
    --root "$ROBODOJO_DATA_ROOT" --source-prepared "$ROBODOJO_RUNTIME/prepared" \
    --output "$ROBODOJO_HELIOS_PREPARED" \
    2>&1 | tee "$OUTPUT_DIR/prepare-assets.log"
fi

# The other worker waits in rendezvous while rank 0 prepares the small CPU assets.
torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  scripts/prepare_robodojo_helios.py latents \
  --root "$ROBODOJO_DATA_ROOT" --output "$ROBODOJO_HELIOS_LATENTS" --tasks all \
  2>&1 | tee "$OUTPUT_DIR/prepare-latents.worker-$NODE_RANK.log"

ROBODOJO_PREFLIGHT_ONLY=1 python -B experiments/robodojo/train.py \
  task=robodojo_helios 'data.train.tasks=[all]' batch_size=16 gradient_accumulation_steps=1 \
  2>&1 | tee "$OUTPUT_DIR/preflight.worker-$NODE_RANK.log"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 16 --num_machines 2 --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --deepspeed_multinode_launcher standard \
  experiments/robodojo/train.py task=robodojo_helios 'data.train.tasks=[all]' \
  "output_dir=$OUTPUT_DIR" batch_size=16 gradient_accumulation_steps=1 num_workers=16 \
  max_steps=20000 save_every=2500 learning_rate=0.0001 lr_scheduler_type=cosine \
  warmup_ratio=0.05 weight_decay=0.01 max_grad_norm=1.0 mixed_precision=bf16 seed=42 \
  model.helios_config.condition_augmentation=true model.helios_config.backbone_init_checkpoint=null \
  resume=null save_training_state=true save_final_checkpoint=true eval_every=0 \
  wandb.enabled=true "wandb.workspace=$WANDB_ENTITY" "wandb.project=$WANDB_PROJECT" \
  "wandb.name=$RUN_ID" wandb.mode=online \
  2>&1 | tee "$OUTPUT_DIR/train.worker-$NODE_RANK.log"

if [[ "$NODE_RANK" == 0 ]]; then
  python -B scripts/upload_checkpoints.py --run "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/tos-upload.log"
fi
