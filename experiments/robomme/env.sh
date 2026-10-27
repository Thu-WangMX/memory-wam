#!/usr/bin/env bash
export ROBOMME_FASTWAM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROBOMME_TRAIN_VENV="${ROBOMME_TRAIN_VENV:-/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv}"
export ROBOMME_KEVIN_DATA="${ROBOMME_KEVIN_DATA:-/mnt/vepfs02/output/kevin_wang/memorywam/data/robomme_all16_anchor1_event_k8_v1_full}"
export ROBOMME_DATA_ROOT="${ROBOMME_DATA_ROOT:-$ROBOMME_KEVIN_DATA/lerobot/multitask}"
export ROBOMME_HELIOS_LATENTS="${ROBOMME_HELIOS_LATENTS:-$ROBOMME_KEVIN_DATA/phase_latents}"
export ROBOMME_HELIOS_PREPARED="${ROBOMME_HELIOS_PREPARED:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/prepared-anchor-helios}"
export PATH="$ROBOMME_TRAIN_VENV/bin:$PATH"
export ROBOMME_PROTOCOL_PYTHON="${ROBOMME_PROTOCOL_PYTHON:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/python311}"
export PYTHONPATH="$ROBOMME_PROTOCOL_PYTHON:$ROBOMME_FASTWAM_REPO/runtime/python:$ROBOMME_FASTWAM_REPO/src:$ROBOMME_FASTWAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [[ -z "${CUDA_HOME:-}" && -x /mnt/vepfs01/output/spidy.wang/runtime-robomme/cuda-12.2/bin/nvcc ]]; then
  export CUDA_HOME=/mnt/vepfs01/output/spidy.wang/runtime-robomme/cuda-12.2
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/vepfs01/output/spidy.wang/runtime-robomme/torch_extensions/$(hostname)}"
export MAX_JOBS="${MAX_JOBS:-4}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
