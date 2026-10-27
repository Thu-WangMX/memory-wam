#!/usr/bin/env bash
# Source without touching the existing RMBench or StarWAM environments.
ROBODOJO_FASTWAM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROBODOJO_FASTWAM_REPO
ROBODOJO_USER_ROOT="${ROBODOJO_USER_ROOT:-/mnt/vepfs01/output/spidy.wang}"
export ROBODOJO_RUNTIME="${ROBODOJO_RUNTIME:-$ROBODOJO_USER_ROOT/runtime-robodojo}"
export ROBODOJO_TRAIN_VENV="${ROBODOJO_TRAIN_VENV:-$ROBODOJO_USER_ROOT/runtime/starwam-libero/venv}"
export PATH="$ROBODOJO_TRAIN_VENV/bin:$PATH"
export PYTHONPATH="$ROBODOJO_FASTWAM_REPO/runtime/python:$ROBODOJO_FASTWAM_REPO/src:$ROBODOJO_FASTWAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export ROBODOJO_DATA_ROOT="${ROBODOJO_DATA_ROOT:-$ROBODOJO_RUNTIME/datasets/official/data/RoboDojo_lerobot_v21_video}"
export ROBODOJO_HELIOS_PREPARED="${ROBODOJO_HELIOS_PREPARED:-$ROBODOJO_RUNTIME/prepared-anchor-helios}"
export ROBODOJO_HELIOS_LATENTS="${ROBODOJO_HELIOS_LATENTS:-$ROBODOJO_HELIOS_PREPARED/latents-memory6}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
