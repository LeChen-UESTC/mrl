#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate cl

EXTRA="audio"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${ROOT_DIR}/configs/train/vast_lora_volume.yaml"
OUTPUT_DIR=""
EVAL_STEPS=""
LOSS_MODE=""
WANDB_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extra|--extra_modalities)
      EXTRA="$2"
      shift 2
      ;;
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --eval_steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --loss_mode)
      LOSS_MODE="$2"
      shift 2
      ;;
    --wandb_mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ARGS=(--config "${CONFIG}" --extra_modalities "${EXTRA}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${EVAL_STEPS}" ]]; then
  ARGS+=(--eval_steps "${EVAL_STEPS}")
fi
if [[ -n "${LOSS_MODE}" ]]; then
  ARGS+=(--loss_mode "${LOSS_MODE}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  ARGS+=(--wandb_mode "${WANDB_MODE}")
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT_DIR}/scripts/train_lora_volume.py" \
  "${ARGS[@]}"
