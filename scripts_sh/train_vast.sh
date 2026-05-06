#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/my_conda/etc/profile.d/conda.sh
conda activate /root/my_conda/envs/cl

EXTRA=""
MODALITIES=()
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${ROOT_DIR}/configs/train/vast_lora_volume.yaml"
OUTPUT_DIR=""
EVAL_STEPS=""
LOSS_MODE=""
WANDB_MODE=""
DO_EVAL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --modality|--modalities)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODALITIES+=("$1")
        shift
      done
      if [[ "${#MODALITIES[@]}" -eq 0 ]]; then
        echo "--modality requires at least one modality value" >&2
        exit 2
      fi
      ;;
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
    --do_eval)
      DO_EVAL="$2"
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

ARGS=(--config "${CONFIG}")
if [[ "${#MODALITIES[@]}" -gt 0 ]]; then
  ARGS+=(--modality "${MODALITIES[@]}")
fi
if [[ -n "${EXTRA}" ]]; then
  ARGS+=(--extra_modalities "${EXTRA}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${EVAL_STEPS}" ]]; then
  ARGS+=(--eval_steps "${EVAL_STEPS}")
fi
if [[ -n "${DO_EVAL}" ]]; then
  ARGS+=(--do_eval "${DO_EVAL}")
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
