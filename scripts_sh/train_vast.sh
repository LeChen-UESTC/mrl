#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/my_conda/etc/profile.d/conda.sh
conda activate /root/my_conda/envs/cl

EXTRA=""
MODALITIES=()
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${ROOT_DIR}/configs/train/vast_lora_volume.yaml"
EPOCHS=""
MAX_STEPS=""
BATCH_SIZE=""
EVAL_BATCH_SIZE=""
EVAL_NFRAMES=""
NUM_WORKERS=""
LEARNING_RATE=""
WEIGHT_DECAY=""
MAX_GRAD_NORM=""
LOG_STEPS=""
SAVE_STEPS=""
EVAL_STEPS=""
LOSS_MODE=""
WANDB_MODE=""
DO_EVAL=""
LORA_R=""
LORA_ALPHA=""
LORA_DROPOUT=""
LORA_TARGET_MODULES=()
LORA_BIAS=""

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
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --eval_batch_size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --eval_nframes)
      EVAL_NFRAMES="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --learning_rate|--lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --max_grad_norm)
      MAX_GRAD_NORM="$2"
      shift 2
      ;;
    --log_steps)
      LOG_STEPS="$2"
      shift 2
      ;;
    --save_steps)
      SAVE_STEPS="$2"
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
    --lora_r)
      LORA_R="$2"
      shift 2
      ;;
    --lora_alpha)
      LORA_ALPHA="$2"
      shift 2
      ;;
    --lora_dropout)
      LORA_DROPOUT="$2"
      shift 2
      ;;
    --lora_target_modules)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        LORA_TARGET_MODULES+=("$1")
        shift
      done
      if [[ "${#LORA_TARGET_MODULES[@]}" -eq 0 ]]; then
        echo "--lora_target_modules requires at least one module value" >&2
        exit 2
      fi
      ;;
    --lora_bias)
      LORA_BIAS="$2"
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
if [[ -n "${EPOCHS}" ]]; then
  ARGS+=(--epochs "${EPOCHS}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  ARGS+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
  ARGS+=(--eval_batch_size "${EVAL_BATCH_SIZE}")
fi
if [[ -n "${EVAL_NFRAMES}" ]]; then
  ARGS+=(--eval_nframes "${EVAL_NFRAMES}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  ARGS+=(--num_workers "${NUM_WORKERS}")
fi
if [[ -n "${LEARNING_RATE}" ]]; then
  ARGS+=(--learning_rate "${LEARNING_RATE}")
fi
if [[ -n "${WEIGHT_DECAY}" ]]; then
  ARGS+=(--weight_decay "${WEIGHT_DECAY}")
fi
if [[ -n "${MAX_GRAD_NORM}" ]]; then
  ARGS+=(--max_grad_norm "${MAX_GRAD_NORM}")
fi
if [[ -n "${LOG_STEPS}" ]]; then
  ARGS+=(--log_steps "${LOG_STEPS}")
fi
if [[ -n "${SAVE_STEPS}" ]]; then
  ARGS+=(--save_steps "${SAVE_STEPS}")
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
if [[ -n "${LORA_R}" ]]; then
  ARGS+=(--lora_r "${LORA_R}")
fi
if [[ -n "${LORA_ALPHA}" ]]; then
  ARGS+=(--lora_alpha "${LORA_ALPHA}")
fi
if [[ -n "${LORA_DROPOUT}" ]]; then
  ARGS+=(--lora_dropout "${LORA_DROPOUT}")
fi
if [[ "${#LORA_TARGET_MODULES[@]}" -gt 0 ]]; then
  ARGS+=(--lora_target_modules "${LORA_TARGET_MODULES[@]}")
fi
if [[ -n "${LORA_BIAS}" ]]; then
  ARGS+=(--lora_bias "${LORA_BIAS}")
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT_DIR}/scripts/train_lora_volume.py" \
  "${ARGS[@]}"
