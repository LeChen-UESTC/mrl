#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate cl

NFRAMES=""
MAX_SAMPLES=""
LOG_EVERY=""
CACHE_MEDIA_FEATURES=""
CACHE_RAW_PROCESSOR_TENSORS=""
FEATURE_DTYPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nframes)
      NFRAMES="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --log_every)
      LOG_EVERY="$2"
      shift 2
      ;;
    --cache_media_features)
      CACHE_MEDIA_FEATURES="$2"
      shift 2
      ;;
    --cache_raw_processor_tensors)
      CACHE_RAW_PROCESSOR_TENSORS="$2"
      shift 2
      ;;
    --feature_dtype)
      FEATURE_DTYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument for downstream preprocessing: $1" >&2
      exit 2
      ;;
  esac
done

ARGS=()
if [[ -n "${NFRAMES}" ]]; then
  ARGS+=(--nframes "${NFRAMES}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${LOG_EVERY}" ]]; then
  ARGS+=(--log_every "${LOG_EVERY}")
fi
if [[ -n "${CACHE_MEDIA_FEATURES}" ]]; then
  ARGS+=(--cache_media_features "${CACHE_MEDIA_FEATURES}")
fi
if [[ -n "${CACHE_RAW_PROCESSOR_TENSORS}" ]]; then
  ARGS+=(--cache_raw_processor_tensors "${CACHE_RAW_PROCESSOR_TENSORS}")
fi
if [[ -n "${FEATURE_DTYPE}" ]]; then
  ARGS+=(--feature_dtype "${FEATURE_DTYPE}")
fi

python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/msrvtt_test.yaml" \
  "${ARGS[@]}"
python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/didemo_test.yaml" \
  "${ARGS[@]}"
python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/vatex_test.yaml" \
  "${ARGS[@]}"
