#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate cl

NFRAMES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nframes)
      NFRAMES="$2"
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

python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/msrvtt_test.yaml" \
  "${ARGS[@]}"
python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/didemo_test.yaml" \
  "${ARGS[@]}"
python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/vatex_test.yaml" \
  "${ARGS[@]}"
