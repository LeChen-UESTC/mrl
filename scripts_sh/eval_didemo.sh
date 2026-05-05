#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate cl

python "${ROOT_DIR}/scripts/eval_retrieval.py" \
  --config "${ROOT_DIR}/configs/eval/didemo.yaml" \
  "$@"
