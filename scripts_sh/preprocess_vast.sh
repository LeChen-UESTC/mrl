#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /Users/bytedance/.pyenv/versions/3.10.15/envs/env310/bin/activate

python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/vast_train.yaml" \
  "$@"
