#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/my_conda/etc/profile.d/conda.sh
conda activate /root/my_conda/envs/cl

python "${ROOT_DIR}/scripts/preprocess_dataset.py" \
  --config "${ROOT_DIR}/configs/preprocess/vast_train.yaml" \
  "$@"
