#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/my_conda/etc/profile.d/conda.sh
conda activate /root/my_conda/envs/cl

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT_DIR}/scripts/eval_retrieval.py" \
  --config "${ROOT_DIR}/configs/eval/didemo.yaml" \
  "${ARGS[@]}"
