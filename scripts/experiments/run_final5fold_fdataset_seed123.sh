#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

fold_id="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${fold_id}" ]]; then
  echo "Usage: $0 <fold_id>  or set SLURM_ARRAY_TASK_ID"
  exit 2
fi

python -m lmdda \
  --mode train \
  --dataset Fdataset \
  --run-name final5fold_f_bestcl_seed123 \
  --device cuda:0 \
  --seed 123 \
  --num-folds 5 \
  --fold-index "$fold_id" \
  --epochs 2000 \
  --patience 200 \
  --batch-size 300000 \
  --hidden-dim 128 \
  --num-heads 2 \
  --num-layers 1 \
  --contrastive-dim 128 \
  --activation gelu \
  --dropout 0.2 \
  --proj-norm-type batchnorm \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --scheduler none \
  --use-pos-weight 1 \
  --use-contrastive 1 \
  --contrastive-weight 0.02 \
  --contrastive-temperature 0.2
