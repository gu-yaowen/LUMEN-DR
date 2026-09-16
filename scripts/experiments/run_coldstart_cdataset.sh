#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

split_mode="${1:-}"
task_id="${2:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "$split_mode" || -z "${task_id}" ]]; then
  echo "Usage: $0 <cold_drug|cold_disease> <fold_id>"
  exit 2
fi

if [[ "$split_mode" != "cold_drug" && "$split_mode" != "cold_disease" ]]; then
  echo "Invalid split_mode=$split_mode"
  exit 2
fi

if (( task_id < 0 || task_id >= 5 )); then
  echo "Invalid fold_id=${task_id}; expected 0..4"
  exit 2
fi

run_name="cold_c_${split_mode}"
fold_dir="$ROOT/outputs/Cdataset/$run_name/folds/fold_${task_id}"
if [[ -f "$fold_dir/metrics.json" ]]; then
  echo "Skipping existing Cdataset ${split_mode} fold ${task_id}"
  exit 0
fi

python -m lmdda \
  --mode train \
  --dataset Cdataset \
  --run-name "$run_name" \
  --device cuda:0 \
  --seed 42 \
  --num-folds 5 \
  --fold-index "$task_id" \
  --split-mode "$split_mode" \
  --epochs 2000 \
  --patience 200 \
  --hidden-dim 128 \
  --num-heads 2 \
  --num-layers 1 \
  --contrastive-dim 128 \
  --activation gelu \
  --dropout 0.05 \
  --proj-norm-type batchnorm \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --scheduler none \
  --use-pos-weight 1 \
  --drug-feature-source pretrained \
  --disease-feature-source pretrained \
  --protein-feature-source pretrained \
  --use-contrastive 1 \
  --contrastive-weight 0.10 \
  --contrastive-temperature 0.2
