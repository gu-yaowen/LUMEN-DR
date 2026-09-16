#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

task_id="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${task_id}" ]]; then
  echo "Usage: $0 <task_id>  or set SLURM_ARRAY_TASK_ID"
  exit 2
fi

run_names=(
  "abl_b_full_model"
  "abl_b_drug_id"
  "abl_b_disease_id"
  "abl_b_protein_id"
  "abl_b_no_cl"
)
drug_sources=(pretrained id pretrained pretrained pretrained)
disease_sources=(pretrained pretrained id pretrained pretrained)
protein_sources=(pretrained pretrained pretrained id pretrained)
use_cls=(1 1 1 1 0)

num_variants=${#run_names[@]}
num_folds=5
max_task_id=$(( num_variants * num_folds - 1 ))

if (( task_id < 0 || task_id > max_task_id )); then
  echo "Invalid task_id=${task_id}; expected 0..${max_task_id}"
  exit 2
fi

variant_id=$(( task_id / num_folds ))
fold_id=$(( task_id % num_folds ))
run_name="${run_names[$variant_id]}"

fold_dir="$ROOT/outputs/Bdataset/$run_name/folds/fold_${fold_id}"
if [[ -f "$fold_dir/metrics.json" ]]; then
  echo "Skipping existing Bdataset variant=${variant_id} fold=${fold_id} run=${run_name}"
  exit 0
fi

python -m lmdda \
  --mode train \
  --dataset Bdataset \
  --run-name "$run_name" \
  --device cuda:0 \
  --seed 42 \
  --num-folds 5 \
  --fold-index "$fold_id" \
  --epochs 500 \
  --patience 75 \
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
  --drug-feature-source "${drug_sources[$variant_id]}" \
  --disease-feature-source "${disease_sources[$variant_id]}" \
  --protein-feature-source "${protein_sources[$variant_id]}" \
  --use-contrastive "${use_cls[$variant_id]}" \
  --contrastive-weight 0.10 \
  --contrastive-temperature 0.2
