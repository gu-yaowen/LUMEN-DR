#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

python -m lmdda \
  --mode export_embeddings \
  --dataset Cdataset \
  --run-name final5fold_c_best \
  --device cuda:0 \
  --seed 42 \
  --num-folds 5 \
  --hidden-dim 128 \
  --num-heads 2 \
  --num-layers 1 \
  --contrastive-dim 128 \
  --activation gelu \
  --dropout 0.05 \
  --proj-norm-type batchnorm \
  --drug-feature-source pretrained \
  --disease-feature-source pretrained \
  --protein-feature-source pretrained
