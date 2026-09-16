#!/usr/bin/env bash
set -euo pipefail

# Run one standard pair-level cross-validation experiment from the repository
# root. Feature caches are intentionally external and must exist under
# features/<DATASET>/ before training starts.
dataset="${1:-Bdataset}"
run_name="${2:-hgt_main}"
device="${3:-cuda:0}"

python -m lmdda \
  --mode train \
  --dataset "$dataset" \
  --run-name "$run_name" \
  --device "$device" \
  --data-root data \
  --features-root features \
  --output-root outputs
