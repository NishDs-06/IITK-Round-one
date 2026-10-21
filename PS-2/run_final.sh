#!/bin/bash
# The exact command that produced adv_images/.
# Add --no-transfer for the pure query-only variant (no substitute networks).
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u src/run_attack.py \
  --data data/p2_data --out adv_images --tag final \
  --seeds 4 --bsearch 20 --iters 0 \
  --refine-rounds 1 --geo 2000 --walk 2000 \
  --rescue-rounds 2 --rescue-frac 0.30 --rescue-geo 700 --rescue-walk 700 \
  --grid-rounds 1 --grid-geo 800 --grid-walk 800 \
  --greedy-rounds 5 --verify-shrink 0.97 --max-batch 2048 "$@"
