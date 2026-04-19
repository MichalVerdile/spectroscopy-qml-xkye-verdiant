#!/usr/bin/env bash
# Runs 5–8: Focal Loss (γ=2) on best γ=3 configs
# Run after Run 4 (10.3 z_score) finishes.
#
# Uses the same data split as Run 1 for fair comparison.

set -e

export PYTORCH_ENABLE_MPS_FALLBACK=1

SPLIT="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/data_split_seed42_all.npz"

# ── Run 5: 10.2 percentile + Focal Loss γ=2 ──────────────────────────────────
echo "=========================================="
echo "Run 5: Exp 10.2 | percentile | Focal γ=2"
echo "=========================================="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/train.py \
  --lorentz-norm-mode percentile \
  --loss-type focal \
  --focal-gamma 2.0 \
  --split-path "$SPLIT" \
  --no-compile \
  --output-dir "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_$(date +%Y%m%d_%H%M%S)_focal_g2_percentile"

# ── Run 6: 10.2 percentile + Focal Loss γ=1 ──────────────────────────────────
echo "=========================================="
echo "Run 6: Exp 10.2 | percentile | Focal γ=1"
echo "=========================================="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/train.py \
  --lorentz-norm-mode percentile \
  --loss-type focal \
  --focal-gamma 1.0 \
  --split-path "$SPLIT" \
  --no-compile \
  --output-dir "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_$(date +%Y%m%d_%H%M%S)_focal_g1_percentile"

# ── Run 7: 10.3 percentile + Focal Loss γ=2 ──────────────────────────────────
echo "=========================================="
echo "Run 7: Exp 10.3 | percentile | Focal γ=2"
echo "=========================================="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_3/train.py \
  --voigt-norm-mode percentile \
  --loss-type focal \
  --focal-gamma 2.0 \
  --split-path "$SPLIT" \
  --no-compile \
  --output-dir "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_3/results/full_dataset_run_$(date +%Y%m%d_%H%M%S)_focal_g2_percentile"

# ── Run 8: 10.4 SG + Focal Loss γ=2 ─────────────────────────────────────────
echo "=========================================="
echo "Run 8: Exp 10.4 | SG max_abs | Focal γ=2"
echo "=========================================="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_4/train.py \
  --loss-type focal \
  --focal-gamma 2.0 \
  --split-path "$SPLIT" \
  --no-compile \
  --output-dir "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_4/results/full_dataset_run_$(date +%Y%m%d_%H%M%S)_focal_g2_maxabs"

echo "=========================================="
echo "Runs 5–8 abgeschlossen."
echo "=========================================="
