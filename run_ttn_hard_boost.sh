#!/bin/bash
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Hard class indices: Acyl halide(1), Enamine(13), Enol(14), Hydrazine(18),
# Hydrazone(19), Imine(21), Ketone(24), Sulfide(28), Sulfoxide(33), Thioamide(35)
HARD_INDICES="1,13,14,18,19,21,24,28,33,35"

OUTPUT_DIR="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/hard_boost_run_$(date +%Y%m%d_%H%M%S)"
SPLIT_PATH="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/data_split_seed42_all.npz"

echo "=== TTN 10.2 Retrain mit Hard Class Boost ==="
echo "Output: $OUTPUT_DIR"
echo ""

python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/train.py \
  --data-dir data/raw \
  --output-dir "$OUTPUT_DIR" \
  --split-path "$SPLIT_PATH" \
  --device mps \
  --epochs 200 \
  --batch-size 1024 \
  --loss-type focal \
  --focal-gamma 2.0 \
  --pos-weight-power 1.0 \
  --pos-weight-max 50.0 \
  --hard-class-boost 5.0 \
  --hard-class-indices "$HARD_INDICES" \
  --chi 64 \
  --early-stopping-patience 20 \
  --min-epochs-before-stopping 30

echo ""
echo "=== Training abgeschlossen ==="
