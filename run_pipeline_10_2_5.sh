#!/usr/bin/env bash
set -euo pipefail

TTN_CHECKPOINT="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/ttn_ir_best.pt"
TTN_CONFIG="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/run_config.json"
OUTPUT_DIR="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_5/results"

echo "=== Schritt 1: Quanvolutional Specialist Heads trainieren (experiment10_2_5) ==="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_5/train.py \
  --ttn-checkpoint "$TTN_CHECKPOINT" \
  --ttn-config "$TTN_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 150 \
  --batch-size 1024 \
  --learning-rate 3e-4 \
  --loss-type focal \
  --threshold-mode per_class \
  --threshold-target-metric per_class_f1

echo
echo "=== Schritt 2: TTN 10.2 + Quanvolutional Specialist Override evaluieren ==="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_5/ensemble_predict.py \
  --specialist-dir "$OUTPUT_DIR" \
  --ttn-checkpoint "$TTN_CHECKPOINT" \
  --ttn-config "$TTN_CONFIG"
