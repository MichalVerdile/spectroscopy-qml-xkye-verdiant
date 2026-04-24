#!/usr/bin/env bash
set -euo pipefail

CACHE_PATH="${CACHE_PATH:-data/cache/ir_spectra_len1800_snv_all.npz}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment14/results}"

COMMON_ARGS=(
  --spectra-cache "$CACHE_PATH"
  --feature-source raw
  --compression-dim 32
  --quantum-qubits 4
  --quantum-layers 3
  --batch-size 256
  --feature-batch-size 512
  --epochs 150
  --learning-rate 3e-4
  --threshold-mode per_class
  --threshold-target-metric per_class_f1
)

for variant in raw_quantum pca_quantum tn_quantum tn_classical; do
  echo
  echo "=== Experiment 14 :: $variant ==="
  python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment14.train \
    --variant "$variant" \
    --output-dir "$BASE_OUTPUT_DIR/$variant" \
    "${COMMON_ARGS[@]}"
done
