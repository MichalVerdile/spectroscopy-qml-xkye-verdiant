#!/bin/bash
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1

TTN_CHECKPOINT="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/ttn_ir_best.pt"
TTN_CONFIG="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/run_config.json"
SPECTRA_CACHE="data/cache/ir_spectra_len1800_snv_all.npz"
SPLIT_PATH="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_1/results/data_split_seed42_all.npz"
OUTPUT_DIR="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_4/results"

echo "=== Schritt 1: 1D-CNN Specialist Head trainieren (experiment10_2_4) ==="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_4/train.py \
  --spectra-cache "$SPECTRA_CACHE" \
  --ttn-checkpoint "$TTN_CHECKPOINT" \
  --ttn-config "$TTN_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --split-path "$SPLIT_PATH" \
  --cnn-channels 32 \
  --pool-size 8 \
  --device mps

echo ""
echo "=== Schritt 2: Ensemble evaluieren ==="
python -u src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_4/ensemble_predict.py \
  --specialist-dir "$OUTPUT_DIR" \
  --spectra-cache "$SPECTRA_CACHE" \
  --ttn-checkpoint "$TTN_CHECKPOINT" \
  --ttn-config "$TTN_CONFIG" \
  --split-path "$SPLIT_PATH"

echo ""
echo "=== Pipeline abgeschlossen ==="
