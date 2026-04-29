#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"

RESULT_ROOT="src/spectroscopy_qml/ir/MPS_TTN/results/export_run"
CACHE_PATH="$RESULT_ROOT/shared_spectra.npz"
SPLIT_PATH="$RESULT_ROOT/shared_split.npz"

python src/spectroscopy_qml/ir/MPS_TTN/run.py \
  --model mps \
  --device cuda \
  --spectra-cache "$CACHE_PATH" \
  --split-path "$SPLIT_PATH" \
  --output-dir "$RESULT_ROOT"

python src/spectroscopy_qml/ir/MPS_TTN/run.py \
  --model ttn \
  --device cuda \
  --ttn-device cuda \
  --spectra-cache "$CACHE_PATH" \
  --split-path "$SPLIT_PATH" \
  --output-dir "$RESULT_ROOT" \
  --no-ttn-compile
