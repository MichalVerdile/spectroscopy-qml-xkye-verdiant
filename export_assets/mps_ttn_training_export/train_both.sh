#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"

python src/spectroscopy_qml/ir/MPS_TTN/run.py \
  --model both \
  --device cuda \
  --ttn-device cuda \
  --spectra-cache src/spectroscopy_qml/ir/MPS_TTN/results/export_run/shared_spectra.npz \
  --split-path src/spectroscopy_qml/ir/MPS_TTN/results/export_run/shared_split.npz \
  --output-dir src/spectroscopy_qml/ir/MPS_TTN/results/export_run \
  --no-ttn-compile
