#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs src/spectroscopy_qml/ir/MPS_TTN/results/mac_seq
export PYTHONPATH="$PWD/src"
export PYTORCH_ENABLE_MPS_FALLBACK=1

python3 -u - <<'PY'
import sys
from pathlib import Path

from spectroscopy_qml.ir.mps_encoder_final.config import TRAINING_CONFIG
from spectroscopy_qml.ir.MPS_TTN.run import main

TRAINING_CONFIG.batch_size = 128
TRAINING_CONFIG.num_workers = 0
TRAINING_CONFIG.pin_memory = False
TRAINING_CONFIG.parallel_fold_workers = 1
TRAINING_CONFIG.device = "cpu"

base = Path("src/spectroscopy_qml/ir/MPS_TTN/results/mac_seq")
sys.argv = [
    "run.py",
    "--model", "mps",
    "--device", "cpu",
    "--spectra-cache", str(base / "shared_spectra_full_mac.npz"),
    "--split-path", str(base / "shared_split_full_mac.npz"),
    "--output-dir", str(base / "mps_then_ttn"),
]
main()
PY

python3 -u src/spectroscopy_qml/ir/MPS_TTN/run.py \
  --model ttn \
  --device cpu \
  --ttn-device mps \
  --ttn-batch-size 128 \
  --ttn-num-workers 0 \
  --spectra-cache src/spectroscopy_qml/ir/MPS_TTN/results/mac_seq/shared_spectra_full_mac.npz \
  --split-path src/spectroscopy_qml/ir/MPS_TTN/results/mac_seq/shared_split_full_mac.npz \
  --output-dir src/spectroscopy_qml/ir/MPS_TTN/results/mac_seq/mps_then_ttn \
  --no-ttn-compile
