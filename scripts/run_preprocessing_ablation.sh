#!/usr/bin/env bash
# Preprocessing / input-representation ablation on top of the multi-seed benchmark.
#
# The default run is legacy plus 13 shared configs. Every config is applied with the
# same spec and the same input dim to all five spectra (IR, H-NMR, C-NMR, MS/MS+,
# MS/MS-); the shared configs also feed identical inputs to CNN, XGBoost, MPS and
# TTN. All configs reuse one set of shared splits, so differences between configs
# are paired per seed. Results land in
#   $ABLATION_DIR/dim_<INPUT_DIM>/<config>/<modality>/seed_<seed>/<model>/...
# and scripts/aggregate_preprocessing_ablation.py turns them into separate
# architecture, preprocessing and representation tables.
#
# Usage: scripts/run_preprocessing_ablation.sh INPUT_DIM   (600 or 1800; one per call)
# Env:   CONFIGS      space separated config names (default: legacy + 13 shared configs;
#                     --list --include-extra shows further, redundant variants)
#        MODALITIES   default: all five spectra (cnmr hnmr ir msms_pos msms_neg)
#        SEEDS, EPOCHS, DEVICE, DATA_DIR, PYTHON as in run_multiseed_benchmark.sh
#        ABLATION_DIR default benchmark/preprocessing_ablation
# List configs: PYTHONPATH=src python -m spectroscopy_qml.benchmarking.preprocessing --list

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 INPUT_DIM   (one input dim per call, e.g. 600 or 1800)" >&2
    exit 2
fi
INPUT_DIM="$1"

PYTHON="${PYTHON:-python}"
ABLATION_DIR="${ABLATION_DIR:-benchmark/preprocessing_ablation}"
export MODALITIES="${MODALITIES:-cnmr hnmr ir msms_pos msms_neg}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CONFIGS:-}" ]]; then
    CONFIGS="$("$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing --list --include-legacy | cut -f1 | tr '\n' ' ')"
fi
read -r -a CONFIG_VALUES <<< "$CONFIGS"

# Validate every config before the first (long) training run starts.
for config in "${CONFIG_VALUES[@]}"; do
    "$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing "$config" > /dev/null
done

for config in "${CONFIG_VALUES[@]}"; do
    echo "=== Ablation: input_dim=$INPUT_DIM config=$config modalities=$MODALITIES ==="
    PREPROCESSING="$config" \
    EXPERIMENT_DIR="$ABLATION_DIR/dim_${INPUT_DIM}/$config" \
    SPLIT_DIR="$ABLATION_DIR/splits" \
    SKIP_AGGREGATE=1 \
        scripts/run_multiseed_benchmark.sh "$INPUT_DIM"
done

"$PYTHON" scripts/aggregate_preprocessing_ablation.py --ablation-dir "$ABLATION_DIR"
