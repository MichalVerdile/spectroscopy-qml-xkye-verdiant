#!/usr/bin/env bash
# Preprocessing / input-representation ablation on top of the multi-seed benchmark.
#
# The default run is legacy plus 13 shared configs; the shared ones feed identical
# inputs to CNN, XGBoost, MPS and TTN, and
# all configs and input dims reuse one set of shared splits, so differences
# between configs are paired per seed. Results land in
#   $ABLATION_DIR/dim_<INPUT_DIM>/<config>/<modality>/seed_<seed>/<model>/...
# and scripts/aggregate_preprocessing_ablation.py turns them into separate
# architecture, preprocessing and representation tables.
#
# Usage: scripts/run_preprocessing_ablation.sh INPUT_DIM [INPUT_DIM ...]
# Env:   CONFIGS      space separated config names (default: legacy + 13 shared configs;
#                     --list --include-extra shows further, redundant variants)
#        MODALITIES   default "ir cnmr" (the two modalities in the paper)
#        SEEDS, EPOCHS, DEVICE, DATA_DIR, PYTHON as in run_multiseed_benchmark.sh
#        ABLATION_DIR default benchmark/preprocessing_ablation
# List configs: PYTHONPATH=src python -m spectroscopy_qml.benchmarking.preprocessing --list

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 INPUT_DIM [INPUT_DIM ...]" >&2
    exit 2
fi

PYTHON="${PYTHON:-python}"
ABLATION_DIR="${ABLATION_DIR:-benchmark/preprocessing_ablation}"
export MODALITIES="${MODALITIES:-ir cnmr}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CONFIGS:-}" ]]; then
    CONFIGS="$("$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing --list --include-legacy | cut -f1 | tr '\n' ' ')"
fi
read -r -a CONFIG_VALUES <<< "$CONFIGS"

# Validate every config before the first (long) training run starts.
for config in "${CONFIG_VALUES[@]}"; do
    "$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing "$config" > /dev/null
done

for input_dim in "$@"; do
    for config in "${CONFIG_VALUES[@]}"; do
        modalities="$MODALITIES"
        # IR spectra already have 1800 points: at >= 1800 every resampling method
        # returns the reference input, so those runs would only duplicate shared_mssd.
        if [[ "$config" == resample_* ]] && (( input_dim >= 1800 )); then
            modalities="$(tr ' ' '\n' <<< "$MODALITIES" | grep -vx ir | tr '\n' ' ' | xargs || true)"
        fi
        if [[ -z "$modalities" ]]; then
            echo "=== Skipping input_dim=$input_dim config=$config (no-op for $MODALITIES) ==="
            continue
        fi
        echo "=== Ablation: input_dim=$input_dim config=$config modalities=$modalities ==="
        MODALITIES="$modalities" \
        PREPROCESSING="$config" \
        EXPERIMENT_DIR="$ABLATION_DIR/dim_${input_dim}/$config" \
        SPLIT_DIR="$ABLATION_DIR/splits" \
        SKIP_AGGREGATE=1 \
            scripts/run_multiseed_benchmark.sh "$input_dim"
    done
done

"$PYTHON" scripts/aggregate_preprocessing_ablation.py --ablation-dir "$ABLATION_DIR"
