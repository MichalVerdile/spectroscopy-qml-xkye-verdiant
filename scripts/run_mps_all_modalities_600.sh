#!/usr/bin/env bash
# Train the MPS classifier at input dimension 600 for every modality and seed.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNNER_PATH="$PWD/scripts/$(basename "${BASH_SOURCE[0]}")"

INPUT_DIM=600
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/raw}"
# Preprocessing: a single step name applies ONLY that step; every other step stays
# neutral (linear resample, no smoothing, intensity, no normalization).
# Selectable values (PREPROCESSING=<value>):
#   shared_mssd / none  linear interpolation only, no further processing
#   Normalization: snv minmax max area l2 quantile pqn
#   Smoothing:     savgol (25,4)  gaussian (sigma 2)  moving_average (window 5)  lorentz (gamma 3, half width 15)
#   Representation: sqrt log1p derivative
#   Resampling:    linear bin_mean bin_max fir
#   Custom spec:   unlisted keys stay neutral, e.g. "norm=snv" or "smooth=savgol:11:3,norm=area"
#                  (keys: resample, smooth, repr, norm; see src/spectroscopy_qml/benchmarking/preprocessing.py)
#   legacy         historical per-model preprocessing
#   Ablation presets (built on Lorentz + SNV): shared_lorentz_snv resample_bin_max smooth_savgol norm_area
#
# Default (PREPROCESSING unset): run ALL 15 combinations of FIR resampling, Gaussian
# smoothing, sqrt representation and SNV normalization, each into
# $EXPERIMENT_DIR/<combination> (default benchmark/multiseed_600/combinations).
# Setting PREPROCESSING runs only that single preprocessing, as before.
source scripts/preprocessing_combinations.sh
if sweep_preprocessing_combinations "$RUNNER_PATH" "${EXPERIMENT_DIR:-benchmark/multiseed_600/combinations}"; then
    exit 0
fi
PREPROCESSING="${PREPROCESSING:-shared_mssd}"
SEEDS_STRING="${SEEDS:-41 42 43 44 45}"
MODALITIES_STRING="${MODALITIES:-cnmr hnmr ir msms_pos msms_neg}"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-auto}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-benchmark/multiseed_600/$PREPROCESSING}"
SPLIT_DIR="${SPLIT_DIR:-benchmark/shared_splits_600}"

if [[ "$DEVICE" == "mps" ]]; then
    echo "The MPS classifier does not support Apple's MPS backend; using CPU."
    DEVICE=cpu
fi

read -r -a SEED_VALUES <<< "$SEEDS_STRING"
read -r -a MODALITY_VALUES <<< "$MODALITIES_STRING"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing "$PREPROCESSING" > /dev/null

for modality in "${MODALITY_VALUES[@]}"; do
    case "$modality" in
        cnmr|hnmr|ir|msms_pos|msms_neg) ;;
        *) echo "Unsupported modality '$modality'." >&2; exit 2 ;;
    esac

    for seed in "${SEED_VALUES[@]}"; do
        split_path="$SPLIT_DIR/${modality}_seed_${seed}.npz"
        output_dir="$EXPERIMENT_DIR/$modality/seed_$seed/mps"
        metrics_path="$output_dir/results/metrics.json"
        mkdir -p "$output_dir"

        if [[ ! -f "$split_path" ]]; then
            "$PYTHON" scripts/create_multilabel_split.py \
                --data-dir "$DATA_DIR" \
                --modality "$modality" \
                --input-dim "$INPUT_DIM" \
                --seed "$seed" \
                --output "$split_path"
        fi

        if [[ -f "$metrics_path" ]] && "$PYTHON" scripts/validate_benchmark_result.py \
            --metrics "$metrics_path" --model mps --preprocessing "$PREPROCESSING" \
            --input-dim "$INPUT_DIM" --seed "$seed" --split-path "$split_path"; then
            echo "Skipping completed MPS run: $metrics_path"
            continue
        fi

        echo "=== MPS: modality=$modality seed=$seed input_dim=$INPUT_DIM ==="
        "$PYTHON" scripts/run_mps_benchmark.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$output_dir" \
            --modality "$modality" \
            --input-dim "$INPUT_DIM" \
            --split-path "$split_path" \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            --device "$DEVICE" \
            --preprocessing "$PREPROCESSING"
    done
done
