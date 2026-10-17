#!/usr/bin/env bash
# Train the CNN at input dimension 1800 for every spectroscopy modality and seed.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIM=1800
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/raw}"
PREPROCESSING="${PREPROCESSING:-shared_mssd}"
SEEDS_STRING="${SEEDS:-41 42 43 44 45}"
MODALITIES_STRING="${MODALITIES:-cnmr hnmr ir msms_pos msms_neg}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-benchmark/multiseed_1800/$PREPROCESSING}"
SPLIT_DIR="${SPLIT_DIR:-benchmark/shared_splits_1800}"

read -r -a SEED_VALUES <<< "$SEEDS_STRING"
read -r -a MODALITY_VALUES <<< "$MODALITIES_STRING"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing "$PREPROCESSING" > /dev/null

for modality in "${MODALITY_VALUES[@]}"; do
    case "$modality" in
        cnmr) column="c_nmr_spectra"; baseline_output_dir="cnmr" ;;
        hnmr) column="h_nmr_spectra"; baseline_output_dir="hnmr" ;;
        ir) column="ir_spectra"; baseline_output_dir="ir" ;;
        msms_pos) column="pos_msms"; baseline_output_dir="pos_msms" ;;
        msms_neg) column="neg_msms"; baseline_output_dir="neg_msms" ;;
        *) echo "Unsupported modality '$modality'." >&2; exit 2 ;;
    esac

    for seed in "${SEED_VALUES[@]}"; do
        split_path="$SPLIT_DIR/${modality}_seed_${seed}.npz"
        output_dir="$EXPERIMENT_DIR/$modality/seed_$seed/cnn"
        metrics_path="$output_dir/$baseline_output_dir/original/metrics.json"
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
            --metrics "$metrics_path" --model cnn --preprocessing "$PREPROCESSING" \
            --input-dim "$INPUT_DIM" --seed "$seed" --split-path "$split_path"; then
            echo "Skipping completed CNN run: $metrics_path"
            continue
        fi

        echo "=== CNN: modality=$modality seed=$seed input_dim=$INPUT_DIM ==="
        "$PYTHON" benchmark/cnn/scripts/run_cnn_jung_baseline.py \
            --analytical_data "$DATA_DIR" \
            --base_out_path "$output_dir" \
            --columns "$column" \
            --seed "$seed" \
            --input_dim "$INPUT_DIM" \
            --split_path "$split_path" \
            --preprocessing "$PREPROCESSING" \
            --no_kfold
    done
done

