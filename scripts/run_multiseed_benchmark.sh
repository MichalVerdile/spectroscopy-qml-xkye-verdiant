#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 INPUT_DIM" >&2
    exit 2
fi

INPUT_DIM="$1"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-200}"
DATA_DIR="${DATA_DIR:-data/raw}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-benchmark/multiseed_${INPUT_DIM}}"
SEEDS_STRING="${SEEDS:-41 42 43 44 45}"
MODALITIES_STRING="${MODALITIES:-cnmr hnmr ir msms_pos msms_neg}"

read -r -a SEED_VALUES <<< "$SEEDS_STRING"
read -r -a MODALITY_VALUES <<< "$MODALITIES_STRING"

if (( ${#SEED_VALUES[@]} < 5 )); then
    echo "At least five seeds are required; received: $SEEDS_STRING" >&2
    exit 2
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

run_if_missing() {
    local metrics_path="$1"
    shift
    if [[ -f "$metrics_path" ]]; then
        echo "Skipping completed run: $metrics_path"
        return
    fi
    "$@"
}

for modality in "${MODALITY_VALUES[@]}"; do
    case "$modality" in
        cnmr)
            column="c_nmr_spectra"
            baseline_output_dir="cnmr"
            ttn_script="src/spectroscopy_qml/cnmr/tree_tensor_network/train.py"
            ;;
        hnmr)
            column="h_nmr_spectra"
            baseline_output_dir="hnmr"
            ttn_script="src/spectroscopy_qml/hnmr/tree_tensor_network/train.py"
            ;;
        ir)
            column="ir_spectra"
            baseline_output_dir="ir"
            ttn_script="src/spectroscopy_qml/ir/tree_tensor_network/train.py"
            ;;
        msms_pos)
            column="pos_msms"
            baseline_output_dir="pos_msms"
            ttn_script="src/spectroscopy_qml/msms_pos/tree_tensor_network/train.py"
            ;;
        msms_neg)
            column="neg_msms"
            baseline_output_dir="neg_msms"
            ttn_script="src/spectroscopy_qml/msms_neg/tree_tensor_network/train.py"
            ;;
        *)
            echo "Unsupported modality '$modality'." >&2
            exit 2
            ;;
    esac

    for seed in "${SEED_VALUES[@]}"; do
        run_dir="$EXPERIMENT_DIR/$modality/seed_$seed"
        split_path="$EXPERIMENT_DIR/splits/${modality}_seed_${seed}.npz"
        mkdir -p "$run_dir/cnn" "$run_dir/xgboost" "$run_dir/mps" "$run_dir/ttn"

        if [[ ! -f "$split_path" ]]; then
            "$PYTHON" scripts/create_multilabel_split.py \
                --data-dir "$DATA_DIR" \
                --modality "$modality" \
                --input-dim "$INPUT_DIM" \
                --seed "$seed" \
                --output "$split_path"
        fi

        run_if_missing "$run_dir/cnn/$baseline_output_dir/original/metrics.json" \
            "$PYTHON" benchmark/cnn/scripts/run_cnn_jung_baseline.py \
            --analytical_data "$DATA_DIR" \
            --base_out_path "$run_dir/cnn" \
            --columns "$column" \
            --seed "$seed" \
            --input_dim "$INPUT_DIM" \
            --split_path "$split_path" \
            --no_kfold

        xgb_device="$DEVICE"
        if [[ "$xgb_device" == "auto" || "$xgb_device" == "mps" ]]; then
            xgb_device="cpu"
        fi
        run_if_missing "$run_dir/xgboost/$baseline_output_dir/original/metrics.json" \
            "$PYTHON" benchmark/xgb/scripts/run_xgb_baseline.py \
            --analytical_data "$DATA_DIR" \
            --base_out_path "$run_dir/xgboost" \
            --columns "$column" \
            --seed "$seed" \
            --input_dim "$INPUT_DIM" \
            --split_path "$split_path" \
            --device "$xgb_device" \
            --no_log_learning_curve \
            --no_kfold

        mps_device="$DEVICE"
        if [[ "$mps_device" == "mps" ]]; then
            echo "MPS classifier does not support Apple's MPS backend; using CPU for this model."
            mps_device="cpu"
        fi
        run_if_missing "$run_dir/mps/results/metrics.json" \
            "$PYTHON" scripts/run_mps_benchmark.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$run_dir/mps" \
            --modality "$modality" \
            --input-dim "$INPUT_DIM" \
            --split-path "$split_path" \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            --device "$mps_device"

        run_if_missing "$run_dir/ttn/metrics.json" \
            "$PYTHON" "$ttn_script" \
            --data-dir "$DATA_DIR" \
            --output-dir "$run_dir/ttn" \
            --split-path "$split_path" \
            --input-dim "$INPUT_DIM" \
            --seed "$seed" \
            --device "$DEVICE" \
            --epochs "$EPOCHS"
    done
done

"$PYTHON" scripts/aggregate_multiseed_results.py \
    --experiment-dir "$EXPERIMENT_DIR"
