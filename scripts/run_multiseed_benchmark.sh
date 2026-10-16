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
# "legacy" keeps every model's historical preprocessing. Any other spec or
# ablation config name (see spectroscopy_qml.benchmarking.preprocessing) is
# applied identically to all four models.
PREPROCESSING="${PREPROCESSING:-legacy}"
# Splits depend only on labels and seed, so ablations can share one split dir.
SPLIT_DIR="${SPLIT_DIR:-$EXPERIMENT_DIR/splits}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

read -r -a SEED_VALUES <<< "$SEEDS_STRING"
read -r -a MODALITY_VALUES <<< "$MODALITIES_STRING"

if (( ${#SEED_VALUES[@]} < 5 )); then
    echo "At least five seeds are required; received: $SEEDS_STRING" >&2
    exit 2
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Fail fast on a typo before hours of training.
"$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing "$PREPROCESSING" > /dev/null

run_if_missing() {
    local metrics_path="$1"
    local expected_model="$2"
    local expected_seed="$3"
    local expected_split="$4"
    shift 4
    if [[ -f "$metrics_path" ]] && "$PYTHON" scripts/validate_benchmark_result.py \
        --metrics "$metrics_path" \
        --model "$expected_model" \
        --preprocessing "$PREPROCESSING" \
        --input-dim "$INPUT_DIM" \
        --seed "$expected_seed" \
        --split-path "$expected_split"; then
        echo "Skipping completed run: $metrics_path"
        return
    fi
    if [[ -f "$metrics_path" ]]; then
        echo "Existing result is stale or incompatible; rerunning: $metrics_path"
    fi
    "$@"
}

for modality in "${MODALITY_VALUES[@]}"; do
    case "$modality" in
        cnmr)
            column="c_nmr_spectra"
            baseline_output_dir="cnmr"
            ttn_script="src/spectroscopy_qml/tree_tensor_network/cnmr/train.py"
            ;;
        hnmr)
            column="h_nmr_spectra"
            baseline_output_dir="hnmr"
            ttn_script="src/spectroscopy_qml/tree_tensor_network/hnmr/train.py"
            ;;
        ir)
            column="ir_spectra"
            baseline_output_dir="ir"
            ttn_script="src/spectroscopy_qml/tree_tensor_network/ir/train.py"
            ;;
        msms_pos)
            column="pos_msms"
            baseline_output_dir="pos_msms"
            ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_pos/train.py"
            ;;
        msms_neg)
            column="neg_msms"
            baseline_output_dir="neg_msms"
            ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_neg/train.py"
            ;;
        *)
            echo "Unsupported modality '$modality'." >&2
            exit 2
            ;;
    esac

    if [[ ! -f "$ttn_script" ]]; then
        echo "TTN training script not found for '$modality': $ttn_script" >&2
        exit 2
    fi

    for seed in "${SEED_VALUES[@]}"; do
        run_dir="$EXPERIMENT_DIR/$modality/seed_$seed"
        split_path="$SPLIT_DIR/${modality}_seed_${seed}.npz"
        mkdir -p "$run_dir/cnn" "$run_dir/xgboost" "$run_dir/mps" "$run_dir/ttn"

        if [[ ! -f "$split_path" ]]; then
            "$PYTHON" scripts/create_multilabel_split.py \
                --data-dir "$DATA_DIR" \
                --modality "$modality" \
                --input-dim "$INPUT_DIM" \
                --seed "$seed" \
                --output "$split_path"
        fi

        run_if_missing "$run_dir/cnn/$baseline_output_dir/original/metrics.json" cnn "$seed" "$split_path" \
            "$PYTHON" benchmark/cnn/scripts/run_cnn_jung_baseline.py \
            --analytical_data "$DATA_DIR" \
            --base_out_path "$run_dir/cnn" \
            --columns "$column" \
            --seed "$seed" \
            --input_dim "$INPUT_DIM" \
            --split_path "$split_path" \
            --preprocessing "$PREPROCESSING" \
            --no_kfold

        xgb_device="$DEVICE"
        if [[ "$xgb_device" == "auto" || "$xgb_device" == "mps" ]]; then
            xgb_device="cpu"
        fi
        run_if_missing "$run_dir/xgboost/$baseline_output_dir/original/metrics.json" xgboost "$seed" "$split_path" \
            "$PYTHON" benchmark/xgb/scripts/run_xgb_baseline.py \
            --analytical_data "$DATA_DIR" \
            --base_out_path "$run_dir/xgboost" \
            --columns "$column" \
            --seed "$seed" \
            --input_dim "$INPUT_DIM" \
            --split_path "$split_path" \
            --device "$xgb_device" \
            --preprocessing "$PREPROCESSING" \
            --no_log_learning_curve \
            --no_kfold

        mps_device="$DEVICE"
        if [[ "$mps_device" == "mps" ]]; then
            echo "MPS classifier does not support Apple's MPS backend; using CPU for this model."
            mps_device="cpu"
        fi
        run_if_missing "$run_dir/mps/results/metrics.json" mps "$seed" "$split_path" \
            "$PYTHON" scripts/run_mps_benchmark.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$run_dir/mps" \
            --modality "$modality" \
            --input-dim "$INPUT_DIM" \
            --split-path "$split_path" \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            --device "$mps_device" \
            --preprocessing "$PREPROCESSING"

        run_if_missing "$run_dir/ttn/metrics.json" ttn "$seed" "$split_path" \
            "$PYTHON" "$ttn_script" \
            --data-dir "$DATA_DIR" \
            --output-dir "$run_dir/ttn" \
            --split-path "$split_path" \
            --input-dim "$INPUT_DIM" \
            --seed "$seed" \
            --device "$DEVICE" \
            --epochs "$EPOCHS" \
            --preprocessing "$PREPROCESSING"
    done
done

if [[ "$SKIP_AGGREGATE" != "1" ]]; then
    "$PYTHON" scripts/aggregate_multiseed_results.py \
        --experiment-dir "$EXPERIMENT_DIR"
fi
