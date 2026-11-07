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
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
XGB_N_ESTIMATORS="${XGB_N_ESTIMATORS:-1000000}"
DATA_DIR="${DATA_DIR:-data/raw}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-benchmark/multiseed_${INPUT_DIM}}"
SEEDS_STRING="${SEEDS:-41 42 43 44 45}"
MIN_SEEDS="${MIN_SEEDS:-5}"
MODALITIES_STRING="${MODALITIES:-cnmr hnmr ir msms_pos msms_neg}"
# "legacy" keeps every model's historical preprocessing. Any other spec or
# ablation config name (see spectroscopy_qml.benchmarking.preprocessing) is
# applied identically to all four models.
# A single step name applies ONLY that step; every other step stays neutral
# (linear resample, no smoothing, intensity, no normalization). Selectable values:
#   Normalization: snv minmax max area l2 quantile pqn
#   Smoothing:     savgol (25,4)  gaussian (sigma 2)  moving_average (window 5)  lorentz (gamma 3, half width 15)
#   Representation: sqrt log1p derivative
#   Resampling:    linear bin_mean bin_max fir
#   none / shared_mssd  linear interpolation only
#   Custom spec:   unlisted keys stay neutral, e.g. "norm=snv" or "smooth=savgol:11:3,norm=area"
#   Ablation presets (built on Lorentz + SNV): shared_lorentz_snv resample_bin_max smooth_savgol norm_area
PREPROCESSING="${PREPROCESSING:-legacy}"
# TTN contractions can overflow in float16/bfloat16 on real batches. Keep the
# benchmark numerically stable in float32 unless AMP is explicitly requested.
TTN_AMP="${TTN_AMP:-0}"
# Splits depend only on labels and seed, so ablations can share one split dir.
SPLIT_DIR="${SPLIT_DIR:-$EXPERIMENT_DIR/splits}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
# Optional guard for short, requeued Slurm allocations. Before starting a new
# model, requeue the job when less than this many seconds remain. This keeps a
# model run inside one allocation instead of restarting it after a timeout.
MIN_START_REMAINING_SECONDS="${MIN_START_REMAINING_SECONDS:-0}"

read -r -a SEED_VALUES <<< "$SEEDS_STRING"
read -r -a MODALITY_VALUES <<< "$MODALITIES_STRING"

if (( ${#SEED_VALUES[@]} < MIN_SEEDS )); then
    echo "At least $MIN_SEEDS seed(s) are required; received: $SEEDS_STRING" >&2
    exit 2
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

slurm_time_left_seconds() {
    local value days=0 hours=0 minutes=0 seconds=0
    local -a fields

    [[ -n "${SLURM_JOB_ID:-}" ]] || return 1
    value="$(squeue -h -j "$SLURM_JOB_ID" -o '%L')" || return 1
    [[ -n "$value" && "$value" != "N/A" && "$value" != "UNLIMITED" ]] || return 1

    if [[ "$value" == *-* ]]; then
        days="${value%%-*}"
        value="${value#*-}"
    fi
    IFS=: read -r -a fields <<< "$value"
    case "${#fields[@]}" in
        3) hours="${fields[0]}"; minutes="${fields[1]}"; seconds="${fields[2]}" ;;
        2) minutes="${fields[0]}"; seconds="${fields[1]}" ;;
        1) seconds="${fields[0]}" ;;
        *) return 1 ;;
    esac
    printf '%s\n' "$((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}

requeue_before_new_model_if_needed() {
    local remaining
    (( MIN_START_REMAINING_SECONDS > 0 )) || return 0
    remaining="$(slurm_time_left_seconds)" || return 0
    if (( remaining < MIN_START_REMAINING_SECONDS )); then
        echo "Only ${remaining}s remain; not starting another model. Requeueing ${SLURM_JOB_ID}."
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
}

# Fail fast on a typo before hours of training.
"$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing "$PREPROCESSING" > /dev/null

run_if_missing() {
    local metrics_path="$1"
    local expected_model="$2"
    local expected_seed="$3"
    local expected_split="$4"
    local expected_limit_option="$5"
    local expected_limit="$6"
    shift 6
    if [[ -f "$metrics_path" ]] && "$PYTHON" scripts/validate_benchmark_result.py \
        --metrics "$metrics_path" \
        --model "$expected_model" \
        --preprocessing "$PREPROCESSING" \
        --input-dim "$INPUT_DIM" \
        --seed "$expected_seed" \
        --split-path "$expected_split" \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
        "$expected_limit_option" "$expected_limit"; then
        echo "Skipping completed run: $metrics_path"
        return
    fi
    if [[ -f "$metrics_path" ]]; then
        echo "Existing result is stale or incompatible; rerunning: $metrics_path"
    fi
    requeue_before_new_model_if_needed
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

        run_if_missing "$run_dir/cnn/$baseline_output_dir/original/metrics.json" cnn "$seed" "$split_path" --max-epochs "$EPOCHS" \
            "$PYTHON" benchmark/cnn/scripts/run_cnn_jung_baseline.py \
            --analytical_data "$DATA_DIR" \
            --base_out_path "$run_dir/cnn" \
            --columns "$column" \
            --seed "$seed" \
            --input_dim "$INPUT_DIM" \
            --split_path "$split_path" \
            --preprocessing "$PREPROCESSING" \
            --epochs "$EPOCHS" \
            --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
            --no_kfold

        xgb_device="$DEVICE"
        if [[ "$xgb_device" == "auto" || "$xgb_device" == "mps" ]]; then
            xgb_device="cpu"
        fi
        run_if_missing "$run_dir/xgboost/$baseline_output_dir/original/metrics.json" xgboost "$seed" "$split_path" --n-estimators "$XGB_N_ESTIMATORS" \
            "$PYTHON" benchmark/xgb/scripts/run_xgb_baseline.py \
            --analytical_data "$DATA_DIR" \
            --base_out_path "$run_dir/xgboost" \
            --columns "$column" \
            --seed "$seed" \
            --input_dim "$INPUT_DIM" \
            --split_path "$split_path" \
            --device "$xgb_device" \
            --preprocessing "$PREPROCESSING" \
            --n-estimators "$XGB_N_ESTIMATORS" \
            --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
            --no_log_learning_curve \
            --no_kfold

        mps_device="$DEVICE"
        if [[ "$mps_device" == "mps" ]]; then
            echo "MPS classifier does not support Apple's MPS backend; using CPU for this model."
            mps_device="cpu"
        fi
        run_if_missing "$run_dir/mps/results/metrics.json" mps "$seed" "$split_path" --max-epochs "$EPOCHS" \
            "$PYTHON" scripts/run_mps_benchmark.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$run_dir/mps" \
            --modality "$modality" \
            --input-dim "$INPUT_DIM" \
            --split-path "$split_path" \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
            --device "$mps_device" \
            --preprocessing "$PREPROCESSING"

        run_if_missing "$run_dir/ttn/metrics.json" ttn "$seed" "$split_path" --max-epochs "$EPOCHS" \
            "$PYTHON" "$ttn_script" \
            --data-dir "$DATA_DIR" \
            --output-dir "$run_dir/ttn" \
            --split-path "$split_path" \
            --input-dim "$INPUT_DIM" \
            --seed "$seed" \
            --device "$DEVICE" \
            --epochs "$EPOCHS" \
            --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
            "$([[ "$TTN_AMP" == "1" ]] && printf '%s' '--amp' || printf '%s' '--no-amp')" \
            --preprocessing "$PREPROCESSING"
    done
done

if [[ "$SKIP_AGGREGATE" != "1" ]]; then
    "$PYTHON" scripts/aggregate_multiseed_results.py \
        --experiment-dir "$EXPERIMENT_DIR"
fi
