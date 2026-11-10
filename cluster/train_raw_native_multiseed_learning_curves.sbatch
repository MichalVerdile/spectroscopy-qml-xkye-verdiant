#!/usr/bin/env bash
# Native-resolution raw-data benchmark for CNN, MPS, and TTN.
#
# One array task handles one seed/modality pair and runs the three models in
# sequence.  NMR and MS/MS use their native 10,000-point representation; IR
# uses its native 1,800-point representation.  No smoothing, normalization,
# nonlinear representation, or dimensionality reduction is applied.
#
# The shared seed-specific splits from the 600-point preprocessing benchmark
# are reused deliberately.  Splits depend only on labels and seed, so this
# keeps the train/validation/test molecules identical for the comparison.
# Every trainer writes per-epoch train/validation Micro-F1 and Macro-F1 logs.
#
# Submit through cluster/submit_raw_native_multiseed_learning_curves.sh.
#SBATCH --job-name=raw-native-lc
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-24%5
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

requeue_before_timeout() {
    echo "Approaching the 24-hour limit; requeueing ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_before_timeout USR1

seeds=(41 42 43 44 45)
modalities=(cnmr hnmr ir msms_pos msms_neg)
task_id="${SLURM_ARRAY_TASK_ID}"
task_count=$(( ${#seeds[@]} * ${#modalities[@]} ))

if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( task_id >= task_count )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0-$((task_count - 1))." >&2
    exit 2
fi

seed_index=$((task_id / ${#modalities[@]}))
modality_index=$((task_id % ${#modalities[@]}))
seed="${seeds[seed_index]}"
modality="${modalities[modality_index]}"

case "$modality" in
    cnmr)
        input_dim=10000
        column="c_nmr_spectra"
        baseline_output_dir="cnmr"
        ttn_script="src/spectroscopy_qml/tree_tensor_network/cnmr/train.py"
        ;;
    hnmr)
        input_dim=10000
        column="h_nmr_spectra"
        baseline_output_dir="hnmr"
        ttn_script="src/spectroscopy_qml/tree_tensor_network/hnmr/train.py"
        ;;
    ir)
        input_dim=1800
        column="ir_spectra"
        baseline_output_dir="ir"
        ttn_script="src/spectroscopy_qml/tree_tensor_network/ir/train.py"
        ;;
    msms_pos)
        input_dim=10000
        column="pos_msms"
        baseline_output_dir="pos_msms"
        ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_pos/train.py"
        ;;
    msms_neg)
        input_dim=10000
        column="neg_msms"
        baseline_output_dir="neg_msms"
        ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_neg/train.py"
        ;;
    *)
        echo "Unsupported modality: $modality" >&2
        exit 2
        ;;
esac

python=".venv/bin/python"
storage_repo="${RAW_STORAGE_REPO:-$PWD}"
data_dir="${RAW_DATA_DIR:-$storage_repo/data/raw}"
preprocessing="none"
epochs=1000000
patience=15
fixed_threshold=0.5
experiment_dir="${RAW_EXPERIMENT_DIR:-$storage_repo/benchmark/preprocessing_training/raw_native_learning_curves}"
split_dir="${RAW_SPLIT_DIR:-$storage_repo/benchmark/preprocessing_training/shared_splits_600}"
split_path="${split_dir}/${modality}_seed_${seed}.npz"
run_dir="${experiment_dir}/${modality}/seed_${seed}"

mkdir -p logs "$run_dir/cnn" "$run_dir/mps" "$run_dir/ttn"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "$split_path" ]]; then
    mkdir -p "$split_dir"
    # Recreate the deterministic split exactly as in the 600-point benchmark.
    # The split contains molecule indices and labels; it does not depend on the
    # later native input width used by the model.
    "$python" scripts/create_multilabel_split.py \
        --data-dir "$data_dir" \
        --modality "$modality" \
        --input-dim 600 \
        --seed "$seed" \
        --output "$split_path"
fi

echo "Raw-native learning curves: seed=$seed modality=$modality input_dim=$input_dim preprocessing=$preprocessing threshold=$fixed_threshold patience=$patience"

valid_result() {
    local metrics_path="$1"
    local model="$2"
    local curve_path="$3"

    [[ -s "$metrics_path" && -s "$curve_path" ]] || return 1
    "$python" scripts/validate_benchmark_result.py \
        --metrics "$metrics_path" \
        --model "$model" \
        --preprocessing "$preprocessing" \
        --input-dim "$input_dim" \
        --seed "$seed" \
        --split-path "$split_path" \
        --max-epochs "$epochs" \
        --early-stopping-patience "$patience"
}

cnn_metrics="$run_dir/cnn/$baseline_output_dir/original/metrics.json"
cnn_curve="$run_dir/cnn/$baseline_output_dir/original/training_logs.csv"
if valid_result "$cnn_metrics" cnn "$cnn_curve"; then
    echo "Skipping completed CNN run: $cnn_metrics"
else
    "$python" benchmark/cnn/scripts/run_cnn_jung_baseline.py \
        --analytical_data "$data_dir" \
        --base_out_path "$run_dir/cnn" \
        --columns "$column" \
        --seed "$seed" \
        --input_dim "$input_dim" \
        --split_path "$split_path" \
        --preprocessing "$preprocessing" \
        --epochs "$epochs" \
        --early-stopping-patience "$patience" \
        --no_kfold
fi

mps_metrics="$run_dir/mps/results/metrics.json"
mps_curve="$run_dir/mps/results/training_log.csv"
if valid_result "$mps_metrics" mps "$mps_curve"; then
    echo "Skipping completed MPS run: $mps_metrics"
else
    "$python" scripts/run_mps_benchmark.py \
        --data-dir "$data_dir" \
        --output-dir "$run_dir/mps" \
        --modality "$modality" \
        --input-dim "$input_dim" \
        --split-path "$split_path" \
        --seed "$seed" \
        --epochs "$epochs" \
        --early-stopping-patience "$patience" \
        --device cuda \
        --fixed-threshold "$fixed_threshold" \
        --preprocessing "$preprocessing"
fi

ttn_metrics="$run_dir/ttn/metrics.json"
ttn_curve="$run_dir/ttn/training_log.csv"
if valid_result "$ttn_metrics" ttn "$ttn_curve"; then
    echo "Skipping completed TTN run: $ttn_metrics"
else
    ttn_shape_args=()
    if (( input_dim == 10000 )); then
        # Keeps the first tree level even and bounds the number of leaf nodes.
        ttn_shape_args=(--segment-window-size 48 --segment-stride 44)
    fi
    "$python" scripts/run_ttn_fixed_threshold.py \
        --modality "$modality" \
        --fixed-threshold "$fixed_threshold" \
        --data-dir "$data_dir" \
        --output-dir "$run_dir/ttn" \
        --split-path "$split_path" \
        --input-dim "$input_dim" \
        --seed "$seed" \
        --device cuda \
        --epochs "$epochs" \
        --early-stopping-patience "$patience" \
        --no-amp \
        --preprocessing "$preprocessing" \
        "${ttn_shape_args[@]}"
fi

echo "Completed raw-native task: seed=$seed modality=$modality input_dim=$input_dim"
