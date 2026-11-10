#!/usr/bin/env bash
# Complete the four missing seed-41 TTN runs on native 10,000-point spectra.
#
# A segment stride of 44 produces 228 leaf segments for input_dim=10000 and
# window_size=48.  Keeping the first tree level even avoids carrying a
# 144-dimensional raw leaf beside already merged 64-dimensional states.
# No spectrum is interpolated or downsampled by this job.
#
# Submit through cluster/submit_raw_native_ttn_seed41.sh.
#SBATCH --job-name=raw-ttn-s41
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4
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

modalities=(cnmr hnmr msms_pos msms_neg)
task_id="${SLURM_ARRAY_TASK_ID}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( task_id >= ${#modalities[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0-3." >&2
    exit 2
fi

modality="${modalities[task_id]}"
case "$modality" in
    cnmr) ttn_script="src/spectroscopy_qml/tree_tensor_network/cnmr/train.py" ;;
    hnmr) ttn_script="src/spectroscopy_qml/tree_tensor_network/hnmr/train.py" ;;
    msms_pos) ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_pos/train.py" ;;
    msms_neg) ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_neg/train.py" ;;
    *) echo "Unsupported modality: $modality" >&2; exit 2 ;;
esac

seed=41
input_dim=10000
segment_window_size=48
segment_stride=44
data_dir="data/raw"
experiment_dir="benchmark/preprocessing_training/RawDataTraining"
split_path="$experiment_dir/shared_splits/${modality}_seed_${seed}.npz"
output_dir="$experiment_dir/${modality}/seed_${seed}/ttn"

if [[ ! -f "$split_path" ]]; then
    echo "Required shared split is missing: $split_path" >&2
    exit 2
fi

mkdir -p logs "$output_dir"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "TTN raw-data completion: modality=$modality seed=$seed input_dim=$input_dim window=$segment_window_size stride=$segment_stride preprocessing=none patience=15"

exec .venv/bin/python "$ttn_script" \
    --data-dir "$data_dir" \
    --output-dir "$output_dir" \
    --split-path "$split_path" \
    --input-dim "$input_dim" \
    --seed "$seed" \
    --device cuda \
    --epochs 1000000 \
    --early-stopping-patience 15 \
    --segment-window-size "$segment_window_size" \
    --segment-stride "$segment_stride" \
    --no-amp \
    --preprocessing none
