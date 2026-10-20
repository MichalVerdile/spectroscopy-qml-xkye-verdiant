#!/usr/bin/env bash
# Run one preprocessing configuration at a time and index its cache files.
# Five minutes before the partition limit, requeue the same job; completed
# content-addressed caches are loaded on the next attempt.
# Usage: sbatch run_single_preprocessing_chain.sbatch CONFIG [NEXT_CONFIG ...]
#SBATCH --account=init
#SBATCH --partition=gpu_top_ia
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/preprocess-%x-%j.out
#SBATCH --error=logs/preprocess-%x-%j.err

set -euo pipefail

config="${1:?missing preprocessing configuration}"
shift

requeue_before_timeout() {
    echo "Approaching the four-hour limit; requeueing ${SLURM_JOB_ID} for ${config}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_before_timeout USR1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs data/cache/preprocessed

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# The first chained job also indexes the baseline produced by the initial job.
if [[ "${config}" == "snv" ]]; then
    .venv/bin/python scripts/index_preprocessing_cache.py \
        --data-dir data/raw \
        --cache-dir data/cache/preprocessed \
        --configs none \
        --allow-missing
fi

.venv/bin/python scripts/precompute_preprocessing_cache.py \
    --data-dir data/raw \
    --cache-dir data/cache/preprocessed \
    --modalities ir cnmr hnmr msms_pos msms_neg \
    --input-dims 600 1800 \
    --configs "${config}"

.venv/bin/python scripts/index_preprocessing_cache.py \
    --data-dir data/raw \
    --cache-dir data/cache/preprocessed \
    --configs "${config}"

if (( $# > 0 )); then
    next_config="$1"
    chain_script="${SLURM_SUBMIT_DIR}/cluster/run_single_preprocessing_chain.sbatch"
    next_job_id="$(sbatch --parsable --job-name="prep-${next_config}" "${chain_script}" "$@")"
    next_job_id="${next_job_id%%;*}"
    printf '%s\t%s\n' "${next_config}" "${next_job_id}" >> logs/preprocess_single_step_chain.tsv
    echo "Submitted next single-step preprocessing job: ${next_config} (${next_job_id})"
fi
