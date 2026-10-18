#!/usr/bin/env bash
# Train the TTN at input dimension 600 for every spectroscopy modality and seed.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIM=600
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/raw}"
PREPROCESSING="${PREPROCESSING:-shared_mssd}"
SEEDS_STRING="${SEEDS:-41 42 43 44 45}"
MODALITIES_STRING="${MODALITIES:-cnmr hnmr ir msms_pos msms_neg}"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-auto}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-benchmark/multiseed_600/$PREPROCESSING}"
SPLIT_DIR="${SPLIT_DIR:-benchmark/shared_splits_600}"

read -r -a SEED_VALUES <<< "$SEEDS_STRING"
read -r -a MODALITY_VALUES <<< "$MODALITIES_STRING"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m spectroscopy_qml.benchmarking.preprocessing "$PREPROCESSING" > /dev/null

for modality in "${MODALITY_VALUES[@]}"; do
    case "$modality" in
        cnmr) ttn_script="src/spectroscopy_qml/tree_tensor_network/cnmr/train.py" ;;
        hnmr) ttn_script="src/spectroscopy_qml/tree_tensor_network/hnmr/train.py" ;;
        ir) ttn_script="src/spectroscopy_qml/tree_tensor_network/ir/train.py" ;;
        msms_pos) ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_pos/train.py" ;;
        msms_neg) ttn_script="src/spectroscopy_qml/tree_tensor_network/msms_neg/train.py" ;;
        *) echo "Unsupported modality '$modality'." >&2; exit 2 ;;
    esac

    for seed in "${SEED_VALUES[@]}"; do
        split_path="$SPLIT_DIR/${modality}_seed_${seed}.npz"
        output_dir="$EXPERIMENT_DIR/$modality/seed_$seed/ttn"
        metrics_path="$output_dir/metrics.json"
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
            --metrics "$metrics_path" --model ttn --preprocessing "$PREPROCESSING" \
            --input-dim "$INPUT_DIM" --seed "$seed" --split-path "$split_path"; then
            echo "Skipping completed TTN run: $metrics_path"
            continue
        fi

        echo "=== TTN: modality=$modality seed=$seed input_dim=$INPUT_DIM ==="
        "$PYTHON" "$ttn_script" \
            --data-dir "$DATA_DIR" \
            --output-dir "$output_dir" \
            --split-path "$split_path" \
            --input-dim "$INPUT_DIM" \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            --device "$DEVICE" \
            --preprocessing "$PREPROCESSING"
    done
done
