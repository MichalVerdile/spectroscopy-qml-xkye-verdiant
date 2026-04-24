#!/usr/bin/env bash
set -euo pipefail

BASE="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_5"
SPLIT="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/data_split_seed42_all.npz"
CACHE="data/cache/ir_spectra_len1800_snv_all.npz"
COMMON="--spectra-cache $CACHE --split-path $SPLIT --device mps --batch-size 512 --epochs 150 --seed 42 --n-filters 8 --hidden-dim 64"

run_ensemble() {
    local dir=$1
    python -u "$BASE/ensemble_predict.py" \
        --specialist-dir "$dir" \
        --spectra-cache "$CACHE" \
        --split-path "$SPLIT" \
        --batch-size 1024 --seed 42
}

echo "========================================================"
echo "  OPTION A: Aggressive oversampling + focal gamma"
echo "========================================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 python -u "$BASE/train.py" $COMMON \
    --output-dir "$BASE/results/option_a" \
    --oversample-epoch-multiplier 3.0 \
    --focal-gamma 4.0 \
    --dropout 0.25 \
    --weight-decay 1e-4
run_ensemble "$BASE/results/option_a"

echo "========================================================"
echo "  OPTION B: Balanced per-class sampling"
echo "========================================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 python -u "$BASE/train.py" $COMMON \
    --output-dir "$BASE/results/option_b" \
    --balanced-sampling \
    --dropout 0.2 \
    --weight-decay 1e-5
run_ensemble "$BASE/results/option_b"

echo "========================================================"
echo "  OPTION C: Trainable Conv1d (ablation, no quanv)"
echo "========================================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 python -u "$BASE/train.py" $COMMON \
    --output-dir "$BASE/results/option_c" \
    --use-trainable-conv \
    --dropout 0.2 \
    --weight-decay 1e-5
run_ensemble "$BASE/results/option_c"

echo "========================================================"
echo "  ALL DONE — Zusammenfassung"
echo "========================================================"
for opt in a b c; do
    f="$BASE/results/option_$opt/ensemble_results.json"
    if [ -f "$f" ]; then
        echo "Option $opt:"
        python3 -c "
import json
d = json.load(open('$f'))
print(f'  TTN alone:  macro={d[\"ttn_macro\"]:.4f}  micro={d[\"ttn_micro\"]:.4f}')
print(f'  Specialist: macro={d[\"specialist_macro\"]:.4f}  micro={d[\"specialist_micro\"]:.4f}')
print(f'  Ensemble:   macro={d[\"ensemble_macro\"]:.4f}  micro={d[\"ensemble_micro\"]:.4f}')
"
    fi
done
