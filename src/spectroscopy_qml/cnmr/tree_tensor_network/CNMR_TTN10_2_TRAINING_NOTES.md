# C-NMR TTN 10.2 Training Run (1-File Baseline)

**Run:** 1 Datei / 245, CPU, BCE Loss, SNV

## Run Command
```bash
python -u src/spectroscopy_qml/cnmr/tree_tensor_network/experiment/experiment10_2/train.py \
  --data-dir data/raw \
  --input-dim 10000 \
  --max-files 1 \
  --batch-size 32 \
  --device cpu \
  --no-amp \
  --no-compile \
  --overwrite-cache 2>&1 | tee -a cnmr_ttn10_2_snv.log
```

## Architecture
- Input dim: 10000
- Num labels: 37
- Chi: 64
- Leaf encoder: none
- Feature channels: raw + first_derivative + second_derivative
- Lorentz gamma: 3.0, kernel half-width: 15
- Segment window: 48, stride: 43, mode: overlap
- Merge mode: relaxed, residual weight: 0.1, renormalize: True
- Loss: BCE, pos_weight_power: 0.5
- Preprocessing: SNV
- Batch size: 32, Epochs: 200

## Data Stats (1 file)
- Total samples: 3235
- Train: 2588 / Val: 323 / Test: 324
- Class weights: min=0.21, max=50.86, mean=10.10
- Cache: data/cache/cnmr_spectra_len10000_snv_files1.npz

## Results
- Best epoch: 46
- Best score (blended_f1): 0.4700
- Best val loss: 1.3544
- Test loss: 1.4065
- **Test f1_micro: 0.7111**
- **Test f1_macro: 0.2419**
- Test precision_micro: 0.7025
- Test recall_micro: 0.7199
- Elapsed: 3447s (~57 min)
- Epochs completed: 66/200 (EarlyStopping triggered)

## Epoch Results (val)
| Epoch | train_loss | val_loss | f1_micro | f1_macro | score |
|-------|-----------|---------|---------|---------|-------|
| 1  | 0.4931 | 0.5705 | 0.3105 | 0.2142 | 0.2623 |
| 5  | 0.4037 | 0.5452 | 0.3892 | 0.2394 | 0.3143 |
| 10 | 0.3288 | 0.5281 | 0.4813 | 0.2843 | 0.3828 |
| 11 | 0.3111 | 0.5190 | 0.5296 | 0.2961 | 0.4128 |
| 19 | 0.1630 | 0.7083 | 0.6076 | 0.2691 | 0.4384 |
| 27 | 0.0486 | 0.9267 | 0.6427 | 0.2605 | 0.4516 |
| 35 | 0.0136 | 1.1430 | 0.6691 | 0.2515 | 0.4603 |
| 42 | 0.0077 | 1.2847 | 0.6791 | 0.2479 | 0.4635 |
| 46 | 0.0050 | 1.3544 | 0.6916 | 0.2485 | **0.4700** ← best |
| 66 | 0.0049 | 1.3747 | 0.6727 | 0.2450 | 0.4589 |

## Observations
- Strong overfitting: train_loss → 0.005, val_loss → 1.37 by epoch 66
- f1_micro peaks ~0.71, f1_macro stagnates ~0.24 (only 1 file = few samples)
- EarlyStopping patience 20 triggered at epoch 66 (best was epoch 46)
- f1_macro low because very few positive samples for rare classes with only 1 file
- LR scheduler reduced lr: 3e-4 → 2.7e-4 → 2.43e-4 → 2.19e-4

## Next Steps
- Full run with all 245 files (laufend auf Cluster, tmux session `cnmr_full`)
- Expected: f1_macro >> 0.24 (analog zu IR: 0.24 → 0.605 mit Full Dataset)
- Consider focal loss + hard-class-boost analog zum IR Hard-Boost Run
