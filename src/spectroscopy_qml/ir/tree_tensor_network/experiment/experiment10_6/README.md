# Experiment 10.6 - Optimized TTN with Weighted Loss & Extended Training

**Status:** Ready to run  
**Based on:** Experiment 10.2 (F1: 86.2%)  
**Target:** F1 90-91% with full optimization suite

## Overview

Experiment 10.6 is an optimized version of Experiment 10.2 incorporating findings from CNN baseline error analysis. It implements:

1. **Extended Training** - 300 epochs (vs 200)
2. **Enhanced Early Stopping** - patience: 40 (vs 20)
3. **Weighted Class Balancing** - pos_weight_power: 0.6 (vs 0.5)
4. **Per-Label Threshold Optimization** - tuned on validation set

## Key Changes from Experiment 10.2

| Parameter | 10.2 (Baseline) | 10.6 (Optimized) | Rationale |
|-----------|-----------------|------------------|-----------|
| **Epochs** | 200 | 300 | More training time for convergence |
| **Early Stop Patience** | 20 | 40 | Prevent premature stopping with extended training |
| **Min Epochs** | 30 | 20 | Allow flexibility in early phases |
| **Min Delta** | 1e-4 | 5e-5 | More sensitive to small improvements |
| **Pos Weight Power** | 0.5 | 0.6 | Stronger weighting for rare classes |
| **Threshold Mode** | per_class | per_class | Enhanced optimization per functional group |

## Problem Groups Targeted

Based on CNN baseline error analysis:

| Rank | Functional Group | Error Rate | Samples | Weight |
|------|------------------|-----------|---------|--------|
| 1 | Thial | 100% | 9 | ~88.0x |
| 2 | Azo compound | 98.3% | 115 | ~7.0x |
| 3 | Hydrazone | 79.0% | 1,049 | ~0.76x |
| 4 | Phosphine | 68.0% | 117 | ~6.8x |
| 5 | Sulfoxide | 62.0% | 223 | ~3.1x |

## Running the Experiment

### Quick Start (Default Parameters)
```bash
cd /Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train
```

### Custom Configuration
```bash
# Extended training with larger batch size
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --batch-size 2048 \
    --epochs 300 \
    --early-stopping-patience 40 \
    --loss-type focal \
    --focal-gamma 2.5

# Using focal loss for harder sample weighting
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --loss-type focal \
    --focal-gamma 2.0

# GPU with mixed precision
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --device cuda \
    --amp
```

### Full Help
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train --help
```

## Expected Performance

### Conservative Estimate (Primary Improvements Only)
- **Step 1:** Extended Training → +1.5% F1 (87.7%)
- **Step 2:** Weighted Loss → +1.5% F1 (89.2%)
- **Step 3:** Enhanced Thresholds → +0.8% F1 (90.0%)

### Aggressive Estimate (Full Pipeline)
- Add Focal Loss → +0.5% F1 (90.5%)
- Add Feature Engineering → +0.3% F1 (90.8%)
- Add Ensemble with CNN → +3-4% F1 (93-94%)

## Outputs

All results are saved to:
```
experiment/experiment10_6/results/
```

Key files generated:
- `ttn_ir_best.pt` - Best model checkpoint
- `training_log.csv` - Per-epoch metrics
- `selected_thresholds.json` - Optimized per-label thresholds
- `summary.txt` - Final performance summary
- `run_config.json` - Complete hyperparameter configuration

## Monitoring Progress

### During Training
Check the logs in real-time:
```bash
tail -f experiment/experiment10_6/results/training_log.csv
```

### After Training
View performance metrics:
```bash
cat experiment/experiment10_6/results/summary.txt
```

## Architecture Details

### Model
- **Type:** Tree Tensor Network (TTN)
- **Feature Map:** Lorentzian + 1st + 2nd derivatives
- **Readout:** Linear layer (chi → 37 labels)
- **Chi (bond dimension):** 64

### Data Preprocessing
- **Baseline Subtraction:** Lorentzian smoothing
- **Normalization:** Percentile (25-75%)
- **SNV:** Standard Normal Variate applied
- **Target Length:** 1800 points

### Loss Function
- **Default:** Weighted Binary Cross-Entropy
- **Optional:** Focal Loss (configurable via `--loss-type focal`)
- **Weighting:** Per-class positive weights calculated from training data

## Comparison with Baselines

| Model | F1 (Micro) | F1 (Macro) | Notes |
|-------|-----------|-----------|-------|
| CNN Baseline | 97.3% | - | (158,881 test samples) |
| TTN 10.2 | 86.2% | - | Current baseline (79,440 test samples) |
| **TTN 10.6** | **~90.0%** | - | **Target (this experiment)** |
| TTN 10.6 + Ensemble | ~93-95% | - | Post-processing combination |

## Hyperparameter Reference

```yaml
# Architecture
chi: 64
input_dim: 1800
segment_window_size: 256
segment_stride: 64
lorentz_gamma: 3.0
lorentz_kernel_half_width: 15

# Training
batch_size: 1024
epochs: 300                    # OPTIMIZED
learning_rate: 3e-4
weight_decay: 1e-6
optimizer: Adam

# Early Stopping
early_stopping_metric: blended_f1
early_stopping_patience: 40    # OPTIMIZED
early_stopping_min_delta: 5e-5 # OPTIMIZED
min_epochs_before_stopping: 20 # OPTIMIZED

# Learning Rate Scheduler
lr_scheduler_factor: 0.9
lr_scheduler_patience: 5

# Class Weighting
pos_weight_power: 0.6          # OPTIMIZED
loss_type: bce                 # Can be: bce, focal

# Threshold Optimization
threshold_mode: per_class
threshold_target_metric: per_class_f1
threshold_grid_step: 0.02
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python -m ... --batch-size 512

# Or use gradient accumulation
python -m ... --batch-size 256 --grad-accumulation-steps 4
```

### Training too slow
```bash
# Enable torch.compile (PyTorch 2.0+)
python -m ... --compile
```

### Early stopping too aggressive
```bash
# Increase early stopping patience
python -m ... --early-stopping-patience 60
```

## References

- **CNN Baseline Analysis:** [ERROR_ANALYSIS_REPORT.md](../../reports/ERROR_ANALYSIS_REPORT.md)
- **Optimization Strategy:** [TTN_OPTIMIZATION_STRATEGY.py](../../TTN_OPTIMIZATION_STRATEGY.py)
- **Optimization Config:** [TTN_OPTIMIZATION_CONFIG.md](../../TTN_OPTIMIZATION_CONFIG.md)

---

**Created:** 2026-04-20  
**Optimization Basis:** CNN Baseline (F1: 86.2%) → Target (F1: 90.0%)
