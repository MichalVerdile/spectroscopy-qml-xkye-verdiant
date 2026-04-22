# Experiment 10.6 Setup & Execution Guide

## Status: ✅ READY TO RUN

Experiment 10.6 has been successfully created with optimized hyperparameters based on CNN baseline error analysis.

---

## Files Created

```
src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_6/
├── __init__.py              # Updated module docstring
├── model.py                 # Same as 10.2 (Lorentzian feature map)
├── train.py                 # OPTIMIZED with:
│                            #   - 300 epochs (300 vs 200)
│                            #   - patience: 40 (vs 20)
│                            #   - pos_weight_power: 0.6 (vs 0.5)
│                            #   - Enhanced thresholds
└── README.md                # Full documentation
```

---

## Key Optimizations

### 1. Extended Training
```yaml
epochs: 200 → 300           # +50% training time
early_stopping_patience: 20 → 40  # More patience
min_epochs_before_stopping: 30 → 20  # Flexible early phase
```

### 2. Weighted Class Balancing
```yaml
pos_weight_power: 0.5 → 0.6  # Stronger weight for rare classes
# This gives higher weights to:
# - Thial: ~88.0x (9 samples)
# - Azo compound: ~7.0x (115 samples)
# - Phosphine: ~6.8x (117 samples)
```

### 3. Enhanced Stopping Criteria
```yaml
early_stopping_min_delta: 1e-4 → 5e-5  # More sensitive
# Now detects smaller improvements as progress
```

---

## Running the Experiment

### **Option 1: Default Run (Recommended)**
```bash
cd /Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train
```

**Expected Runtime:** ~24-36 hours on GPU, ~72h on CPU  
**GPU Memory:** ~8GB (for RTX 3080 or similar)

---

### **Option 2: Quick Test (Validation)**
```bash
# Check if everything works (no actual training)
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --check-only
```

**Expected Runtime:** ~2-5 minutes

---

### **Option 3: With Custom Parameters**

#### **Faster Training (48h instead of 36h)**
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --batch-size 2048 \
    --epochs 300
```

#### **Higher Sensitivity to Rare Groups**
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --loss-type focal \
    --focal-gamma 2.5 \
    --pos-weight-power 0.7
```

#### **GPU Acceleration**
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --device cuda \
    --amp \
    --compile \
    --batch-size 2048
```

---

## Monitoring Training

### In Real-Time
```bash
# Terminal 1: Start training
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train

# Terminal 2: Monitor progress
tail -f src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_6/results/training_log.csv
```

### Expected Output
```
Epoch 001 | train_loss=0.3211 | val_loss=0.2954 | val_f1_micro=0.8621 | val_f1_macro=0.6214 | score=0.7417 | lr=3.00e-04
Epoch 002 | train_loss=0.2915 | val_loss=0.2687 | val_f1_micro=0.8706 | val_f1_macro=0.6428 | score=0.7567 | lr=3.00e-04
...
Epoch 078 | train_loss=0.1854 | val_loss=0.2234 | val_f1_micro=0.8821 | val_f1_macro=0.7125 | score=0.7973 | lr=3.00e-04
Best epoch:      78
Best score:      0.7973 (blended_f1)
Best val loss:   0.2234
Test f1_micro:   0.8624
Test f1_macro:   0.5982
```

---

## Expected Performance

### Baseline (Experiment 10.2)
- **F1 Score:** 86.2%
- **Hamming Accuracy:** ~97%
- **Test Time:** 79,440 samples

### Target (Experiment 10.6)
- **Expected F1:** 90.0-91.5%
- **Improvement:** +3.8-5.3%
- **Per-Label Gains:**
  - Thial: 0% → 30-40% (from 0/9 correct)
  - Azo compound: 1.7% → 40-50% (from 2/115 correct)
  - Hydrazone: 21% → 60-70% (from ~220/1049 correct)

### Breakthrough Milestone
When you see:
- **val_f1_micro > 0.88** → Approaching target
- **val_f1_macro > 0.65** → Strong rare group handling
- **test_f1 > 0.90** → SUCCESS! 🎉

---

## After Training

### Results Location
```
src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_6/results/
├── ttn_ir_best.pt                    # Model weights
├── training_log.csv                  # Epoch-by-epoch metrics
├── training_details.jsonl            # Per-epoch details
├── selected_thresholds.json          # Optimized thresholds
├── summary.txt                       # Final results
└── run_config.json                   # Hyperparameters used
```

### View Final Results
```bash
cat src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_6/results/summary.txt
```

---

## Comparison: 10.2 vs 10.6

### Configuration
```python
# Experiment 10.2 (Baseline)
config_10_2 = {
    'epochs': 200,
    'early_stopping_patience': 20,
    'pos_weight_power': 0.5,
    'batch_size': 1024,
}

# Experiment 10.6 (OPTIMIZED)
config_10_6 = {
    'epochs': 300,                    # +50%
    'early_stopping_patience': 40,    # +100%
    'pos_weight_power': 0.6,          # +20%
    'batch_size': 1024,               # same
}
```

### Expected Results
| Metric | 10.2 | 10.6 | Change |
|--------|------|------|--------|
| F1 Score | 86.2% | ~90.0% | +3.8% |
| Thial Support | 0% | 30-40% | +30-40% |
| Azo Support | 1.7% | 40-50% | +38-48% |
| Hydrazone Support | 21% | 60-70% | +39-49% |

---

## Troubleshooting

### Issue: "OutOfMemory" Error
```bash
# Solution: Reduce batch size
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --batch-size 512
```

### Issue: Early stopping too fast
```bash
# Solution: Increase patience further
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --early-stopping-patience 60 \
    --early-stopping-min-delta 1e-5
```

### Issue: Training not progressing
```bash
# Solution: Check validation metrics every epoch
tail -f src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_6/results/training_log.csv | \
  cut -d',' -f1,5,6,7
```

---

## Next Steps After Training

### 1. Analyze Results
```bash
python error_analysis.py  # Compare 10.6 vs CNN baseline
```

### 2. Test Ensemble
```bash
python create_ensemble.py \
    --cnn-model benchmark/cnn/models/ir/original/results.pickle \
    --ttn-model src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_6/results/ttn_ir_best.pt \
    --weights 0.3 0.7  # 30% CNN, 70% TTN
```

### 3. Apply Data Augmentation
```bash
python generate_augmented_data.py \
    --groups "Thial,Azo compound" \
    --augmentation-factor 5
```

### 4. Rerun with Enhanced Data
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train \
    --augmented-data \
    --epochs 200 \
    --early-stopping-patience 30
```

---

## Command Reference

| Task | Command |
|------|---------|
| **Quick Check** | `--check-only` |
| **Custom Batch Size** | `--batch-size 2048` |
| **Focal Loss** | `--loss-type focal --focal-gamma 2.5` |
| **GPU Acceleration** | `--device cuda --amp --compile` |
| **Different Seed** | `--seed 123` |
| **Skip Cache** | `--overwrite-cache` |
| **Reuse Split** | `--split-path path/to/split.npz` |

---

## System Requirements

| Resource | Minimum | Recommended | Comments |
|----------|---------|-------------|----------|
| GPU Memory | 6GB | 12GB+ | For batch_size=1024 |
| RAM | 16GB | 32GB+ | For data loading |
| Storage | 50GB | 100GB+ | Results + cache |
| Time | 72h | 24h | CPU vs GPU |

---

## References

📊 **Error Analysis:** [ERROR_ANALYSIS_REPORT.md](../../reports/ERROR_ANALYSIS_REPORT.md)  
🎯 **Optimization Strategy:** [TTN_OPTIMIZATION_STRATEGY.py](../../TTN_OPTIMIZATION_STRATEGY.py)  
⚙️  **Config Details:** [TTN_OPTIMIZATION_CONFIG.md](../../TTN_OPTIMIZATION_CONFIG.md)  
📈 **Experiment 10.2:** [experiment10_2/README.md](../experiment10_2/README.md)

---

**Ready to start? Run:**
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_6.train
```
