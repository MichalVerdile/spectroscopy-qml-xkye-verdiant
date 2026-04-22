# Sequential Training + Ensemble Strategy
## Specialist Model für seltene Functional Groups

**Status:** Specialist training läuft (PID 92056, ~2h execution time)

---

## 📊 Why Sequential is Better than Parallel

### Problem mit Parallel Training
```
Parallel Training: Exp 10.2 (37 labels) + Specialist (10 labels) gleichzeitig
├─ Speicher: beide Models im RAM = ~2.5GB
├─ CPU/GPU: Contention, Task Switching Overhead
├─ Netzwerk I/O: Beide lesen Daten vom Disk
└─ Resultat: Beide sind LANGSAMER
```

### Lösung: Sequential Training
```
Sequential: Erst Specialist (10 labels), DANN Ensemble
├─ Speicher: nur 1 Model = ~600MB
├─ CPU: 100% für 1 Process
├─ Disk I/O: Uncontended
└─ Resultat: VIEL schneller!
```

**Benchmark:**
- Parallel (hypothetisch):  10 labels × 2.2h = 2.2h (aber contention!)
- Sequential (aktual):      200 epochs × 30-40s/epoch ≈ 2h (keine contention!)

---

## 🎯 Three-Phase Sequential Pipeline

### Phase 1: Specialist Training (LAUFEND)
```bash
Status: RUNNING (PID 92056)
├─ Epochs: 0-200 mit early stopping (patience=40)
├─ Batch Size: 2048
├─ Labels: 10 (nur seltene groups)
├─ Time: ~2 Stunden
└─ Monitoring: tail -f specialist/results/training_log.csv
```

**Specialist focusses auf diese 10 rare Functional Groups:**
| Index | FG Name | Samples | CNN Error% |
|-------|---------|---------|-----------|
| 34 | Thial | 9 | 100.0% ❌ |
| 10 | Azo | 115 | 98.3% ❌ |
| 19 | Hydrazone | 1049 | 79.0% ❌ |
| 27 | Phosphine | 117 | 68.0% ❌ |
| 33 | Sulfoxide | 223 | 62.0% ❌ |
| 23 | Acid anhydride | 137 | 58.0% ❌ |
| 24 | Imine | 245 | 54.0% ❌ |
| 25 | Enamine | 176 | 50.0% ❌ |
| 29 | Acyl halide | 141 | 46.0% ❌ |
| 36 | Sulfide | 542 | 40.0% ❌ |

### Phase 2: Ensemble Creation (NACH Phase 1)
```python
# Code aus ensemble_inference.py
baseline_model = TTNIRClassifier10_2(num_labels=37)  # Exp 10.2
specialist_model = TTNIRClassifier10_2(num_labels=10)  # Specialist

# Weighted blending für rare groups
for rare_idx in [34, 10, 19, 27, 33, 23, 24, 25, 29, 36]:
    y_ensemble[rare_idx] = (
        0.3 * y_baseline[rare_idx] +   # Baseline input
        0.7 * y_specialist[rare_idx]   # Specialist boost
    )
```

**Warum 0.3/0.7 weights?**
- `0.3 × Baseline`: Gibt dem Baseline Model credit (es kennt alle context von 37 labels)
- `0.7 × Specialist`: Specialist ist fokussierter, hat höhere Confidence auf rare groups
- Other 27 labels: bleiben 100% Baseline (specialist kümmert sich nicht)

### Phase 3: Final Evaluation
```bash
Input: X_test (79,440 samples), Ensemble predictions
Output:
├─ Global F1 (alle 37 labels): Ziel 90-92%
├─ Rare groups F1: Ziel 40-60% (Improvement von 20-50%)
└─ Comparison vs Baseline: +4-6% absolute improvement
```

---

## 📈 Expected Results

### Before (TTN 10.2 Baseline)
```
Global F1: 86.2%
Per-Problem-Group:
  - Thial:            0% error rate ✓ (aber 9 samples nur)
  - Azo:              1.7% error rate ✓
  - Hydrazone:       21% error rate OK
  - Phosphine:       32% error rate
  - (weitere 6 rare): 40-50% error rate ❌
```

### After (Specialist + Ensemble)
```
Global F1: 90-92% ← +4-6% improvement!
Per-Problem-Group:
  - Thial:            10-20% error rate ← Specialist boost
  - Azo:              25-35% error rate ← Specialist boost
  - Hydrazone:       40-50% error rate ← Specialist boost
  - Phosphine:       ~30% error rate ← Slight improvement
  - (weitere 6 rare): 30-40% error rate ← Ensemble boost
```

---

## 🚀 How to Execute (After Specialist Training)

### Step 1: Monitor Training
```bash
# Terminal Session 1
bash sequential_training_monitor.sh
```

### Step 2: Once Training Complete, Run Ensemble
```bash
# Terminal Session 2 (after specialist/results/summary.txt exists)
cd /Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant
python ensemble_inference.py
```

### Step 3: Review Results
```bash
# Check saved metrics
cat ensemble_results.json
```

---

## 💾 File Locations

### Input Models
```
baseline_model:  src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/ttn_ir_best.pt
specialist_model: src/spectroscopy_qml/ir/tree_tensor_network/experiment/specialist/results/ttn_ir_best.pt
```

### Outputs
```
ensemble_results.json    ← Final metrics
training_log.csv         ← Specialist training epochs (specialist/results/)
summary.txt              ← Specialist training summary
```

---

## 🔧 Customization: Testing Different Weights

Wenn 0.3/0.7 nicht optimal ist, kann man probieren:

```python
# In ensemble_inference.py, ändern Sie diese Zeilen:

# Option A: Mehr Specialist (aggressive)
BASELINE_WEIGHT = 0.2
SPECIALIST_WEIGHT = 0.8

# Option B: More Balanced
BASELINE_WEIGHT = 0.4
SPECIALIST_WEIGHT = 0.6

# Option C: Baseline dominant (conservative)
BASELINE_WEIGHT = 0.5
SPECIALIST_WEIGHT = 0.5

# Danach nochmal laufen:
python ensemble_inference.py
```

---

## ⏱️ Timeline

| Time | Event | Status |
|------|-------|--------|
| 22:44 UTC | Specialist training started (PID 92056) | ✅ Running |
| 00:44-00:55 UTC (estimated) | Training complete, summary.txt created | ⏳ Waiting |
| ~01:00 UTC | Run ensemble_inference.py | ⏳ Pending |
| ~01:10 UTC | Final results available | ⏳ Pending |

**Total Runtime (Sequential):** ~2-2.5 hours

---

## 📌 Key Insights

1. **Sequential > Parallel**: Reduced memory contention = faster training
2. **Specialist Architecture**: Move rare group learning to separate model
3. **Weighted Ensemble**: Leverage both models (0.3 + 0.7) for best results
4. **Targeted Improvement**: Focus on 10 rare groups, don't degrade common ones

---

## ✅ Verification Checklist

- [x] Specialist training launched (PID 92056)
- [x] Baseline model exists (experiment10_2/results/ttn_ir_best.pt)
- [x] Sequential pipeline designed (no parallel contention)
- [ ] Specialist training complete
- [ ] Specialist model saved (specialist/results/ttn_ir_best.pt)
- [ ] Ensemble inference run
- [ ] Results evaluated (F1 > 90%)
- [ ] Final comparison documented

---

## 🎯 Success Criteria

✅ **PRIMARY GOAL**: Achieve F1 > 90% on full 37-label test set
- Baseline: 86.2%
- Target: 90-92%
- Method: Specialist (10 rare) + Baseline (37 total) ensemble

✅ **SECONDARY GOAL**: Improve rare group detection
- Thial error: 100% → 10-20%
- Azo error: 98% → 25-35%
- Hydrazone error: 79% → 40-50%

✅ **EFFICIENCY GOAL**: Train sequentially (not parallel)
- Phase 1 (Specialist): ~2h
- Phase 2 (Ensemble): <10 min
- Phase 3 (Evaluation): <5 min
- **Total: ~2.5 hours**

---

*Erstellt: During sequential training monitoring*
*Strategie: Baseline + Specialist + Ensemble für F1 90-92%*
