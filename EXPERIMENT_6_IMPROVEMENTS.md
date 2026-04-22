# Experiment 6 - Optimierungsstrategien für bessere Scores

## Status: Analyse & Roadmap

Experiment 6 ist der **Basis-TTN** mit Raw + 1st + 2nd Derivatives.  
Basierend auf **CNN Baseline Fehleranalyse** → Konkrete Verbesserungen für bessere Scores.

---

## 🎯 Aktuelle Situation

### Experiment 6 (Baseline)
```yaml
Features:          raw + first_derivative + second_derivative
Chi:               64
Leaf encoder:      hidden_dim (trainable)
Readout:           Linear or with hidden_dim
Epochs:            200
Batch size:        1024
Learning rate:     3e-4
Loss:              BCE (uniform weights)
Thresholds:        Per-class optimized
```

### CNN Baseline zur Referenz
```yaml
F1 Score:          97.3%
Test Samples:      158,881
Problem Groups:    Thial (100%), Azo (98%), Hydrazone (79%)
Lesson:            Class Imbalance ist DAS Kernproblem
```

---

## 🚀 PRIORITÄT 1: Class Imbalancing (HIGH IMPACT)

### Problem
Raw Features (Derivatives) ignorieren Klassenungleichgewicht → Seltene Gruppen werden zu niedrig gewichtet.

### Lösung: Weighted Loss + Extended Training

```python
# 1. Gewichte berechnen (pro Label)
class_weights = []
for label_idx in range(37):
    pos_count = (y_train[:, label_idx] == 1).sum()
    neg_count = (y_train[:, label_idx] == 0).sum()
    weight = neg_count / (pos_count + 1e-8)
    class_weights.append(weight)

# 2. In Loss verwenden
pos_weight = torch.tensor(class_weights, device=device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 3. Training länger laufen lassen
epochs: 200 → 300-400  # Mehr Zeit für Konvergenz
early_stopping_patience: 20 → 50  # Nicht zu früh stoppen
```

**Erwarteter Gain:** +2-3% F1

---

## 🔧 PRIORITÄT 2: Feature Engineering

### Problem
Finite Differences (1st/2nd Derivatives) reintroduzieren Rauschen → Lorentzian smoothing besser.

### Lösung 1: Lorentzian-smothed Features (wie Exp. 10.2)

```python
# Übernahme aus Experiment 10.2
# Lorentzian kernel für sanftere Ableitungen
from experiment10_2.model import LorentzianFeatureMap

# In Experiment 6 integrieren:
feature_map = LorentzianFeatureMap(
    gamma=3.0,
    kernel_half_width=15,
    norm_mode='percentile'  # Better than max_abs for rare groups
)
```

**Erwarteter Gain:** +1-2% F1

---

## 📈 PRIORITÄT 3: Extended Training & Hyperparameter Tuning

### A: Längeres Training
```yaml
# VORHER (Baseline)
epochs: 200
early_stopping_patience: 20
min_epochs_before_stopping: 30

# NACHHER (Optimized)
epochs: 400
early_stopping_patience: 60
min_epochs_before_stopping: 20
early_stopping_min_delta: 5e-5  # Sensitive to small improvements
```

### B: Chi (Bond Dimension) erhöhen
```yaml
# VORHER
chi: 64

# NACHHER (für mehr Expressivität)
chi: 96  # oder 128 wenn GPU Memory genug hat
```

**Erwarteter Gain:** +1-2% F1

---

## 🎲 PRIORITÄT 4: Focal Loss für schwierige Samples

```python
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # BCE + Focus on hard negatives
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_term * bce
        
        return loss.mean()
```

**Verwenden mit:** `--loss-type focal --focal-gamma 2.5`  
**Erwarteter Gain:** +0.5-1% F1

---

## 🧠 PRIORITÄT 5: Data Augmentation für seltene Gruppen

### Problem
- Thial: nur 9 Samples
- Azo compound: 115 Samples
- Training sieht diese zu selten

### Lösung: Spectral Augmentation

```python
def augment_spectrum(spectrum, n_points=1800):
    """Augmentiert Spektren für Training"""
    augmentations = []
    
    # Original
    augmentations.append(spectrum)
    
    # 1. Gaussian noise (SNR 50-100 dB)
    snr_db = np.random.uniform(50, 100)
    noise = np.random.normal(0, np.sqrt(np.mean(spectrum**2) / 10**(snr_db/10)), spectrum.shape)
    augmentations.append(spectrum + noise)
    
    # 2. Baseline drift
    drift = np.linspace(0, np.random.uniform(-0.05, 0.05), n_points)
    augmentations.append(spectrum + drift)
    
    # 3. Wavelength shift (±5 cm⁻¹)
    shift = np.random.randint(-5, 6)
    augmentations.append(np.roll(spectrum, shift))
    
    # 4. Smoothing
    from scipy.ndimage import gaussian_filter1d
    augmentations.append(gaussian_filter1d(spectrum, sigma=0.5))
    
    return augmentations

# Für Thial & Azo: 5x augmentation
for rare_group_idx in [Thial_indices, Azo_indices]:
    for idx in rare_group_idx:
        spectrum = X_train[idx]
        augmented = augment_spectrum(spectrum)
        X_train_augmented.extend(augmented)
        y_train_augmented.extend([y_train[idx]] * len(augmented))
```

**Erwarteter Gain:** +0.5-1% F1

---

## ⚙️ Implementierungsroadmap

### Phase 1: SOFORT (2h - heute)
```bash
# Experiment 6 mit erweitertem Training
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --epochs 400 \
    --early-stopping-patience 60 \
    --pos-weight-power 0.6 \
    --read-output TUI
```

✅ Erwartet: F1 ~ baseline + 2%

---

### Phase 2: FEATURE UPGRADE (4h - parallel)
```bash
# Copy experiment 6 → experiment 6b (mit Lorentzian)
cp -r experiment6 experiment6_lorentzian

# Integriere Lorentzian Feature Map
# Verwende code aus experiment10_2/model.py
```

✅ Erwartet: F1 ~ baseline + 3-4%

---

### Phase 3: FOCAL LOSS (2h - Testing)
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --epochs 400 \
    --loss-type focal \
    --focal-gamma 2.5 \
    --pos-weight-power 0.6
```

✅ Erwartet: F1 ~ baseline + 3-5%

---

### Phase 4: DATA AUGMENTATION (6h - Preprocessing)
```python
# Implementiere augment_spectrum()
# Trainiere mit augmented data
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --epochs 300 \
    --with-augmentation \
    --augmentation-factor 5
```

✅ Erwartet: F1 ~ baseline + 4-6%

---

## 📊 Vergleich: Strategien & Impacts

| Strategie | Aufwand | Gain | Priorität | Tools |
|-----------|---------|------|-----------|-------|
| **Extended Training** | 1h | +2% | 🔴 CRITICAL | `--epochs 400` |
| **Weighted Loss** | 1h | +1.5% | 🔴 CRITICAL | `--pos-weight-power 0.6` |
| **Lorentzian Features** | 3h | +1.5% | 🟠 HIGH | Copy from Exp 10.2 |
| **Focal Loss** | 2h | +0.5-1% | 🟠 MEDIUM | `--loss-type focal` |
| **Data Augmentation** | 4h | +1-2% | 🟠 MEDIUM | Custom augmentation |
| **Chi erhöhen** | 0.5h | +0.5% | 🟡 LOW (GPU Memory) | `--chi 96` |

---

## 🎯 KONKRETE EXPERIMENTVARIANTEN

### Experiment 6a - SCHNELLE OPTIMIERUNG
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --output-dir "experiment/experiment6/results_opt1" \
    --epochs 400 \
    --early-stopping-patience 60 \
    --pos-weight-power 0.6 \
    --early-stopping-min-delta 5e-5
```
**Runtime:** 36-48h | **Expected F1:** baseline + 2-3%

---

### Experiment 6b - LORENTZIAN UPGRADE
```bash
# Erst Experiment 6 zu 6b kopieren und Lorentzian integrieren
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6b.train \
    --output-dir "experiment/experiment6/results_opt2" \
    --epochs 400 \
    --pos-weight-power 0.6 \
    --lorentz-gamma 3.0
```
**Runtime:** 48-60h | **Expected F1:** baseline + 3-4%

---

### Experiment 6c - FOCAL LOSS + EXTENDED
```bash
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --output-dir "experiment/experiment6/results_opt3" \
    --epochs 400 \
    --loss-type focal \
    --focal-gamma 2.5 \
    --pos-weight-power 0.6
```
**Runtime:** 48-60h | **Expected F1:** baseline + 4-5%

---

### Experiment 6d - AUGMENTED DATA (最高性能)
```bash
# Generiere augmented data
python generate_augmented_data.py \
    --source-dir "data/raw" \
    --target-dir "data/raw_augmented" \
    --augmentation-factor 5 \
    --rare-groups "Thial,Azo compound,Hydrazone"

# Trainiere mit augmented data
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --output-dir "experiment/experiment6/results_opt4" \
    --data-dir "data/raw_augmented" \
    --epochs 300 \
    --loss-type focal \
    --pos-weight-power 0.6
```
**Runtime:** 48-60h | **Expected F1:** baseline + 5-7%

---

## 📋 CHECKLISTE FÜR IMPLEMENTIERUNG

### ✅ Schritt 1: Weighted Loss (1h)
- [ ] Modify `experiment6/train.py` - pos_weight Berechnung
- [ ] Add `--pos-weight-power` argument (default 0.6)
- [ ] Test mit `--check-only`

### ✅ Schritt 2: Extended Training (0.5h)
- [ ] Change `--epochs` default 200 → 400
- [ ] Change `--early-stopping-patience` 20 → 60
- [ ] Add `--early-stopping-min-delta` 5e-5

### ✅ Schritt 3: Lorentzian Integration (3h)
- [ ] Create `experiment6_lorentzian/model.py`
- [ ] Copy LorentzianFeatureMap from experiment10_2
- [ ] Create `experiment6_lorentzian/train.py` variant
- [ ] Test training

### ✅ Schritt 4: Focal Loss (2h)
- [ ] Add `FocalLoss` class to `losses.py`
- [ ] Modify `build_loss()` for focal option
- [ ] Test with `--loss-type focal`

### ✅ Schritt 5: Data Augmentation (4h)
- [ ] Create `augmentation.py` script
- [ ] Generate augmented dataset
- [ ] Integrate into data loader
- [ ] Test training with augmented data

---

## 🎁 Quick Wins (Sofort umsetzbar)

### SCHNELLSTART - Gain +3-4%
```bash
# Einfach nur Parameter ändern & trainieren
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --epochs 400 \
    --early-stopping-patience 60 \
    --pos-weight-power 0.6 \
    --learning-rate 3e-4 \
    --batch-size 1024 \
    --seed 42
```

Fertig! Erwartete F1: baseline + 2-3%

---

## 📊 Performance Projection

```
Aktuell (Experiment 6):        ? (unbekannt)
                  ↓
+ Extended Training            +2%
↓
+ Weighted Loss                +1.5-3%  (abhängig von Daten)
↓  
+ Lorentzian Features          +1.5%
↓
+ Focal Loss                   +0.5-1%
↓
+ Data Augmentation            +1-2%
↓
ZIEL (Voll optimiert):         +7-9% über Baseline
```

---

## 🤔 FAQs

**Q: Welche Strategie zuerst?**  
A: **Weighted Loss + Extended Training** → einfach, großer Impact (2-3 Stunden)

**Q: GPU Memory Probleme?**  
A: Reduziere batch_size: 1024 → 512 oder 256

**Q: Wie lange dauert alles?**  
A: Jeder Run ~48h GPU, also sequenziell 7-10 Tage für alle

**Q: Kann ich parallel trainieren?**  
A: JA! Auf verschiedenen GPUs oder CPUS gleichzeitig

**Q: Was ist realistisch zu erreichen?**  
A: +3-5% F1 ist konservativ, +6-8% mit vollem Setup möglich

---

## Recommendation

**START HEUTE:**
```bash
# Phase 1: Simple Parameter-Optimierung (2h Setup + 48h Training)
python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \
    --epochs 400 \
    --early-stopping-patience 60 \
    --pos-weight-power 0.6 \
    --output-dir "experiment/experiment6/results_improved"
```

**Dann gleichzeitig:**
- [ ] Implementiere Lorentzian Features (Experiment 6b)
- [ ] Implementiere Focal Loss Option
- [ ] Vorbereitung Data Augmentation

---

**Welche Strategie möchten Sie ZUERST implementieren?**
