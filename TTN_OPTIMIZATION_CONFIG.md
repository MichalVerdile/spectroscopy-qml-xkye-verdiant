# Optimierte TTN 10.2 Konfiguration
# Basierend auf CNN Baseline Fehleranalyse

## ============================================================================
## PRIORITÄT 1: CRITICAL CHANGES (Müssen implementiert werden)
## ============================================================================

### 1. WEIGHTED LOSS mit Class Balancing
### Location: experiment/experiment10_2/train.py

```python
# VORHER:
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

# NACHHER:
# Berechne Class Weights für jedes Label
class_weights = []
for label_idx in range(n_labels):
    y_label = y_train[:, label_idx]
    pos_count = (y_label == 1).sum()
    neg_count = (y_label == 0).sum()
    pos_weight = neg_count / (pos_count + 1e-8)  # Avoid division by zero
    class_weights.append(pos_weight)

# Nutze pos_weight in BCEWithLogitsLoss
pos_weight = torch.tensor(class_weights, device=device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
```

**Effekt:**
- Thial (9 Samples):    pos_weight ≈ 88.0
- Azo compound (115):   pos_weight ≈ 7.0
- Phosphine (117):      pos_weight ≈ 6.8
- Hydrazone (1049):     pos_weight ≈ 0.76

---

### 2. EXTENDED TRAINING - Länger trainieren

```yaml
# train.py Hyperparameter

# VORHER:
max_epochs: 200
early_stopping_patience: 15
early_stopping_threshold: 0.001

# NACHHER:
max_epochs: 300              # Erhöht von 200
early_stopping_patience: 40  # Erhöht von 15 (kein zu frühes Stopping)
early_stopping_threshold: 0.0005  # Kleinerer Threshold
monitor_metric: 'val_f1_macro'    # Nicht nur val_loss!
```

**Begründung:**
- TTN 10.2 wurde bei Epoch 78/200 gestoppt
- Mit mehr Geduld könnte F1 noch um 2-3% steigen
- blended_f1 als primary metric verwenden

---

### 3. PER-LABEL THRESHOLD OPTIMIZATION

```python
# VORHER:
predictions = (output > 0.5).float()  # Fixed threshold 0.5 for all labels

# NACHHER:
# Finde optimalen Threshold für jedes Label auf Validation Set
optimal_thresholds = np.zeros(n_labels)

for label_idx in range(n_labels):
    y_val_label = y_val[:, label_idx]
    probs_val_label = output[:, label_idx].cpu().numpy()
    
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.linspace(0.2, 0.8, 50):
        preds = (probs_val_label > thresh).astype(int)
        f1 = f1_score(y_val_label, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    optimal_thresholds[label_idx] = best_thresh
    
    # Debug output für problematische Labels
    if label_idx in [34, 10, 19, 27, 33]:  # Problem labels
        print(f"Label {label_idx}: optimal_threshold = {best_thresh:.3f} (F1={best_f1:.3f})")

# Anwendung bei Inference:
predictions = torch.zeros_like(output)
for label_idx in range(n_labels):
    predictions[:, label_idx] = (output[:, label_idx] > optimal_thresholds[label_idx]).float()
```

**Erwartete Verbesserung:** +2-4% für problematische Gruppen

---

## ============================================================================
## PRIORITÄT 2: MEDIUM CHANGES (Sehr empfohlen)
## ============================================================================

### 4. DATA AUGMENTATION für seltene Gruppen

```python
# In data_loader.py oder Preprocessing

def augment_spectrum(spectrum, n_points=1800):
    """
    Augmentiert IR-Spektren für seltene Functional Groups
    """
    augmentations = []
    
    # Original
    augmentations.append(spectrum)
    
    # 1. Rausch-Addition (SNR 50-100 dB)
    snr_db = np.random.uniform(50, 100)
    noise_power = np.sqrt(np.mean(spectrum**2) / (10**(snr_db/10)))
    augmentations.append(spectrum + np.random.normal(0, noise_power, spectrum.shape))
    
    # 2. Baseline-Drift
    baseline_drift = np.random.uniform(-0.05, 0.05)
    linear_baseline = np.linspace(0, baseline_drift, n_points)
    augmentations.append(spectrum + linear_baseline)
    
    # 3. Wellenzahlen-Verschiebung (±5 cm⁻¹)
    shift = np.random.randint(-5, 6)
    augmentations.append(np.roll(spectrum, shift))
    
    # 4. Spektrale Glättung
    from scipy.ndimage import gaussian_filter1d
    augmentations.append(gaussian_filter1d(spectrum, sigma=0.5))
    
    return augmentations  # Return multiple versions

# Für seltene Gruppen (Thial, Azo) mehrfach augmentieren
rare_group_indices = {
    'Thial': np.where(label_data[:, 34] == 1)[0],      # Label 34
    'Azo': np.where(label_data[:, 10] == 1)[0],        # Label 10
}

for group_name, indices in rare_group_indices.items():
    for idx in indices:
        # 5x augmentation für seltene Gruppen
        spectrum = X_train[idx]
        augmented = augment_spectrum(spectrum)
        X_train_augmented.extend(augmented)
        y_train_augmented.extend([y_train[idx]] * len(augmented))
```

### 5. FOCAL LOSS - Fokus auf schwierige Samples

```python
# Ersetze BCEWithLogitsLoss durch Focal Loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # BCE
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # Focal term
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal Loss
        loss = self.alpha * focal_term * bce
        
        return loss.mean()

# Verwende in Training:
loss_fn = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
```

---

## ============================================================================
## PRIORITÄT 3: NICE-TO-HAVE (Optional)
## ============================================================================

### 6. FEATURE ENGINEERING - Spektralbereiche fokussieren

```python
# Erstelle Region-Maps für problematische Gruppen
# IR Wellenzahlen sind typischerweise 400-4000 cm⁻¹

wavenumber_map = {
    'Thial': (1050, 1150),      # C=S Stretch
    'Azo': (1400, 1600),         # N=N Stretch
    'Hydrazone': (900, 1050),    # N-N Stretch
    'Phosphine': (800, 1200),    # P-C Stretch
    'Sulfoxide': (1030, 1070),   # S=O Stretch
}

# Erstelle Gewichts-Maske
def create_spectral_weight_mask(n_points=1800):
    """Erstelle Gewichte für wichtige Spektralbereiche"""
    weights = np.ones(n_points)
    
    # Normalisiere Wellenzahlen auf Index
    wavenumber_range = np.linspace(400, 4000, n_points)
    
    for group, (start_wn, end_wn) in wavenumber_map.items():
        mask = (wavenumber_range >= start_wn) & (wavenumber_range <= end_wn)
        weights[mask] *= 2.0  # 2x Gewicht in wichtigen Regionen
    
    return weights

# Anwendendung:
weights = create_spectral_weight_mask()
weighted_spectrum = spectrum * weights
```

---

## ============================================================================
## KONFIGURATIONSänderungen ZUSAMMENFASSUNG
## ============================================================================

### Datei: experiment/experiment10_2/train.py

```diff
- loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
+ pos_weight = torch.tensor(class_weights, device=device)
+ loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

- max_epochs = 200
+ max_epochs = 300

- early_stopping_patience = 15
+ early_stopping_patience = 40

- predictions = (output > 0.5).float()
+ predictions = apply_optimal_thresholds(output, optimal_thresholds)

- # No augmentation
+ X_train = augment_rare_groups(X_train, y_train)
```

---

## ============================================================================
## ERWARTETE VERBESSERUNGEN
## ============================================================================

| Schritt | Aktion | Erwarteter F1 Gain | Gesamt F1 |
|---------|--------|-------------------|----------|
| Baseline | TTN 10.2 aktuell | - | 86.2% |
| +1 | Weighted Loss | +1.5% | 87.7% |
| +2 | Extended Training | +1.5% | 89.2% |
| +3 | Threshold Optimization | +0.8% | 90.0% |
| +4 | Data Augmentation | +0.7% | 90.7% |
| +5 | Focal Loss | +0.5% | 91.2% |
| +6 | Feature Engineering | +0.3% | 91.5% |
| ENSEMBLE | CNN + TTN (gewichtet) | +3-4% | 94.0% |

---

## ============================================================================
## KONKRETE NÄCHSTE SCHRITTE
## ============================================================================

### Phase 1: IMMEDIATE (Heute)
1. [ ] Lade experiment10_2/train.py
2. [ ] Implementiere Weighted Loss
3. [ ] Setze max_epochs auf 300
4. [ ] Starte neuen Run (wird ~24h dauern)
   ```bash
   cd experiment10_2
   python train.py --name "exp10_2_weighted_loss_v1"
   ```

### Phase 2: PARALLEL (Während Training läuft)
1. [ ] Implementiere Threshold Optimization Script
2. [ ] Vorbereitung für Data Augmentation
3. [ ] Feature Engineering Analyse

### Phase 3: POST-TRAINING (Nach neuem Run)
1. [ ] Evaluiere Metriken
2. [ ] Wende Threshold Optimization an
3. [ ] Teste Data Augmentation
4. [ ] Starte Ensemble-Modell

---

## Welche Änderung möchten Sie ZUERST implementieren?

Empfohlene Reihenfolge:
1. **Weighted Loss** (einfach, großer Impact) ← START HIER
2. **Extended Training** (automatisch mit Weighted Loss)
3. **Threshold Optimization** (nach Training fertig)
4. **Data Augmentation** (parallel)
5. **Ensemble** (zum Schluss)
