# Experiment 6 - TTN IR Classifier Architecture (with Position Embeddings)

## Overview
A **Tree Tensor Network (TTN)** for multi-label functional group classification of IR spectra with **spectral derivatives** and **learnable position embeddings** as physically motivated features.

---

## 1. Input Layer
```
IR Spectrum: 1800 wavenumber bins
↓
SNV Normalization (Standard Normal Variate)
↓
[1800] values in range [-1, 1]
```

---

## 2. Learnable Position Embedding 🔗

### Parallel Processing:
```
Input Spectrum [1800]
    ↓
Position Indices: [0, 1, 2, ..., 1799]
    ↓
Position Embedding Lookup (nn.Embedding)
├─ Shape: [1800, 1]  ← One scalar per spectral position
├─ Init: N(0, 1/√1800) ≈ N(0, 0.0236)
└─ Trainable: ✅ Yes
    ↓
Squeeze + Unsqueeze: [1, 1800]
    ↓
ADD to Raw Spectrum
    ↓
Positioned Raw: [batch, 1800]
```

### Physical Intuition:
- Teaches the network: **Spectral position matters**
- 4000 cm⁻¹ (position 0) ≠ 500 cm⁻¹ (position 1799)
- Enables absolute vs. relative feature analysis

---

## 3. SpectralDerivativeFeatureMap 🔄

### Input: 
`positioned_raw` [batch, 1800]

### Generates 3 feature channels per sample:

| Channel | Computation | Meaning |
|---------|-----------|---------|
| **Channel 0** | `positioned_raw` | Raw intensity + absolute spectral position |
| **Channel 1** | `∂spectrum/∂x` (1st derivative) | Peak slope → detects transitions |
| **Channel 2** | `∂²spectrum/∂x²` (2nd derivative) | Curvature → peak sharpness |

### Normalization: 
Each channel normalized **individually** (L∞-norm) per sample

### Output: 
`[batch, 1800, 3]`

---

## 4. Segmentation 📊

The position-aware spectrum is decomposed into **overlapping windows:**

```
Input: [batch, 1800, 3]
    ↓
Spectral Positions: [0...1799]
    ↓
Segment Slices (e.g. window=64, stride=58):
├─ Seg 1: [0...63]
├─ Seg 2: [6...69]
├─ Seg 3: [12...75]
├─ ...
└─ Seg 31: [1736...1799]

Total Segments: ~31 (for 1800-point spectrum)
Overlap: ~10% between adjacent segments
```

### Output: 
`[batch, num_segments, max_segment_length(64), 3_channels]`

---

## 5. SegmentLeafEncoder 🍃

### Per segment:

```
Segment Input: [batch, 64, 3]  (window_size × 3_channels)
    ↓
    ├─ Flatten: [batch, 64 × 3 = 192]
    │
    ├─ LayerNorm([batch, 192])
    │
    ├─ Linear([batch, 192] → [batch, hidden_dim])
    │   └─ hidden_dim = max(4 × chi, 2 × 192)
    │      = max(256, 384) = 384
    │
    ├─ GELU Activation
    │
    ├─ Dropout(0.1)
    │
    ├─ Linear([batch, 384] → [batch, chi])
    │   └─ chi = 64
    │
    ├─ Skip Connection: Linear([batch, 192] → [batch, 64])
    │
    ├─ Add: output = fc2 + skip
    │
    └─ LayerNorm + Normalize([batch, 64])
        ↓
    Output: [batch, chi(64)]
```

### Effect via Position Embeddings:
- Each segment has distinct position-dependent features
- Segment [0...63] (low wavenumber) ≠ Segment [1736...1799] (high wavenumber)
- Position info flows through all layers

### Output: 
`[batch, num_segments, chi]` = 31 "Leaf Nodes"

---

## 6. Tree Tensor Network (TTN) 🌳

### Tree structure with log₂(num_segments) = 5 levels:

```
Level 0 (Leaves): [seg1, seg2, ..., seg31]  
                   ↓ [batch, 31, chi]

Level 1 Merge: Pairwise Merges
    ├─ merge(seg1, seg2) → node₁
    ├─ merge(seg3, seg4) → node₂
    ├─ ...
    ├─ merge(seg29, seg30) → node₁₅
    └─ seg31 (odd) → node₁₆
    ↓ [batch, 16, chi]

Level 2 Merge:
    ├─ merge(node1, node2) → node'₁
    ├─ merge(node3, node4) → node'₂
    ├─ ...
    ↓ [batch, 8, chi]

Level 3 Merge: → [batch, 4, chi]
Level 4 Merge: → [batch, 2, chi]
Level 5 Merge: → [batch, 1, chi]
                    ↓ Root Node
```

### Merge Operation (FastRelaxedIsometricMerge):

```python
Input: left [chi], right [chi]
    ↓
1. Outer Product: left ⊗ right → [chi²]
   Example: [64] ⊗ [64] → [4096]
    
2. QR Decomposition (numerically stable vs. Gram-Schmidt)
   - Factors into unitary matrix Q
   - Reduces to [chi] dimensions
    
3. Residual Connection:
   output = (1 - w) × compressed + w × 0.5×(left + right)
   - weight = 0.15 (default)
   - Balances new vs. old information
    
4. Optional Layer Normalization
    
5. Output: [chi]
```

---

## 7. Readout Head 📤

```
Root Node: [batch, chi(64)]  ← TTN Final Output
    ↓
LayerNorm([batch, 64])
    ↓
Linear([batch, 64] → [batch, 256])
├─ hidden_dim = max(128, 4×chi) = 256
└─ Learns high-dimensional representation
    ↓
GELU Activation
    ↓
Dropout(0.15)  ← Prevents overfitting
    ↓
Linear([batch, 256] → [batch, 34])
├─ Output: 34 functional groups
└─ Raw logits (pre-sigmoid)
    ↓
Sigmoid Activation
    ↓
Probs: [batch, 34] ∈ [0,1]
    ↓
Per-Class Thresholds (grid search, variable)
    ↓
Multi-Label Predictions: [batch, 34] ∈ {0,1}
```

---

## 8. Loss & Optimization

### Binary Cross-Entropy with Pos-Weights:
```
BCE(y, ŷ) = -1/N × Σ [y × log(ŷ) + (1-y) × log(1-ŷ)] × pos_weight

pos_weight = (class_imbalance_ratio)^0.5
```

### Training Loop:
- **Optimizer:** Adam (lr=5e-4, weight_decay=1e-6)
- **LR Scheduler:** ReduceLROnPlateau (factor=0.9, patience=5 epochs)
- **Gradient Clipping:** norm=1.0
- **Early Stopping:** blended F1 = 0.5×F1_macro + 0.5×F1_micro
- **Patience:** 20 epochs, min 30 epochs before stopping

---

## 9. Threshold Tuning 🎯

### After training: Per-Class Thresholds Optimization

```
Val Predictions (raw probs): [batch, 34]
    ↓
Grid Search: [0.01, 0.02, 0.03, ..., 0.99]
    ↓
For each possible threshold:
├─ Apply threshold → binary predictions
├─ Compute F1_macro / per_class_f1
└─ Store best thresholds
    ↓
Final Thresholds: [t₁, t₂, ..., t₃₄]
```

---

## Data Flow Summary

```
IR Spectrum [1800]
    ↓
+ Position Embeddings [1800, 1]  ← New Info!
    ↓ (ADD)
Positioned Spectrum [1800]
    ↓
Feature Map (3 Channels)  
    ↓ [1800, 3]
Segmentation
    ↓ [31, 64, 3]
Leaf Encoders (31 parallel)
    ↓ [31, 64]
TTN Merge (5 Levels, binary tree)
    ↓ [1, 64]
Readout Head (3 layers, 1 norm)
    ↓ [34]
Sigmoid
    ↓ [34] ∈ [0,1]
Per-Class Thresholds
    ↓ [34] ∈ {0,1}
Multi-Label Predictions
```

---

## Tunable Hyperparameters

| Parameter | Default | Sweep Range | Effect |
|-----------|---------|-------------|--------|
| **chi** | 64 | [64, 128, 256] | Model capacity |
| **window_size** | 64 | [48, 64, 80] | Segment size |
| **stride** | 58 | [43, 58, 72] | Overlap amount |
| **leaf_dropout** | 0.1 | [0.05, 0.1, 0.15, 0.2] | Leaf regularization |
| **readout_dropout** | 0.0 | [0.0, 0.1, 0.15] | Output regularization |
| **merge_residual_weight** | 0.15 | [0.1, 0.15, 0.25, 0.3] | Compression vs. info |
| **learning_rate** | 5e-4 | [5e-4, 1e-3, 2e-3] | Training speed |
| **batch_size** | 1024 | [1024, 2048, 4096] | Gradient stability |

---

**Position Embeddings fully integrated!** 🎯
