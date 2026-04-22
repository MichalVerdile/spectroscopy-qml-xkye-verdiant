# MS/MS Positive (SCARF) — TTN Training Notes

## CNN Baseline (PQN Normalization)

**Run:** `pqn_normalization/run_cnn_with_pqn.py`, Keras/TF, 5-Fold CV

| Metric | Value |
|--------|-------|
| CV F1 (mean ± std) | 0.6388 ± 0.0004 |
| Best fold (Fold 2) | 0.6393 |
| Test F1 | **0.6377** |

### Configuration
- Spectrum type: pos_msms (SCARF)
- Input shape: (600, 1)
- Num functional groups: 37
- Normalization: PQN (Probabilistic Quotient Normalization)
- Grouping: Coarse Buckets + Butina (Tanimoto threshold 0.6)
- Train samples: 635522 (80%) / Test: 158881 (20%)
- CV folds: 5, seed: 3245

---

## TTN Experiment (planned)

### Data specs
- Input dim: 600
- Num labels: 37
- Total samples: ~794k
- Normalization: PQN or SNV (to be determined)

### Architecture notes (relative to IR/C-NMR)
- IR: input_dim=1800, segment_window=48, stride=43 → ~42 segments
- C-NMR: input_dim=10000, segment_window=48, stride=43 → ~233 segments
- MS/MS Pos: input_dim=600, smaller → may need smaller segment_window (e.g. 16–24) or different chi

### Target
- Beat CNN baseline: F1 > 0.6377
- Compare with IR TTN: f1_micro=0.862, f1_macro=0.605

### Next Steps
1. Adapt data_loader for MS/MS Pos parquet format
2. Tune segment_window / stride for input_dim=600
3. Run experiment10_2 equivalent on cluster
4. Consider focal loss + hard-class-boost for rare classes
