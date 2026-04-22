"""Experiment 10.6: Optimized TTN with extended training and weighted class balancing.

Improvements over Experiment 10.2:
- 300 epochs (up from 200) for better convergence
- 40 early stopping patience (up from 20) to prevent premature stopping
- Weighted class balancing: pos_weight_power=0.6 (up from 0.5)
- Enhanced per-label threshold optimization
- Targeted improvements for rare functional groups:
  * Thial (100% error in CNN baseline)
  * Azo compound (98.3% error)
  * Hydrazone (79% error)

Expected Results:
- TTN 10.2 baseline: F1 86.2%
- TTN 10.6 target: F1 90-91% with full optimization pipeline
"""
