"""Specialist: TTN classifier for 10 rare functional groups.

Designed to work with experiment10_2 baseline in ensemble mode.
Trains ONLY on: Thial, Azo compound, Hydrazone, Phosphine, Sulfoxide,
Acid anhydride, Imine, Enamine, Acyl halide, Sulfide.

Expected performance:
- Baseline (10.2) on these groups: ~10-20% F1
- Specialist alone: ~50-70% F1
- Ensemble (0.3 baseline + 0.7 specialist): ~70-80% F1
"""
