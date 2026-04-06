"""Experiment 6: TTN encoder with raw-intensity and derivative feature channels.

Current reference defaults are tuned for the overlap10 layout:
window=64, stride=58, batch_size=1024, learning_rate=5e-4, leaf_dropout=0.1,
readout_dropout=0.0, merge_residual_weight=0.15, threshold_grid_step=0.05.
"""
