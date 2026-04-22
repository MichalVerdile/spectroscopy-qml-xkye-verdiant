"""
Experiment 10.2 TTN classifier for C-NMR spectroscopy.

Adapted from IR experiment 10.2 with input dimension of 10000 for C-NMR spectra.
"""

from .model import (
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    LorentzianFeatureMap,
    TTNCNMRClassifier10_2,
)

__all__ = [
    "LorentzianFeatureMap",
    "TTNCNMRClassifier10_2",
    "DEFAULT_SEGMENT_WINDOW_SIZE",
    "DEFAULT_SEGMENT_STRIDE",
]
