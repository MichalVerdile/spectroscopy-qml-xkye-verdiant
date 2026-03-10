"""
MPS Encoder for Spectroscopy Functional Group Classification.

This package implements a pure tensor network (Matrix Product State) approach
for multi-label classification of functional groups from spectroscopy data
(IR, NMR, MS/MS).
"""

from .config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
    DataConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
)
from .model import LocalFeatureMap, MPSEncoder, MPSFunctionalGroupClassifier

__version__ = "1.0.0"

__all__ = [
    "LocalFeatureMap",
    "MPSEncoder",
    "MPSFunctionalGroupClassifier",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "PathConfig",
    "MODEL_CONFIG",
    "DATA_CONFIG",
    "TRAINING_CONFIG",
    "PATH_CONFIG",
]
