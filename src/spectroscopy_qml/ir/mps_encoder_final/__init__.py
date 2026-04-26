"""
MPS Encoder for IR Spectroscopy Functional Group Classification.

This package implements a pure tensor network (Matrix Product State) approach
for multi-label classification of functional groups from IR spectra.
"""

from .config import (
    DATA_CONFIG,
    GRID_SEARCH_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
    DataConfig,
    GridSearchConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
)
from .model import (
    LocalFeatureMap,
    MPSEncoder,
    MPSFunctionalGroupClassifier,
)

__version__ = "1.0.0"

__all__ = [
    "LocalFeatureMap",
    "MPSEncoder",
    "MPSFunctionalGroupClassifier",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "PathConfig",
    "GridSearchConfig",
    "MODEL_CONFIG",
    "DATA_CONFIG",
    "TRAINING_CONFIG",
    "PATH_CONFIG",
    "GRID_SEARCH_CONFIG",
]
