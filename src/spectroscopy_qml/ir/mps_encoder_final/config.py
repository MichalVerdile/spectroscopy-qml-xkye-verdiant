"""Compatibility aliases for legacy checkpoint deserialization.

Older IR MPS checkpoints were saved with config objects from
`spectroscopy_qml.ir.mps_encoder_final.config`. The hybrid pipeline only needs
those objects to be importable during `torch.load`, so we re-export the current
IR MPS classifier config types here.
"""

from spectroscopy_qml.ir.mps_classifier.config import (
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

__all__ = [
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
