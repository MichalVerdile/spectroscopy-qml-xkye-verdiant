"""Compatibility wrapper for experiment 2 legacy TTN model."""

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment2.legacy_model import (
    LegacyTTNIRClassifier,
    QuantumFeatureMap,
    TensorMerge,
)

__all__ = [
    "LegacyTTNIRClassifier",
    "QuantumFeatureMap",
    "TensorMerge",
]
