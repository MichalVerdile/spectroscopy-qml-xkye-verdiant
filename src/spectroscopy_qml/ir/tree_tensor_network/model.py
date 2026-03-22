"""Compatibility wrapper for experiment 3 TTN model."""

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment3.model import (
    QuantumFeatureMap,
    SegmentCompressor,
    TensorMerge,
    TTNIRClassifier,
)

__all__ = [
    "QuantumFeatureMap",
    "SegmentCompressor",
    "TensorMerge",
    "TTNIRClassifier",
]
