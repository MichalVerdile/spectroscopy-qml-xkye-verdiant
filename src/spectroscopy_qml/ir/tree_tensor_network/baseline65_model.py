"""Compatibility wrapper for experiment 1 baseline65 TTN model."""

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment1.baseline65_model import (
    Baseline65TTNIRClassifier,
    QuantumFeatureMap,
    SegmentCompressor,
    TensorMerge,
)

__all__ = [
    "Baseline65TTNIRClassifier",
    "QuantumFeatureMap",
    "SegmentCompressor",
    "TensorMerge",
]
