"""
QML model for IR spectroscopy functional group classification.

Hybrid classical-quantum approach using PennyLane + PyTorch.
"""

from .model_qml import QMLModel
from .model_quanv1d import Quanvolution1D, QuanvolutionalIRClassifier

__all__ = ["QMLModel", "Quanvolution1D", "QuanvolutionalIRClassifier"]
