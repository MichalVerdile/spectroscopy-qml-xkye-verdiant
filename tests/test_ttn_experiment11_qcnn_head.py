from __future__ import annotations

import pytest
import torch

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment11_qcnn_head.model import (
    CNN_FEATURE_DIM,
    QCNNBinaryHead,
    SpecialistHeadEnsemble,
)


@pytest.mark.parametrize("head_type", ["linear", "mlp", "qcnn"])
def test_specialist_head_ensemble_forward_shape(head_type: str) -> None:
    model = SpecialistHeadEnsemble(
        num_specialist_classes=5,
        head_type=head_type,
        input_dim=CNN_FEATURE_DIM,
        qcnn_qubits=4,
        qcnn_projection_hidden_dim=16,
    )
    x = torch.randn(3, CNN_FEATURE_DIM)
    logits = model(x)
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()


def test_specialist_head_sigmoid_output_range() -> None:
    model = SpecialistHeadEnsemble(
        num_specialist_classes=3,
        head_type="qcnn",
        qcnn_qubits=4,
        qcnn_projection_hidden_dim=16,
    )
    probs = model(torch.randn(2, CNN_FEATURE_DIM), apply_sigmoid=True)
    assert probs.shape == (2, 3)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_qcnn_binary_head_invalid_qubit_count_raises() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        QCNNBinaryHead(n_qubits=5)
