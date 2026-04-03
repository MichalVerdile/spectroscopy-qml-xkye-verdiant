from __future__ import annotations

import pytest
import torch

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment7.model import (
    MLPEncoderIRClassifier7,
    SpectrumMLPEncoder,
)


def test_experiment7_forward_shapes() -> None:
    model = MLPEncoderIRClassifier7(
        num_labels=7,
        chi=16,
        input_dim=1800,
        encoder_hidden_dim=128,
        readout_hidden_dim=64,
    )
    x = torch.randn(4, 1800)

    logits = model(x)
    probabilities = model(x, apply_sigmoid=True)

    assert logits.shape == (4, 7)
    assert probabilities.shape == (4, 7)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_experiment7_invalid_input_shape_raises() -> None:
    model = MLPEncoderIRClassifier7(num_labels=3, chi=8, input_dim=1800, encoder_hidden_dim=64)

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(4, 3, 1800))

    with pytest.raises(ValueError, match="Expected input feature dimension 1800"):
        model(torch.randn(4, 1799))


def test_spectrum_mlp_encoder_validates_feature_shape() -> None:
    encoder = SpectrumMLPEncoder(input_dim=1800, output_dim=16, hidden_dim=64)

    with pytest.raises(ValueError, match="input_dim=1800"):
        encoder(torch.randn(2, 1799, 3))

    with pytest.raises(ValueError, match="feature_dim=3"):
        encoder(torch.randn(2, 1800, 2))
