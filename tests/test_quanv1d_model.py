from __future__ import annotations

import torch

from spectroscopy_qml.ir.qml.model_quanv1d import (
    Quanvolution1D,
    QuanvolutionalIRClassifier,
    build_quanvolution_lookup,
)


def test_quanvolution_lookup_is_deterministic() -> None:
    lookup_a, _ = build_quanvolution_lookup(patch_size=4, n_filters=3, seed=123)
    lookup_b, _ = build_quanvolution_lookup(patch_size=4, n_filters=3, seed=123)

    assert lookup_a.shape == (3, 16)
    assert (lookup_a == lookup_b).all()


def test_quanvolution_layer_forward_shape() -> None:
    layer = Quanvolution1D(input_dim=32, patch_size=4, stride=4, n_filters=5, seed=7)
    x = torch.randn(2, 32)

    out = layer(x)

    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out).all()


def test_quanvolutional_ir_classifier_forward_shape() -> None:
    model = QuanvolutionalIRClassifier(
        input_dim=32,
        num_classes=37,
        patch_size=4,
        stride=4,
        n_filters=3,
        conv1_channels=6,
        conv2_channels=8,
        fc_hidden_dim=16,
        fc_pool_bins=2,
        seed=11,
    )
    model.eval()
    x = torch.randn(2, 32)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 37)
    assert torch.isfinite(logits).all()
