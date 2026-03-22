from __future__ import annotations

import torch

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment4.model4a import (
    IsometricTensorMerge,
    TTNIRClassifier4A,
)


def test_experiment4a_forward_shapes() -> None:
    model = TTNIRClassifier4A(num_labels=7, chi=16, num_segments=32)
    x = torch.randn(4, 1800)

    logits = model(x)
    probabilities = model(x, apply_sigmoid=True)

    assert logits.shape == (4, 7)
    assert probabilities.shape == (4, 7)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_experiment4a_merge_preserves_shape() -> None:
    merge = IsometricTensorMerge(chi=8)
    left = torch.randn(3, 4, 8)
    right = torch.randn(3, 4, 8)

    merged = merge(left, right)

    assert merged.shape == (3, 4, 8)


def test_experiment4a_merge_columns_are_orthonormal() -> None:
    merge = IsometricTensorMerge(chi=8, renormalize_output=False)

    with torch.no_grad():
        isometry = merge._orthonormalize_columns(merge.raw_isometry)
        gram = isometry.transpose(0, 1) @ isometry

    assert torch.allclose(gram, torch.eye(8), atol=1e-5)


def test_experiment4a_segments_cover_input() -> None:
    model = TTNIRClassifier4A(num_labels=5, chi=8, num_segments=32, input_dim=1800)

    assert len(model.segment_lengths) == 32
    assert sum(model.segment_lengths) == 1800
