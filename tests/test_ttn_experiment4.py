from __future__ import annotations

import torch

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment4.model import (
    QuantumFeatureMap,
    TTNIRClassifier,
    TensorMerge,
)


def test_experiment4_forward_shapes() -> None:
    model = TTNIRClassifier(num_labels=7, chi=16, num_segments=32)
    x = torch.randn(4, 1800)

    logits = model(x)
    probabilities = model(x, apply_sigmoid=True)

    assert logits.shape == (4, 7)
    assert probabilities.shape == (4, 7)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_experiment4_quantum_feature_map_preserves_unit_norm() -> None:
    feature_map = QuantumFeatureMap(scale=0.1, x_max_mode="per_sample")
    x = torch.randn(4, 32, 57)

    embedded = feature_map(x)
    norms = torch.linalg.norm(embedded, dim=-1)

    assert embedded.shape == (4, 32, 57, 2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_experiment4_segments_cover_input() -> None:
    model = TTNIRClassifier(num_labels=5, chi=8, num_segments=32, input_dim=1800)

    assert len(model.segment_lengths) == 32
    assert sum(model.segment_lengths) == 1800
    assert max(model.segment_lengths) == 57
    assert min(model.segment_lengths) == 56


def test_experiment4_merge_preserves_shape() -> None:
    merge = TensorMerge(chi=8)
    left = torch.randn(3, 4, 8)
    right = torch.randn(3, 4, 8)

    merged = merge(left, right)

    assert merged.shape == (3, 4, 8)
