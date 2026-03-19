from __future__ import annotations

import pytest
import torch

from spectroscopy_qml.ir.tree_tensor_network import QuantumFeatureMap, TTNIRClassifier, TensorMerge


def test_ttn_forward_shapes() -> None:
    model = TTNIRClassifier(num_labels=7, chi=16)
    x = torch.randn(4, 1800)

    logits = model(x)
    probabilities = model(x, apply_sigmoid=True)

    assert logits.shape == (4, 7)
    assert probabilities.shape == (4, 7)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_quantum_feature_map_preserves_unit_norm() -> None:
    feature_map = QuantumFeatureMap(scale=0.1, x_max_mode="per_sample")
    x = torch.randn(4, 32, 57)

    embedded = feature_map(x)
    norms = torch.linalg.norm(embedded, dim=-1)

    assert embedded.shape == (4, 32, 57, 2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_segment_lengths_cover_input() -> None:
    model = TTNIRClassifier(num_labels=5, chi=8, num_segments=32, input_dim=1800)

    assert len(model.segment_lengths) == 32
    assert sum(model.segment_lengths) == 1800
    assert max(model.segment_lengths) == 57
    assert min(model.segment_lengths) == 56


def test_invalid_input_shape_raises() -> None:
    model = TTNIRClassifier(num_labels=3, chi=8)

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(4, 32, 1800))

    with pytest.raises(ValueError, match="Expected input feature dimension 1800"):
        model(torch.randn(4, 1799))


def test_invalid_segment_count_raises() -> None:
    with pytest.raises(ValueError, match="power of two"):
        TTNIRClassifier(num_labels=3, chi=8, num_segments=30)


def test_invalid_global_scaling_config_raises() -> None:
    with pytest.raises(ValueError, match="global_x_max"):
        TTNIRClassifier(num_labels=3, chi=8, x_max_mode="global")


def test_tensor_merge_dimension_check() -> None:
    merge = TensorMerge(chi=8)

    with pytest.raises(ValueError, match="chi=8"):
        merge(torch.randn(4, 7), torch.randn(4, 7))
