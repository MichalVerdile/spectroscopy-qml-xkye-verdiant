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
    assert model.num_readout_scales > 1
    assert model.output_head[0].in_features == model.multi_scale_readout_dim


def test_quantum_feature_map_preserves_unit_norm() -> None:
    feature_map = QuantumFeatureMap(scale=0.1, x_max_mode="per_sample")
    x = torch.randn(4, 32, 57)

    embedded = feature_map(x)
    norms = torch.linalg.norm(embedded, dim=-1)

    assert embedded.shape == (4, 32, 57, 2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_segment_lengths_cover_input() -> None:
    model = TTNIRClassifier(
        num_labels=5,
        chi=8,
        segment_window_size=64,
        segment_stride=32,
        input_dim=1800,
    )

    assert model.num_segments > 32
    assert model.segment_slices[0][0] == 0
    assert model.segment_slices[-1][1] == 1800
    assert max(model.segment_lengths) == 64
    assert min(model.segment_lengths) > 0
    assert sum(model.segment_lengths) > 1800
    assert model.segment_slices[1][0] < model.segment_slices[0][1]


def test_invalid_input_shape_raises() -> None:
    model = TTNIRClassifier(num_labels=3, chi=8)

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(4, 32, 1800))

    with pytest.raises(ValueError, match="Expected input feature dimension 1800"):
        model(torch.randn(4, 1799))


def test_invalid_window_config_raises() -> None:
    with pytest.raises(ValueError, match="segment_window_size"):
        TTNIRClassifier(num_labels=3, chi=8, segment_window_size=0)

    with pytest.raises(ValueError, match="segment_stride"):
        TTNIRClassifier(num_labels=3, chi=8, segment_stride=0)


def test_invalid_global_scaling_config_raises() -> None:
    with pytest.raises(ValueError, match="global_x_max"):
        TTNIRClassifier(num_labels=3, chi=8, x_max_mode="global")


def test_tensor_merge_dimension_check() -> None:
    merge = TensorMerge(chi=8)

    with pytest.raises(ValueError, match="chi=8"):
        merge(torch.randn(4, 7), torch.randn(4, 7))


def test_tensor_merge_uses_soft_normalization_and_residual_path() -> None:
    merge = TensorMerge(chi=8, normalization="layernorm", residual_weight=0.25)
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)

    merged = merge(left, right)
    norms = torch.linalg.norm(merged, dim=-1)

    assert merged.shape == left.shape
    assert torch.isfinite(merged).all()
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_invalid_tensor_merge_config_raises() -> None:
    with pytest.raises(ValueError, match="normalization"):
        TensorMerge(chi=8, normalization="invalid")

    with pytest.raises(ValueError, match="residual_weight"):
        TensorMerge(chi=8, residual_weight=1.5)


def test_tensor_merge_orthogonalization_produces_near_orthonormal_columns() -> None:
    merge = TensorMerge(chi=8)

    orthogonal = merge._orthogonalize(merge.left_factor.detach())
    gram = orthogonal.transpose(0, 1) @ orthogonal

    assert orthogonal.shape == (8, 8)
    assert torch.allclose(gram, torch.eye(8), atol=1e-4, rtol=1e-4)
