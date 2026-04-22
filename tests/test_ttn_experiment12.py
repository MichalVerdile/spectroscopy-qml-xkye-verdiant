from __future__ import annotations

import pytest
import torch

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment12.model import (
    Experiment12FeatureExtractor,
    HierarchicalTNCompressor,
    MLPClassifier,
    TNCompressionClassifier,
)


@pytest.mark.parametrize("feature_source", ["sg_no_norm", "sg_z_score", "voigt_z_score"])
def test_feature_extractor_shape(feature_source: str) -> None:
    extractor = Experiment12FeatureExtractor(feature_source=feature_source)
    out = extractor(torch.randn(4, 1800))
    assert out.shape == (4, 1800, 3)
    assert torch.isfinite(out).all()


def test_feature_extractor_optional_raw_channel() -> None:
    extractor = Experiment12FeatureExtractor(feature_source="sg_no_norm", include_raw_channel=True)
    x = torch.randn(2, 1800)
    out = extractor(x)
    assert out.shape == (2, 1800, 4)
    assert torch.allclose(out[:, :, 0], x)


def test_invalid_feature_source_raises() -> None:
    with pytest.raises(ValueError, match="feature_source"):
        Experiment12FeatureExtractor(feature_source="unknown")


def test_hierarchical_tn_compressor_shape_and_finite_for_odd_sites() -> None:
    compressor = HierarchicalTNCompressor(
        sequence_length=30,
        feature_dim=3,
        bond_dim=16,
        site_length=8,
    )
    out = compressor(torch.randn(5, 30, 3))
    assert out.shape == (5, 16)
    assert torch.isfinite(out).all()


def test_mlp_classifier_shape() -> None:
    model = MLPClassifier(input_dim=128, num_labels=7, hidden_dims=(32, 16))
    logits = model(torch.randn(3, 128))
    assert logits.shape == (3, 7)
    assert torch.isfinite(logits).all()


def test_tn_compression_classifier_shape() -> None:
    model = TNCompressionClassifier(
        sequence_length=1800,
        feature_dim=3,
        num_labels=11,
        bond_dim=32,
        site_length=15,
        head_hidden_dims=(16,),
    )
    logits = model(torch.randn(2, 1800, 3))
    assert logits.shape == (2, 11)
    assert torch.isfinite(logits).all()
