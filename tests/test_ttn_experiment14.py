from __future__ import annotations

import torch
from sklearn.decomposition import PCA

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment14.model import (
    Experiment14Classifier,
    Experiment14FeatureExtractor,
    PCACompressor,
    RawSubsampleCompressor,
    SharedClassicalHead,
    SharedQuantumHead,
    TNSequenceCompressor,
)


def test_experiment14_raw_feature_extractor_shape() -> None:
    extractor = Experiment14FeatureExtractor(feature_source="raw")
    out = extractor(torch.randn(3, 32))
    assert out.shape == (3, 32, 1)
    assert torch.isfinite(out).all()


def test_raw_subsample_compressor_shape() -> None:
    compressor = RawSubsampleCompressor(output_dim=8)
    out = compressor(torch.randn(2, 32, 1))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_pca_compressor_shape_from_sklearn() -> None:
    x = torch.randn(10, 16, 1).flatten(start_dim=1).numpy()
    pca = PCA(n_components=4, random_state=42).fit(x)
    compressor = PCACompressor.from_sklearn(pca)
    out = compressor(torch.randn(2, 16, 1))
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_tn_quantum_variant_forward_shape() -> None:
    model = Experiment14Classifier(
        compressor=TNSequenceCompressor(sequence_length=24, feature_dim=1, output_dim=8, site_length=6),
        head=SharedQuantumHead(
            input_dim=8,
            num_labels=5,
            n_qubits=4,
            n_layers=2,
            adapter_hidden_dim=8,
            head_hidden_dim=8,
            dropout=0.0,
        ),
    )
    logits = model(torch.randn(2, 24, 1))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_tn_classical_variant_forward_shape() -> None:
    model = Experiment14Classifier(
        compressor=TNSequenceCompressor(sequence_length=24, feature_dim=1, output_dim=8, site_length=6),
        head=SharedClassicalHead(input_dim=8, num_labels=5, hidden_dims=(8,), dropout=0.0),
    )
    logits = model(torch.randn(2, 24, 1))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
