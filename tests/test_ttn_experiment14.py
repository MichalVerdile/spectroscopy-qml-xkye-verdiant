from __future__ import annotations

import argparse

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
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment14.train import (
    make_check_only_split_indices,
    resolve_feature_cache_path,
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


def test_check_only_split_indices_are_disjoint_and_exhaustive() -> None:
    split = make_check_only_split_indices(
        101,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    combined = torch.tensor(
        list(split["train"]) + list(split["val"]) + list(split["test"]),
        dtype=torch.int64,
    )
    assert combined.numel() == 101
    assert combined.unique().numel() == 101
    assert len(split["train"]) > 0
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0


def test_feature_cache_path_is_stable_for_same_feature_config(tmp_path) -> None:
    args = argparse.Namespace(
        spectra_cache=tmp_path / "spectra.npz",
        feature_cache_path=None,
        feature_source="raw",
        include_raw_channel=False,
        sg_window_length=11,
        sg_polyorder=3,
        voigt_gamma_l=3.0,
        voigt_gamma_g=2.0,
        voigt_eta=0.5,
        voigt_kernel_half_width=20,
    )
    args.spectra_cache.touch()

    path_a = resolve_feature_cache_path(args)
    path_b = resolve_feature_cache_path(args)

    assert path_a == path_b
    assert path_a.suffix == ".pt"
