import numpy as np
import torch

from spectroscopy_qml.ir.mps_qcnn_hybrid.config import SpecialistSelectionConfig
from spectroscopy_qml.ir.mps_qcnn_hybrid.frozen_mps_model import MPSFunctionalGroupClassifier
from spectroscopy_qml.ir.mps_qcnn_hybrid.train import (
    _merge_probabilities,
    _select_specialist_labels,
)


def test_mps_latent_roundtrip_matches_mps_classifier_logits() -> None:
    model = MPSFunctionalGroupClassifier(
        input_dim=12,
        num_sites=3,
        physical_dim=4,
        bond_dim=3,
        num_classes=5,
        classifier_head="mps",
        num_sites_2=6,
        physical_dim_2=4,
        bond_dim_2=3,
    )
    model.eval()

    inputs = torch.randn(4, 12)
    with torch.no_grad():
        direct_logits = model(inputs)
        latent = model.encode_latent(inputs)
        latent_logits = model.classify_from_latent(latent)

    assert latent.shape == (4, 256)
    assert torch.allclose(direct_logits, latent_logits, atol=1e-6)


def test_specialist_label_selection_captures_rare_and_hard_classes() -> None:
    y_train = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    y_val = np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )
    baseline_val_prob = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.9, 0.1, 0.2],
            [0.8, 0.2, 0.9],
            [0.9, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    selected, report = _select_specialist_labels(
        y_train,
        y_val,
        baseline_val_prob,
        0.5,
        SpecialistSelectionConfig(
            max_labels=2,
            min_train_positives=1,
            max_train_prevalence=0.25,
            max_val_f1=0.6,
        ),
    )

    assert set(selected) == {1, 2}
    assert report.loc[report["label_idx"] == 1, "selected"].item()
    assert report.loc[report["label_idx"] == 2, "selected"].item()


def test_merge_probabilities_replace_only_updates_specialist_columns() -> None:
    baseline = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    specialist = np.array(
        [
            [0.9, 0.8],
            [0.7, 0.6],
        ],
        dtype=np.float32,
    )

    merged = _merge_probabilities(
        baseline,
        specialist,
        [0, 2],
        strategy="replace",
        blend_alpha=0.5,
    )

    np.testing.assert_allclose(merged[:, 0], specialist[:, 0])
    np.testing.assert_allclose(merged[:, 2], specialist[:, 1])
    np.testing.assert_allclose(merged[:, 1], baseline[:, 1])


def test_merge_probabilities_blend_interpolates_with_mps() -> None:
    baseline = np.array([[0.2, 0.5, 0.8]], dtype=np.float32)
    specialist = np.array([[0.6, 0.1]], dtype=np.float32)

    blended = _merge_probabilities(
        baseline,
        specialist,
        [0, 2],
        strategy="blend",
        blend_alpha=0.25,
    )

    np.testing.assert_allclose(blended[:, 0], 0.25 * 0.6 + 0.75 * 0.2)
    np.testing.assert_allclose(blended[:, 2], 0.25 * 0.1 + 0.75 * 0.8)
    np.testing.assert_allclose(blended[:, 1], baseline[:, 1])
