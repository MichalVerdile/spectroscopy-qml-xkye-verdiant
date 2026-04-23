from __future__ import annotations

import torch

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_5.model import (
    SPECIALIST_INDICES,
    QuanvolutionalSpecialistHead,
    get_window_indices,
    total_window_size,
)


def test_experiment10_2_5_default_specialist_windows_are_configured() -> None:
    assert len(SPECIALIST_INDICES) == 16
    for class_idx in SPECIALIST_INDICES:
        windows = get_window_indices(class_idx)
        assert windows
        assert total_window_size(class_idx) >= 4


def test_quanvolutional_specialist_head_forward_shape() -> None:
    head = QuanvolutionalSpecialistHead(
        class_idx=24,
        ttn_dim=8,
        patch_size=4,
        stride=2,
        n_filters=3,
        conv_channels=4,
        pool_size=2,
        hidden_dim=8,
        dropout=0.0,
    )

    logits = head(torch.randn(2, 8), torch.randn(2, 1800))

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
