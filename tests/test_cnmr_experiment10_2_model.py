from __future__ import annotations

import torch

from spectroscopy_qml.cnmr.tree_tensor_network.experiment.experiment10_2.model import (
    TTNCNMRClassifier10_2,
)


def test_cnmr_ttn_forward_with_odd_segment_count() -> None:
    model = TTNCNMRClassifier10_2(
        num_labels=5,
        chi=8,
        input_dim=240,
        segment_window_size=48,
        segment_stride=48,
    )
    assert model.num_segments == 5

    logits = model(torch.randn(2, 240))

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
