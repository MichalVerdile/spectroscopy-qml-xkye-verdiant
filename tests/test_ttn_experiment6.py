from __future__ import annotations

import pytest

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.model import (
    DEFAULT_SEGMENT_STRIDE,
    TARGET_SEGMENT_OVERLAP_RATIO,
    TTNIRClassifier6,
    compute_adjacent_overlap_ratios,
    compute_mean_adjacent_overlap_ratio,
)


def test_experiment6_default_stride_tracks_ten_percent_overlap_target() -> None:
    model = TTNIRClassifier6(num_labels=5, chi=8, input_dim=1800)
    overlap_ratios = compute_adjacent_overlap_ratios(model.segment_slices)

    assert model.segment_stride == DEFAULT_SEGMENT_STRIDE
    assert set(overlap_ratios) == {0.09375, 0.109375}
    assert abs(compute_mean_adjacent_overlap_ratio(model.segment_slices) - TARGET_SEGMENT_OVERLAP_RATIO) < 0.01
    assert model.input_position_embedding.embedding_dim == 1
    assert model.input_position_embedding.num_embeddings == model.input_dim
    assert model.output_head[0].in_features == model.chi


def test_experiment6_rejects_configs_far_from_ten_percent_overlap_target() -> None:
    with pytest.raises(ValueError, match="targets 10.0% overlap"):
        TTNIRClassifier6(
            num_labels=5,
            chi=8,
            input_dim=1800,
            segment_window_size=64,
            segment_stride=32,
        )
