from __future__ import annotations

import torch
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


@pytest.mark.parametrize(
    ("segment_window_size", "segment_stride", "expected_num_segments", "expected_merge_levels"),
    [
        (48, 43, 42, 6),
        (64, 58, 31, 5),
        (80, 72, 25, 5),
    ],
)
def test_experiment6_accepts_valid_overlap10_layouts_with_expected_tree_shape(
    segment_window_size: int,
    segment_stride: int,
    expected_num_segments: int,
    expected_merge_levels: int,
) -> None:
    model = TTNIRClassifier6(
        num_labels=5,
        chi=8,
        input_dim=1800,
        segment_window_size=segment_window_size,
        segment_stride=segment_stride,
    )

    overlap_ratios = compute_adjacent_overlap_ratios(model.segment_slices)

    assert model.max_segment_length == segment_window_size
    assert model.leaf_encoder.max_segment_length == segment_window_size
    assert model.num_segments == expected_num_segments
    assert len(model.merge_levels) == expected_merge_levels
    assert abs(sum(overlap_ratios) / len(overlap_ratios) - TARGET_SEGMENT_OVERLAP_RATIO) < 0.02


@pytest.mark.parametrize(
    ("segment_window_size", "segment_stride"),
    [
        (48, 43),
        (64, 58),
        (80, 72),
    ],
)
def test_experiment6_forward_pass_handles_alternative_segment_layouts(
    segment_window_size: int,
    segment_stride: int,
) -> None:
    model = TTNIRClassifier6(
        num_labels=5,
        chi=8,
        input_dim=1800,
        segment_window_size=segment_window_size,
        segment_stride=segment_stride,
    )
    spectra = torch.randn(3, 1800)

    logits = model(spectra)

    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()
