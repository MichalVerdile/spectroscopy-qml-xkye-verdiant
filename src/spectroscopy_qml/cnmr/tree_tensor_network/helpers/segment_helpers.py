"""Experiment 6 TTN-inspired classifier with derivative-aware spectral features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.spectroscopy_qml.cnmr.tree_tensor_network.helpers.isometric_helpers import (
    RelaxedIsometricMerge,
    SegmentLeafEncoder,
)

DEFAULT_SEGMENT_WINDOW_SIZE = 48
TARGET_SEGMENT_OVERLAP_RATIO = 0.10


def recommended_segment_stride(
    segment_window_size: int,
    target_overlap_ratio: float = TARGET_SEGMENT_OVERLAP_RATIO,
) -> int:
    """Return the closest integer stride for the requested overlap target."""
    return max(1, round(segment_window_size * (1.0 - target_overlap_ratio)))


DEFAULT_SEGMENT_STRIDE = recommended_segment_stride(DEFAULT_SEGMENT_WINDOW_SIZE)


def compute_adjacent_overlap_ratios(segment_slices: list[tuple[int, int]]) -> list[float]:
    """Return the overlap ratio for every adjacent segment pair."""
    overlap_ratios: list[float] = []
    for (left_start, left_end), (right_start, right_end) in zip(segment_slices, segment_slices[1:]):
        overlap = max(0, left_end - right_start)

        shortest_window = min(left_end - left_start, right_end - right_start)
        if shortest_window <= 0:
            raise ValueError("segment_slices must contain positive-length windows.")
        overlap_ratios.append(overlap / shortest_window)
    return overlap_ratios


def validate_target_segment_overlap(
    segment_slices: list[tuple[int, int]],
    target_overlap_ratio: float = TARGET_SEGMENT_OVERLAP_RATIO,
) -> None:
    """Reject segment layouts that miss the experiment-6 overlap target badly."""
    overlap_ratios = compute_adjacent_overlap_ratios(segment_slices)
    if not overlap_ratios:
        return

    reference_window = segment_slices[0][1] - segment_slices[0][0]
    tolerance_ratio = 1.0 / max(1, reference_window)
    mean_overlap_ratio = sum(overlap_ratios) / len(overlap_ratios)

    if abs(mean_overlap_ratio - target_overlap_ratio) > tolerance_ratio + 1e-9:
        raise ValueError(
            "Experiment 6 targets "
            f"{target_overlap_ratio:.1%} overlap between adjacent segment windows, "
            f"but the configured layout averages {mean_overlap_ratio:.1%}. "
            "Adjust segment_stride or segment_window_size."
        )
