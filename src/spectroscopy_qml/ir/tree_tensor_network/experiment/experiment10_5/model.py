"""Experiment 10.5 TTN classifier with unnormalised Savitzky-Golay channels.

This variant keeps the Savitzky-Golay analytical kernels from experiment 10.4,
but removes the per-sample channel normalisation after smoothing / derivative
computation. The intended use case is SNV-preprocessed input, where additional
post-derivative rescaling may be unnecessary or even distort amplitude
relationships between channels.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.signal import savgol_coeffs
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.model import (
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    validate_target_segment_overlap,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10.model import (
    FastDirectIsometricMerge,
)


class SavitzkyGolayFeatureMapNoNorm(nn.Module):
    """Three-channel SG feature map without per-sample post-filter normalisation."""

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 3,
    ) -> None:
        super().__init__()
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")

        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self._half = window_length // 2

        k0 = savgol_coeffs(window_length, polyorder, deriv=0).astype(np.float32)
        k1 = savgol_coeffs(window_length, polyorder, deriv=1).astype(np.float32)
        k2 = savgol_coeffs(window_length, polyorder, deriv=2).astype(np.float32)

        # conv1d computes cross-correlation; flip odd kernel k1 to get true convolution.
        self.register_buffer("_k0", torch.from_numpy(k0).view(1, 1, -1), persistent=True)
        self.register_buffer("_k1", torch.from_numpy(k1[::-1].copy()).view(1, 1, -1), persistent=True)
        self.register_buffer("_k2", torch.from_numpy(k2).view(1, 1, -1), persistent=True)

    def _apply_kernel(self, x: Tensor, kernel: Tensor) -> Tensor:
        x_padded = F.pad(x.unsqueeze(1), (self._half, self._half), mode="reflect")
        return F.conv1d(x_padded, kernel).squeeze(1)

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_tensor(x):
            raise TypeError("SavitzkyGolayFeatureMapNoNorm expects a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError(
                f"Expected shape (batch_size, spectrum_length), got {tuple(x.shape)}."
            )

        ch0 = self._apply_kernel(x, self._k0)
        ch1 = self._apply_kernel(x, self._k1)
        ch2 = self._apply_kernel(x, self._k2)
        return torch.stack((ch0, ch1, ch2), dim=-1)


class TTNIRClassifier10_5(nn.Module):
    """Experiment 10.5 TTN classifier with unnormalised SG feature channels."""

    def __init__(
        self,
        num_labels: int,
        chi: int,
        input_dim: int = 1800,
        segment_window_size: int = DEFAULT_SEGMENT_WINDOW_SIZE,
        segment_stride: int = DEFAULT_SEGMENT_STRIDE,
        segment_mode: str = "overlap",
        segment_offset: int | None = None,
        segment_state_normalize: bool = True,
        merge_mode: str = "relaxed",
        merge_residual_weight: float = 0.1,
        merge_renormalize_output: bool = True,
        sg_window_length: int = 11,
        sg_polyorder: int = 3,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if segment_window_size <= 0:
            raise ValueError("segment_window_size must be positive.")
        if segment_stride <= 0:
            raise ValueError("segment_stride must be positive.")
        if segment_mode not in {"overlap", "dual_offset"}:
            raise ValueError("segment_mode must be 'overlap' or 'dual_offset'.")

        self.num_labels = int(num_labels)
        self.chi = int(chi)
        self.input_dim = int(input_dim)
        self.segment_window_size = int(segment_window_size)
        self.segment_stride = int(segment_stride)
        self.segment_mode = segment_mode
        self.segment_offset = int(segment_offset) if segment_offset is not None else None
        self.segment_state_normalize = bool(segment_state_normalize)

        segment_slices = self._compute_segment_slices(
            input_dim=self.input_dim,
            segment_window_size=self.segment_window_size,
            segment_stride=self.segment_stride,
            segment_mode=self.segment_mode,
            segment_offset=self.segment_offset,
        )
        validate_target_segment_overlap(segment_slices)
        self.segment_slices = segment_slices
        self.num_segments = len(segment_slices)
        self.segment_lengths = [end - start for start, end in segment_slices]
        self.max_segment_length = max(self.segment_lengths)
        self.segment_state_dim = self.max_segment_length * 3

        mask = torch.zeros(self.num_segments, self.max_segment_length, dtype=torch.float32)
        for index, (start, end) in enumerate(segment_slices):
            mask[index, : end - start] = 1.0
        self.register_buffer("segment_mask", mask, persistent=False)

        gather_idx = torch.zeros(self.num_segments, self.max_segment_length, dtype=torch.long)
        for seg_i, (start, end) in enumerate(segment_slices):
            gather_idx[seg_i, : end - start] = torch.arange(start, end)
        self.register_buffer("_gather_idx", gather_idx, persistent=False)

        self.input_position_embedding = nn.Embedding(self.input_dim, 1)
        nn.init.normal_(
            self.input_position_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.input_dim),
        )

        self.feature_map = SavitzkyGolayFeatureMapNoNorm(
            window_length=sg_window_length,
            polyorder=sg_polyorder,
        )

        num_levels = 0 if self.num_segments <= 1 else math.ceil(math.log2(self.num_segments))
        merge_input_dims = [self.segment_state_dim] + [self.chi] * max(0, num_levels - 1)
        self.merge_levels = nn.ModuleList(
            [
                FastDirectIsometricMerge(
                    input_dim=d,
                    output_dim=self.chi,
                    mode=merge_mode,
                    residual_weight=merge_residual_weight,
                    renormalize_output=merge_renormalize_output,
                )
                for d in merge_input_dims
            ]
        )

        readout_dim = self.segment_state_dim if self.num_segments == 1 else self.chi
        self.output_norm = nn.LayerNorm(readout_dim)
        self.output_head = nn.Linear(readout_dim, self.num_labels)

    @staticmethod
    def _compute_segment_slices(
        input_dim: int,
        segment_window_size: int,
        segment_stride: int,
        segment_mode: str,
        segment_offset: int | None,
    ) -> list[tuple[int, int]]:
        window_size = min(segment_window_size, input_dim)
        max_start = max(0, input_dim - window_size)
        if max_start == 0:
            return [(0, input_dim)]

        def build_even_starts(approx_stride: int) -> list[int]:
            num_segments = max(2, math.ceil(max_start / approx_stride) + 1)
            starts = [round(i * max_start / (num_segments - 1)) for i in range(num_segments)]
            return sorted(set(starts))

        def build_starts(offset: int) -> list[int]:
            if offset < 0:
                raise ValueError("segment_offset must be non-negative.")
            if offset == 0:
                return build_even_starts(segment_stride)
            starts = list(range(offset, max_start + 1, segment_stride)) if offset <= max_start else []
            starts.extend([0, max_start])
            return [s for s in starts if 0 <= s <= max_start]

        starts = build_starts(0)
        if segment_mode == "dual_offset":
            eff = segment_stride // 2 if segment_offset is None else segment_offset
            if eff > 0:
                starts.extend(build_starts(eff))

        unique_starts = sorted(set(starts))
        slices = [(s, min(s + window_size, input_dim)) for s in unique_starts]
        if slices[0][0] != 0 or slices[-1][1] != input_dim:
            raise RuntimeError("Segment slices do not cover the full input.")
        return slices

    def _segment_feature_sequence(self, features: Tensor) -> Tensor:
        idx = self._gather_idx.unsqueeze(0).expand(features.size(0), -1, -1)
        segments = torch.gather(
            features.unsqueeze(1).expand(-1, self.num_segments, -1, -1),
            dim=2,
            index=idx.unsqueeze(-1).expand(-1, -1, -1, features.size(2)),
        )
        return segments * self.segment_mask.unsqueeze(0).unsqueeze(-1)

    def _segment_states(self, segmented_features: Tensor) -> Tensor:
        flattened = segmented_features.flatten(start_dim=2)
        if self.segment_state_normalize:
            flattened = F.normalize(flattened, dim=-1, eps=1e-8)
        return flattened

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}."
            )
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}.")

        spectral_positions = torch.arange(self.input_dim, device=x.device)
        pos_emb = self.input_position_embedding(spectral_positions).squeeze(-1)

        # Derivative channels computed from clean x; positional info is only added to ch0.
        feature_sequence = self.feature_map(x)
        raw_with_pos = x + pos_emb.unsqueeze(0)
        feature_sequence = torch.cat(
            [raw_with_pos.unsqueeze(-1), feature_sequence[:, :, 1:]], dim=-1
        )
        segmented_features = self._segment_feature_sequence(feature_sequence)
        node_states = self._segment_states(segmented_features)

        level_index = 0
        while node_states.size(1) > 1:
            merge = self.merge_levels[level_index]
            num_nodes = node_states.size(1)
            paired = num_nodes // 2
            left = node_states[:, : 2 * paired : 2, :]
            right = node_states[:, 1 : 2 * paired : 2, :]
            merged = merge(left, right)
            if num_nodes % 2 == 1:
                node_states = torch.cat((merged, node_states[:, -1:, :]), dim=1)
            else:
                node_states = merged
            level_index += 1

        readout = node_states[:, 0, :]
        logits = self.output_head(self.output_norm(readout))
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits

