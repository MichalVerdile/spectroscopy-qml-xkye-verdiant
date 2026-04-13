"""Experiment 9 TTN classifier with derivative-aware features and no leaf hidden layer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.model import (
    RelaxedIsometricMerge,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.model import (
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    SpectralDerivativeFeatureMap,
    validate_target_segment_overlap,
)


class FastRelaxedIsometricMerge(RelaxedIsometricMerge):
    """Drop-in replacement using QR instead of Python Gram-Schmidt."""

    @torch.compiler.disable
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("left and right must have matching shapes.")
        if left.size(-1) != self.chi:
            raise ValueError(f"Expected last dimension chi={self.chi}, got {left.size(-1)}.")

        Q, _ = torch.linalg.qr(self.raw_isometry.float())
        pair_state = torch.einsum("...i,...j->...ij", left, right).flatten(start_dim=-2)
        merged = torch.matmul(pair_state, Q.to(pair_state.dtype))

        if self.residual_weight > 0.0:
            residual = 0.5 * (left + right)
            merged = (1.0 - self.residual_weight) * merged + self.residual_weight * residual

        if self.renormalize_output:
            merged = F.normalize(merged, dim=-1, eps=1e-8)
        return merged


class DirectSegmentLeafEncoder(nn.Module):
    """Encode a segment with a direct residual leaf block and no intermediate hidden layer."""

    def __init__(
        self,
        max_segment_length: int,
        chi: int,
        feature_dim: int = 3,
        dropout: float = 0.05,
        renormalize_output: bool = True,
    ) -> None:
        super().__init__()
        if max_segment_length <= 0:
            raise ValueError("max_segment_length must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")

        self.max_segment_length = int(max_segment_length)
        self.chi = int(chi)
        self.feature_dim = int(feature_dim)
        self.renormalize_output = bool(renormalize_output)

        input_dim = self.max_segment_length * self.feature_dim
        self.input_dim = int(input_dim)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.proj = nn.Linear(self.input_dim, self.chi)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(self.input_dim, self.chi)
        self.output_norm = nn.LayerNorm(self.chi)

    def forward(self, embedded_segments: Tensor, segment_mask: Tensor | None = None) -> Tensor:
        if embedded_segments.ndim != 4:
            raise ValueError(
                "embedded_segments must have shape "
                "(batch_size, num_segments, max_segment_length, feature_dim)."
            )
        if embedded_segments.size(-2) != self.max_segment_length:
            raise ValueError(
                "embedded_segments has an unexpected padded segment length: "
                f"{embedded_segments.size(-2)} != {self.max_segment_length}."
            )
        if embedded_segments.size(-1) != self.feature_dim:
            raise ValueError(
                f"embedded_segments must use feature_dim={self.feature_dim}, got {embedded_segments.size(-1)}."
            )

        masked_segments = embedded_segments
        if segment_mask is not None:
            if segment_mask.ndim == 2:
                segment_mask = segment_mask.unsqueeze(0).unsqueeze(-1)
            elif segment_mask.ndim != 4:
                raise ValueError(
                    "segment_mask must have shape (num_segments, max_segment_length) or "
                    "(1, num_segments, max_segment_length, 1)."
                )
            masked_segments = embedded_segments * segment_mask.to(
                dtype=embedded_segments.dtype,
                device=embedded_segments.device,
            )

        flattened = masked_segments.flatten(start_dim=2)
        normalized = self.input_norm(flattened)
        encoded = self.proj(normalized)
        encoded = self.activation(encoded)
        encoded = self.dropout(encoded)
        encoded = encoded + self.skip(flattened)
        encoded = self.output_norm(encoded)
        if self.renormalize_output:
            encoded = F.normalize(encoded, dim=-1, eps=1e-8)
        return encoded


class TTNIRClassifier9(nn.Module):
    """Experiment 9 TTN classifier with a direct residual leaf projection."""

    def __init__(
        self,
        num_labels: int,
        chi: int,
        input_dim: int = 1800,
        segment_window_size: int = DEFAULT_SEGMENT_WINDOW_SIZE,
        segment_stride: int = DEFAULT_SEGMENT_STRIDE,
        segment_mode: str = "overlap",
        segment_offset: int | None = None,
        leaf_dropout: float = 0.05,
        leaf_renormalize_output: bool = True,
        merge_mode: str = "relaxed",
        merge_residual_weight: float = 0.1,
        merge_renormalize_output: bool = True,
        readout_hidden_dim: int | None = None,
        readout_dropout: float = 0.0,
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
            raise ValueError("segment_mode must be either 'overlap' or 'dual_offset'.")

        self.num_labels = int(num_labels)
        self.chi = int(chi)
        self.input_dim = int(input_dim)
        self.segment_window_size = int(segment_window_size)
        self.segment_stride = int(segment_stride)
        self.segment_mode = segment_mode
        self.segment_offset = int(segment_offset) if segment_offset is not None else None

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

        mask = torch.zeros(self.num_segments, self.max_segment_length, dtype=torch.float32)
        for index, (start, end) in enumerate(segment_slices):
            mask[index, : end - start] = 1.0
        self.register_buffer("segment_mask", mask, persistent=False)

        gather_idx = torch.zeros(self.num_segments, self.max_segment_length, dtype=torch.long)
        for seg_i, (start, end) in enumerate(segment_slices):
            seg_len = end - start
            gather_idx[seg_i, :seg_len] = torch.arange(start, end)
        self.register_buffer("_gather_idx", gather_idx, persistent=False)

        self.input_position_embedding = nn.Embedding(self.input_dim, 1)
        nn.init.normal_(self.input_position_embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.input_dim))
        self.feature_map = SpectralDerivativeFeatureMap()
        self.leaf_encoder = DirectSegmentLeafEncoder(
            max_segment_length=self.max_segment_length,
            chi=self.chi,
            feature_dim=3,
            dropout=leaf_dropout,
            renormalize_output=leaf_renormalize_output,
        )

        num_levels = 0 if self.num_segments <= 1 else math.ceil(math.log2(self.num_segments))
        self.merge_levels = nn.ModuleList(
            [
                FastRelaxedIsometricMerge(
                    chi=self.chi,
                    mode=merge_mode,
                    residual_weight=merge_residual_weight,
                    renormalize_output=merge_renormalize_output,
                )
                for _ in range(num_levels)
            ]
        )

        if readout_hidden_dim is None:
            readout_hidden_dim = max(128, 4 * self.chi)
        self.readout_hidden_dim = int(readout_hidden_dim)

        self.output_norm = nn.LayerNorm(self.chi)
        self.output_head = nn.Sequential(
            nn.Linear(self.chi, self.readout_hidden_dim),
            nn.GELU(),
            nn.Dropout(readout_dropout),
            nn.Linear(self.readout_hidden_dim, self.num_labels),
        )

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
            starts = [round(index * max_start / (num_segments - 1)) for index in range(num_segments)]
            return sorted(set(starts))

        def build_starts(offset: int) -> list[int]:
            if offset < 0:
                raise ValueError("segment_offset must be non-negative.")
            if offset == 0:
                return build_even_starts(segment_stride)

            starts = list(range(offset, max_start + 1, segment_stride)) if offset <= max_start else []
            starts.extend([0, max_start])
            return [start for start in starts if 0 <= start <= max_start]

        starts = build_starts(0)
        if segment_mode == "dual_offset":
            effective_offset = segment_stride // 2 if segment_offset is None else segment_offset
            if effective_offset > 0:
                starts.extend(build_starts(effective_offset))

        unique_starts = sorted(set(starts))
        slices = [(start, min(start + window_size, input_dim)) for start in unique_starts]
        if slices[0][0] != 0 or slices[-1][1] != input_dim:
            raise RuntimeError("Configured segment slices do not cover the full input.")
        return slices

    def _segment_feature_sequence(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(
                "Expected feature tensor shape (batch_size, input_dim, feature_dim), "
                f"got {tuple(features.shape)}."
            )
        if features.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {features.size(1)}.")

        idx = self._gather_idx.unsqueeze(0).expand(features.size(0), -1, -1)
        segments = torch.gather(
            features.unsqueeze(1).expand(-1, self.num_segments, -1, -1),
            dim=2,
            index=idx.unsqueeze(-1).expand(-1, -1, -1, features.size(2)),
        )
        segments = segments * self.segment_mask.unsqueeze(0).unsqueeze(-1)
        return segments

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}.")

        spectral_positions = torch.arange(self.input_dim, device=x.device)
        positioned_raw = x + self.input_position_embedding(spectral_positions).squeeze(-1).unsqueeze(0)
        feature_sequence = self.feature_map(positioned_raw)
        segmented_features = self._segment_feature_sequence(feature_sequence)
        node_states = self.leaf_encoder(segmented_features, self.segment_mask)

        level_index = 0
        while node_states.size(1) > 1:
            merge = self.merge_levels[level_index]
            num_nodes = node_states.size(1)
            paired_nodes = num_nodes // 2

            left = node_states[:, : 2 * paired_nodes : 2, :]
            right = node_states[:, 1 : 2 * paired_nodes : 2, :]
            merged = merge(left, right)

            if num_nodes % 2 == 1:
                carry = node_states[:, -1:, :]
                node_states = torch.cat((merged, carry), dim=1)
            else:
                node_states = merged

            level_index += 1

        readout = node_states[:, 0, :]
        logits = self.output_head(self.output_norm(readout))
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
