"""Experiment 6 TTN-inspired classifier with derivative-aware spectral features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.model import (
    RelaxedIsometricMerge,
    SegmentLeafEncoder,
)


class FastRelaxedIsometricMerge(RelaxedIsometricMerge):
    """Drop-in replacement using torch.linalg.qr instead of Python Gram-Schmidt.

    The base class ``_orthonormalize_columns`` runs O(chi^2) sequential Python
    operations per call. QR decomposition via LAPACK/cuBLAS is significantly
    faster for chi >= 64 and remains numerically stable.
    """

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


class SpectralDerivativeFeatureMap(nn.Module):
    """Build three physically motivated channels from an ordered IR spectrum.

    The channels are:
    1. raw intensity
    2. first derivative
    3. second derivative

    Each channel is normalized independently per sample so the downstream TTN
    sees comparable magnitudes across the three feature types.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def _normalize_channel(self, channel: Tensor) -> Tensor:
        scale = channel.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        return channel / scale

    @staticmethod
    def _first_derivative(x: Tensor) -> Tensor:
        first = torch.zeros_like(x)
        if x.size(-1) <= 1:
            return first

        first[..., 0] = x[..., 1] - x[..., 0]
        first[..., -1] = x[..., -1] - x[..., -2]
        if x.size(-1) > 2:
            first[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2])
        return first

    @staticmethod
    def _second_derivative(x: Tensor) -> Tensor:
        second = torch.zeros_like(x)
        if x.size(-1) <= 2:
            return second

        second[..., 1:-1] = x[..., 2:] - (2.0 * x[..., 1:-1]) + x[..., :-2]
        second[..., 0] = second[..., 1]
        second[..., -1] = second[..., -2]
        return second

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_tensor(x):
            raise TypeError("SpectralDerivativeFeatureMap expects a torch.Tensor input.")
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, spectrum_length), got {tuple(x.shape)}.")

        raw = self._normalize_channel(x)
        first = self._normalize_channel(self._first_derivative(x))
        second = self._normalize_channel(self._second_derivative(x))
        return torch.stack((raw, first, second), dim=-1)


class TTNIRClassifier6(nn.Module):
    """Experiment 6 TTN classifier with derivative-aware local spectral encoding."""

    def __init__(
        self,
        num_labels: int,
        chi: int,
        input_dim: int = 1800,
        segment_window_size: int = 64,
        segment_stride: int = 32,
        segment_mode: str = "overlap",
        segment_offset: int | None = None,
        leaf_hidden_dim: int | None = None,
        leaf_dropout: float = 0.1,
        leaf_renormalize_output: bool = True,
        merge_mode: str = "relaxed",
        merge_residual_weight: float = 0.15,
        merge_renormalize_output: bool = True,
        readout_hidden_dim: int | None = None,
        readout_dropout: float = 0.1,
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

        self.feature_map = SpectralDerivativeFeatureMap()
        self.leaf_encoder = SegmentLeafEncoder(
            max_segment_length=self.max_segment_length,
            chi=self.chi,
            feature_dim=3,
            hidden_dim=leaf_hidden_dim,
            dropout=leaf_dropout,
            renormalize_output=leaf_renormalize_output,
        )
        self.segment_position_embedding = nn.Embedding(self.num_segments, self.chi)
        nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.chi))

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

        self.num_readout_scales = len(self.merge_levels) + 1
        self.readout_dim = self.num_readout_scales * self.chi
        if readout_hidden_dim is None:
            readout_hidden_dim = max(128, 4 * self.chi, self.readout_dim // 2)
        self.readout_hidden_dim = int(readout_hidden_dim)

        self.readout_norms = nn.ModuleList(
            [nn.LayerNorm(self.chi) for _ in range(self.num_readout_scales)]
        )
        self.output_norm = nn.LayerNorm(self.readout_dim)
        self.output_head = nn.Sequential(
            nn.Linear(self.readout_dim, self.readout_hidden_dim),
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

        def build_starts(offset: int) -> list[int]:
            if offset < 0:
                raise ValueError("segment_offset must be non-negative.")
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

    @staticmethod
    def _pool_level(node_states: Tensor) -> Tensor:
        if node_states.ndim != 3:
            raise ValueError("node_states must have shape (batch_size, num_nodes, chi).")
        return node_states.mean(dim=1)

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}.")

        feature_sequence = self.feature_map(x)
        segmented_features = self._segment_feature_sequence(feature_sequence)
        node_states = self.leaf_encoder(segmented_features, self.segment_mask)
        segment_indices = torch.arange(self.num_segments, device=x.device)
        node_states = node_states + self.segment_position_embedding(segment_indices).unsqueeze(0)
        multi_scale_states = [self.readout_norms[0](self._pool_level(node_states))]

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

            multi_scale_states.append(self.readout_norms[level_index + 1](self._pool_level(node_states)))
            level_index += 1

        readout = torch.cat(multi_scale_states, dim=-1)
        logits = self.output_head(self.output_norm(readout))
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
