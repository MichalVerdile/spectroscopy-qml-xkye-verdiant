"""Experiment 5 TTN-inspired classifier for ordered IR spectra."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QuantumFeatureMap(nn.Module):
    """Apply a cosine-sine feature map to scalar inputs."""

    def __init__(
        self,
        scale: float = 0.1,
        x_max_mode: str = "per_sample",
        global_x_max: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.x_max_mode = x_max_mode
        self.eps = float(eps)

        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")
        if self.x_max_mode not in {"per_sample", "global"}:
            raise ValueError("x_max_mode must be either 'per_sample' or 'global'.")
        if self.x_max_mode == "global":
            if global_x_max is None or global_x_max <= 0.0:
                raise ValueError("global_x_max must be provided and positive in global mode.")

        global_scale = 1.0 if global_x_max is None else float(global_x_max)
        self.register_buffer(
            "global_x_max",
            torch.tensor(global_scale, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if not torch.is_tensor(x):
            raise TypeError("QuantumFeatureMap expects a torch.Tensor input.")

        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)

        if self.x_max_mode == "per_sample":
            scaled_source = x.abs()
            if mask is not None:
                scaled_source = scaled_source * mask
            reduce_dims = tuple(range(1, x.ndim))
            x_max = scaled_source.amax(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
        else:
            x_max = self.global_x_max.to(dtype=x.dtype, device=x.device)

        angles = self.scale * (x / x_max)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


class SegmentLeafEncoder(nn.Module):
    """Encode one local IR segment into a TTN bond vector."""

    def __init__(
        self,
        max_segment_length: int,
        chi: int,
        feature_dim: int = 3,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
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
        if hidden_dim is None:
            hidden_dim = max(4 * self.chi, 2 * input_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, self.chi)
        self.skip = nn.Linear(input_dim, self.chi)
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
        encoded = self.input_norm(flattened)
        encoded = self.fc1(encoded)
        encoded = self.activation(encoded)
        encoded = self.dropout(encoded)
        encoded = self.fc2(encoded)
        encoded = encoded + self.skip(flattened)
        encoded = self.output_norm(encoded)
        if self.renormalize_output:
            encoded = F.normalize(encoded, dim=-1, eps=1e-8)
        return encoded


class RelaxedIsometricMerge(nn.Module):
    """TTN-style isometric merge with optional residual relaxation."""

    def __init__(
        self,
        chi: int,
        mode: str = "relaxed",
        residual_weight: float = 0.15,
        renormalize_output: bool = True,
    ) -> None:
        super().__init__()
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if mode not in {"strict", "relaxed"}:
            raise ValueError("mode must be either 'strict' or 'relaxed'.")
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError("residual_weight must be in [0, 1].")

        self.chi = int(chi)
        self.mode = mode
        self.residual_weight = 0.0 if mode == "strict" else float(residual_weight)
        self.renormalize_output = bool(renormalize_output)
        self.raw_isometry = nn.Parameter(torch.empty(self.chi * self.chi, self.chi))
        nn.init.orthogonal_(self.raw_isometry)

    @staticmethod
    def _orthonormalize_columns(matrix: Tensor, eps: float = 1e-6) -> Tensor:
        if matrix.ndim != 2:
            raise ValueError("matrix must be a 2D tensor.")

        columns = matrix.unbind(dim=1)
        orthonormal_columns: list[Tensor] = []

        for column_index, column in enumerate(columns):
            vector = column
            for basis in orthonormal_columns:
                vector = vector - torch.dot(vector, basis) * basis

            fallback = matrix.new_zeros(matrix.size(0))
            fallback[column_index % matrix.size(0)] = 1.0
            for basis in orthonormal_columns:
                fallback = fallback - torch.dot(fallback, basis) * basis

            fallback = fallback / fallback.norm().clamp_min(eps)
            vector_norm = vector.norm()
            use_fallback = (vector_norm < eps).to(dtype=vector.dtype)
            vector = vector / vector_norm.clamp_min(eps)
            vector = (1.0 - use_fallback) * vector + use_fallback * fallback
            orthonormal_columns.append(vector)

        return torch.stack(orthonormal_columns, dim=1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != right.ndim:
            raise ValueError("left and right must have the same number of dimensions.")
        if left.shape != right.shape:
            raise ValueError("left and right must have matching shapes.")
        if left.size(-1) != self.chi:
            raise ValueError(f"Expected last dimension chi={self.chi}, got {left.size(-1)}.")

        pair_state = torch.einsum("...i,...j->...ij", left, right).flatten(start_dim=-2)
        merged = torch.matmul(pair_state, self._orthonormalize_columns(self.raw_isometry))

        if self.residual_weight > 0.0:
            residual = 0.5 * (left + right)
            merged = (1.0 - self.residual_weight) * merged + self.residual_weight * residual

        if self.renormalize_output:
            merged = F.normalize(merged, dim=-1, eps=1e-8)
        return merged


class TTNIRClassifier5(nn.Module):
    """Structured TTN-inspired classifier with overlapping local encoders and multi-scale readout."""

    def __init__(
        self,
        num_labels: int,
        chi: int,
        input_dim: int = 1800,
        segment_window_size: int = 64,
        segment_stride: int = 32,
        segment_mode: str = "overlap",
        segment_offset: int | None = None,
        embedding_scale: float = 0.1,
        x_max_mode: str = "per_sample",
        global_x_max: float | None = None,
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
        positions = torch.zeros(self.num_segments, self.max_segment_length, dtype=torch.float32)
        denominator = max(self.input_dim - 1, 1)
        for index, (start, end) in enumerate(segment_slices):
            length = end - start
            mask[index, :length] = 1.0
            coords = torch.arange(start, end, dtype=torch.float32)
            positions[index, :length] = (2.0 * coords / denominator) - 1.0

        self.register_buffer("segment_mask", mask, persistent=False)
        self.register_buffer("segment_positions", positions, persistent=False)

        self.feature_map = QuantumFeatureMap(
            scale=embedding_scale,
            x_max_mode=x_max_mode,
            global_x_max=global_x_max,
        )
        self.leaf_encoder = SegmentLeafEncoder(
            max_segment_length=self.max_segment_length,
            chi=self.chi,
            feature_dim=3,
            hidden_dim=leaf_hidden_dim,
            dropout=leaf_dropout,
            renormalize_output=leaf_renormalize_output,
        )

        num_levels = 0 if self.num_segments <= 1 else math.ceil(math.log2(self.num_segments))
        self.merge_levels = nn.ModuleList(
            [
                RelaxedIsometricMerge(
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
        """Compute configurable segment slices that preserve 1D order."""
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

    def _segment_input(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}.")

        batch_size = x.size(0)
        segments = x.new_zeros(batch_size, self.num_segments, self.max_segment_length)
        for segment_index, (start, end) in enumerate(self.segment_slices):
            segments[:, segment_index, : end - start] = x[:, start:end]
        return segments

    @staticmethod
    def _pool_level(node_states: Tensor) -> Tensor:
        if node_states.ndim != 3:
            raise ValueError("node_states must have shape (batch_size, num_nodes, chi).")
        return node_states.mean(dim=1)

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        segments = self._segment_input(x)
        embedded_segments = self.feature_map(segments, mask=self.segment_mask.unsqueeze(0))
        position_features = self.segment_positions.unsqueeze(0).unsqueeze(-1).to(
            dtype=embedded_segments.dtype,
            device=embedded_segments.device,
        )
        position_features = position_features.expand(embedded_segments.size(0), -1, -1, -1)
        augmented_segments = torch.cat((embedded_segments, position_features), dim=-1)

        node_states = self.leaf_encoder(augmented_segments, self.segment_mask)
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
