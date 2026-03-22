"""PyTorch TTN-inspired multi-label classifier for IR spectra."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


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
        """Map an input tensor to a cosine-sine embedding."""
        if not torch.is_tensor(x):
            raise TypeError("QuantumFeatureMap expects a torch.Tensor input.")

        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)

        if self.x_max_mode == "per_sample":
            scaled_source = x.abs()
            if mask is not None:
                scaled_source = scaled_source * mask
            if x.ndim <= 1:
                x_max = scaled_source.amax().clamp_min(self.eps)
            else:
                reduce_dims = tuple(range(1, x.ndim))
                x_max = scaled_source.amax(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
        else:
            x_max = self.global_x_max.to(dtype=x.dtype, device=x.device)

        x_scaled = x / x_max
        angles = self.scale * x_scaled
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


class SegmentCompressor(nn.Module):
    """Compress one overlapping window into a TTN site vector."""

    def __init__(self, max_segment_length: int, chi: int, feature_dim: int = 3) -> None:
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

        input_dim = self.max_segment_length * self.feature_dim
        hidden_dim = max(4 * self.chi, 2 * input_dim)

        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, self.chi)
        self.skip = nn.Linear(input_dim, self.chi)
        self.output_norm = nn.LayerNorm(self.chi)

    def forward(self, embedded_segments: Tensor, segment_mask: Tensor | None = None) -> Tensor:
        """Compress padded embedded segments to TTN site vectors."""
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
                f"embedded_segments must use feature_dim={self.feature_dim}, "
                f"got {embedded_segments.size(-1)}."
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

            if segment_mask.size(1) != embedded_segments.size(1):
                raise ValueError("segment_mask has an unexpected num_segments dimension.")
            if segment_mask.size(2) != self.max_segment_length:
                raise ValueError("segment_mask has an unexpected max_segment_length dimension.")

            masked_segments = embedded_segments * segment_mask.to(
                dtype=embedded_segments.dtype,
                device=embedded_segments.device,
            )

        flattened = masked_segments.flatten(start_dim=2)
        normalized = self.input_norm(flattened)
        compressed = self.fc1(normalized)
        compressed = self.activation(compressed)
        compressed = self.dropout(compressed)
        compressed = self.fc2(compressed)
        compressed = compressed + self.skip(flattened)
        return self.output_norm(compressed)


class TensorMerge(nn.Module):
    """Merge two bond vectors with an orthogonalized tensor contraction."""

    def __init__(
        self,
        chi: int,
        use_bias: bool = True,
        normalization: str = "layernorm",
        residual_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if normalization not in {"layernorm", "none"}:
            raise ValueError("normalization must be either 'layernorm' or 'none'.")
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError("residual_weight must be in the range [0, 1].")

        self.chi = int(chi)
        self.normalization = normalization

        self.left_factor = nn.Parameter(torch.empty(self.chi, self.chi))
        self.right_factor = nn.Parameter(torch.empty(self.chi, self.chi))
        self.output_factor = nn.Parameter(torch.empty(self.chi, self.chi))
        nn.init.orthogonal_(self.left_factor)
        nn.init.orthogonal_(self.right_factor)
        nn.init.orthogonal_(self.output_factor)

        self.core_tensor = nn.Parameter(torch.empty(self.chi, self.chi, self.chi))
        nn.init.normal_(self.core_tensor, mean=0.0, std=1.0 / math.sqrt(self.chi))

        residual_logit = torch.logit(torch.tensor(residual_weight).clamp(1e-4, 1.0 - 1e-4))
        self.residual_gate = nn.Parameter(residual_logit.reshape(()))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.chi))
        else:
            self.register_parameter("bias", None)

        self.output_norm = nn.LayerNorm(self.chi) if normalization == "layernorm" else nn.Identity()

    @staticmethod
    def _orthogonalize(matrix: Tensor, eps: float = 1e-6) -> Tensor:
        """Project a square matrix to an orthonormal column basis without QR.

        This uses differentiable modified Gram-Schmidt steps built from basic
        tensor ops so it remains compatible with the MPS backend.
        """
        if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
            raise ValueError("matrix must be a square 2D tensor.")

        num_cols = matrix.size(1)
        columns = matrix.transpose(0, 1).unbind(dim=0)
        orthogonal_columns: list[Tensor] = []

        for column_index, column in enumerate(columns):
            vector = column
            for basis in orthogonal_columns:
                vector = vector - torch.dot(vector, basis) * basis

            fallback = matrix.new_zeros(matrix.size(0))
            fallback[column_index] = 1.0
            for basis in orthogonal_columns:
                fallback = fallback - torch.dot(fallback, basis) * basis

            fallback_norm = fallback.norm().clamp_min(eps)
            fallback = fallback / fallback_norm

            vector_norm = vector.norm()
            use_fallback = (vector_norm < eps).to(dtype=vector.dtype)
            vector = vector / vector_norm.clamp_min(eps)
            vector = (1.0 - use_fallback) * vector + use_fallback * fallback
            orthogonal_columns.append(vector)

        return torch.stack(orthogonal_columns, dim=1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Merge two tensors whose last dimension is ``chi``."""
        if left.ndim != right.ndim:
            raise ValueError("left and right must have the same number of dimensions.")
        if left.shape != right.shape:
            raise ValueError("left and right must have matching shapes.")
        if left.size(-1) != self.chi:
            raise ValueError(f"Expected last dimension to be chi={self.chi}, got {left.size(-1)}.")

        left_iso = torch.matmul(left, self._orthogonalize(self.left_factor))
        right_iso = torch.matmul(right, self._orthogonalize(self.right_factor))
        merged = torch.einsum("...i,...j,ijk->...k", left_iso, right_iso, self.core_tensor)
        merged = torch.matmul(merged, self._orthogonalize(self.output_factor).transpose(0, 1))

        residual = 0.5 * (left + right)
        merged = merged + torch.sigmoid(self.residual_gate) * residual
        if self.bias is not None:
            merged = merged + self.bias
        return self.output_norm(merged)


class TTNIRClassifier(nn.Module):
    """TTN-inspired multi-label classifier for IR spectra with overlapping segments.

    The model:
    1. Extracts overlapping sliding windows from the spectrum.
    2. Applies a cosine-sine feature map to every scalar intensity.
    3. Adds normalized absolute position information per spectral point.
    4. Compresses each local window to a TTN site vector.
    5. Builds a hierarchical binary tree with tensor-based merge operators.
    6. Pools multiple TTN levels and maps the fused multi-scale vector to logits.
    """

    def __init__(
        self,
        num_labels: int,
        chi: int,
        num_segments: int | None = None,
        segment_window_size: int | None = None,
        segment_stride: int | None = None,
        embedding_scale: float = 0.1,
        x_max_mode: str = "per_sample",
        global_x_max: float | None = None,
        use_bias: bool = True,
        merge_normalization: str = "layernorm",
        merge_residual_weight: float = 0.25,
        input_dim: int = 1800,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_segments is not None and num_segments <= 0:
            raise ValueError("num_segments must be positive when provided.")

        if segment_window_size is None:
            if num_segments is None:
                num_segments = 64
            approx_stride = max(1, math.ceil(input_dim / num_segments))
            segment_window_size = min(input_dim, max(approx_stride * 2, approx_stride + 1))
        if segment_stride is None:
            segment_stride = max(1, segment_window_size // 2)

        if segment_window_size <= 0:
            raise ValueError("segment_window_size must be positive.")
        if segment_stride <= 0:
            raise ValueError("segment_stride must be positive.")

        self.num_labels = int(num_labels)
        self.chi = int(chi)
        self.embedding_scale = float(embedding_scale)
        self.x_max_mode = x_max_mode
        self.use_bias = bool(use_bias)
        self.merge_normalization = merge_normalization
        self.merge_residual_weight = float(merge_residual_weight)
        self.input_dim = int(input_dim)
        self.segment_window_size = int(segment_window_size)
        self.segment_stride = int(segment_stride)
        self.position_feature_dim = 1

        segment_slices = self._compute_segment_slices(
            self.input_dim,
            self.segment_window_size,
            self.segment_stride,
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
            scale=self.embedding_scale,
            x_max_mode=self.x_max_mode,
            global_x_max=global_x_max,
        )
        self.segment_compressor = SegmentCompressor(
            max_segment_length=self.max_segment_length,
            chi=self.chi,
            feature_dim=2 + self.position_feature_dim,
        )

        num_levels = 0 if self.num_segments <= 1 else math.ceil(math.log2(self.num_segments))
        self.merge_levels = nn.ModuleList(
            [
                TensorMerge(
                    chi=self.chi,
                    use_bias=self.use_bias,
                    normalization=self.merge_normalization,
                    residual_weight=self.merge_residual_weight,
                )
                for _ in range(num_levels)
            ]
        )
        self.num_readout_scales = len(self.merge_levels) + 1
        self.multi_scale_readout_dim = self.num_readout_scales * self.chi
        head_hidden_dim = max(128, 4 * self.chi, self.multi_scale_readout_dim // 2)
        self.readout_norms = nn.ModuleList(
            [nn.LayerNorm(self.chi) for _ in range(self.num_readout_scales)]
        )
        self.output_norm = nn.LayerNorm(self.multi_scale_readout_dim)
        self.output_head = nn.Sequential(
            nn.Linear(self.multi_scale_readout_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, self.num_labels),
        )

    @staticmethod
    def _compute_segment_slices(
        input_dim: int,
        segment_window_size: int,
        segment_stride: int,
    ) -> list[tuple[int, int]]:
        """Compute overlapping sliding windows that cover the full input."""
        window_size = min(segment_window_size, input_dim)
        max_start = max(0, input_dim - window_size)

        starts = list(range(0, max_start + 1, segment_stride))
        if not starts:
            starts = [0]
        if starts[-1] != max_start:
            starts.append(max_start)

        starts = sorted(set(starts))
        slices = [(start, min(start + window_size, input_dim)) for start in starts]
        if slices[0][0] != 0 or slices[-1][1] != input_dim:
            raise RuntimeError("Sliding-window segmentation did not cover the full input.")
        return slices

    def _segment_input(self, x: Tensor) -> Tensor:
        """Split spectra into padded overlapping windows."""
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, got {x.size(1)}."
            )

        batch_size = x.size(0)
        padded_segments = x.new_zeros(batch_size, self.num_segments, self.max_segment_length)

        for segment_index, (start, end) in enumerate(self.segment_slices):
            segment = x[:, start:end]
            padded_segments[:, segment_index, : end - start] = segment

        return padded_segments

    @staticmethod
    def _pool_level(node_states: Tensor) -> Tensor:
        """Pool a TTN level to a single vector per sample."""
        if node_states.ndim != 3:
            raise ValueError("node_states must have shape (batch_size, num_nodes, chi).")
        return node_states.mean(dim=1)

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        """Run the TTN classifier."""
        segments = self._segment_input(x)
        embedded_segments = self.feature_map(segments, mask=self.segment_mask.unsqueeze(0))

        position_features = self.segment_positions.unsqueeze(0).unsqueeze(-1).to(
            dtype=embedded_segments.dtype,
            device=embedded_segments.device,
        )
        position_features = position_features.expand(embedded_segments.size(0), -1, -1, -1)
        augmented_segments = torch.cat((embedded_segments, position_features), dim=-1)

        node_states = self.segment_compressor(augmented_segments, self.segment_mask)
        if node_states.ndim != 3:
            raise RuntimeError("Segment compressor must return a 3D tensor.")
        if node_states.size(1) != self.num_segments or node_states.size(2) != self.chi:
            raise RuntimeError("Segment compressor returned an unexpected shape.")

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

        if len(multi_scale_states) != self.num_readout_scales:
            raise RuntimeError("Collected multi-scale features do not match the configured readout.")

        readout = torch.cat(multi_scale_states, dim=-1)
        logits = self.output_head(self.output_norm(readout))

        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
