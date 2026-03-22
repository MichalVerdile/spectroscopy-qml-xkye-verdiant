"""Paper-nearer TTN classifier for IR spectra."""

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


class SegmentProjector(nn.Module):
    """Apply a light linear projection from local product-state features to bond space."""

    def __init__(self, max_segment_length: int, chi: int) -> None:
        super().__init__()
        if max_segment_length <= 0:
            raise ValueError("max_segment_length must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")

        self.max_segment_length = int(max_segment_length)
        self.chi = int(chi)
        input_dim = 2 * self.max_segment_length

        self.projection = nn.Linear(input_dim, self.chi, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, embedded_segments: Tensor, segment_mask: Tensor | None = None) -> Tensor:
        if embedded_segments.ndim != 4:
            raise ValueError(
                "embedded_segments must have shape "
                "(batch_size, num_segments, max_segment_length, 2)."
            )
        if embedded_segments.size(-2) != self.max_segment_length:
            raise ValueError(
                "embedded_segments has an unexpected padded segment length: "
                f"{embedded_segments.size(-2)} != {self.max_segment_length}."
            )
        if embedded_segments.size(-1) != 2:
            raise ValueError("embedded_segments must use a size-2 cosine-sine feature dimension.")

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
        projected = self.projection(flattened)
        return F.normalize(projected, dim=-1, eps=1e-8)


class PositionProjector(nn.Module):
    """Inject weak trigonometrically encoded position after the leaf projection."""

    def __init__(self, chi: int, position_scale: float = 0.25, position_strength: float = 0.1) -> None:
        super().__init__()
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive.")
        if position_strength < 0.0:
            raise ValueError("position_strength must be non-negative.")

        self.position_scale = float(position_scale)
        self.position_strength = float(position_strength)
        self.projection = nn.Linear(2, chi, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, segment_centers: Tensor) -> Tensor:
        angles = self.position_scale * segment_centers
        position_features = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        projected = self.projection(position_features)
        return self.position_strength * projected


class TensorMerge(nn.Module):
    """Bilinear TTN merge with orthogonalized factors and no NN-style residual path."""

    def __init__(self, chi: int) -> None:
        super().__init__()
        if chi <= 0:
            raise ValueError("chi must be positive.")

        self.chi = int(chi)
        self.left_factor = nn.Parameter(torch.empty(self.chi, self.chi))
        self.right_factor = nn.Parameter(torch.empty(self.chi, self.chi))
        self.output_factor = nn.Parameter(torch.empty(self.chi, self.chi))
        nn.init.orthogonal_(self.left_factor)
        nn.init.orthogonal_(self.right_factor)
        nn.init.orthogonal_(self.output_factor)

        self.core_tensor = nn.Parameter(torch.empty(self.chi, self.chi, self.chi))
        nn.init.normal_(self.core_tensor, mean=0.0, std=1.0 / math.sqrt(self.chi))

    @staticmethod
    def _orthogonalize(matrix: Tensor, eps: float = 1e-6) -> Tensor:
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

            fallback = fallback / fallback.norm().clamp_min(eps)
            vector_norm = vector.norm()
            use_fallback = (vector_norm < eps).to(dtype=vector.dtype)
            vector = vector / vector_norm.clamp_min(eps)
            vector = (1.0 - use_fallback) * vector + use_fallback * fallback
            orthogonal_columns.append(vector)

        return torch.stack(orthogonal_columns, dim=1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
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
        return F.normalize(merged, dim=-1, eps=1e-8)


class TTNIRClassifier(nn.Module):
    """Paper-nearer TTN classifier with product-state input, light leaf projection, and root readout."""

    def __init__(
        self,
        num_labels: int,
        chi: int,
        num_segments: int = 32,
        embedding_scale: float = 0.1,
        x_max_mode: str = "per_sample",
        global_x_max: float | None = None,
        position_scale: float = 0.25,
        position_strength: float = 0.1,
        input_dim: int = 1800,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if num_segments <= 0:
            raise ValueError("num_segments must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_segments > input_dim:
            raise ValueError("num_segments cannot exceed input_dim.")
        if num_segments & (num_segments - 1):
            raise ValueError("num_segments must be a power of two for a binary tree.")

        self.num_labels = int(num_labels)
        self.chi = int(chi)
        self.num_segments = int(num_segments)
        self.embedding_scale = float(embedding_scale)
        self.x_max_mode = x_max_mode
        self.input_dim = int(input_dim)

        segment_slices = self._compute_segment_slices(self.input_dim, self.num_segments)
        self.segment_slices = segment_slices
        self.segment_lengths = [end - start for start, end in segment_slices]
        self.max_segment_length = max(self.segment_lengths)

        mask = torch.zeros(self.num_segments, self.max_segment_length, dtype=torch.float32)
        centers = torch.zeros(self.num_segments, dtype=torch.float32)
        denominator = max(self.input_dim - 1, 1)
        for index, (start, end) in enumerate(segment_slices):
            length = end - start
            mask[index, :length] = 1.0
            center = 0.5 * (start + end - 1)
            centers[index] = (2.0 * center / denominator) - 1.0

        self.register_buffer("segment_mask", mask, persistent=False)
        self.register_buffer("segment_centers", centers, persistent=False)

        self.feature_map = QuantumFeatureMap(
            scale=embedding_scale,
            x_max_mode=x_max_mode,
            global_x_max=global_x_max,
        )
        self.segment_projector = SegmentProjector(self.max_segment_length, self.chi)
        self.position_projector = PositionProjector(
            chi=self.chi,
            position_scale=position_scale,
            position_strength=position_strength,
        )

        num_levels = int(math.log2(self.num_segments))
        self.merge_levels = nn.ModuleList([TensorMerge(self.chi) for _ in range(num_levels)])
        self.output_head = nn.Linear(self.chi, self.num_labels)

    @staticmethod
    def _compute_segment_slices(input_dim: int, num_segments: int) -> list[tuple[int, int]]:
        base_length, remainder = divmod(input_dim, num_segments)
        slices: list[tuple[int, int]] = []
        start = 0
        for segment_index in range(num_segments):
            length = base_length + int(segment_index < remainder)
            end = start + length
            slices.append((start, end))
            start = end

        if start != input_dim:
            raise RuntimeError("Segment computation did not cover the full input dimension.")
        return slices

    def _segment_input(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        segments = self._segment_input(x)
        embedded_segments = self.feature_map(segments, mask=self.segment_mask.unsqueeze(0))
        node_states = self.segment_projector(embedded_segments, self.segment_mask)

        position_states = self.position_projector(
            self.segment_centers.to(dtype=node_states.dtype, device=node_states.device)
        ).unsqueeze(0)
        node_states = F.normalize(node_states + position_states, dim=-1, eps=1e-8)

        for merge in self.merge_levels:
            left = node_states[:, 0::2, :]
            right = node_states[:, 1::2, :]
            node_states = merge(left, right)

        if node_states.size(1) != 1:
            raise RuntimeError("Tree reduction did not produce a single root node.")

        root = node_states[:, 0, :]
        logits = self.output_head(root)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
