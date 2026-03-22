"""Legacy TTN-inspired IR classifier kept for experiment comparison."""

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
        """Map an input tensor to its cosine-sine embedding."""
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
    """Project padded embedded segments to a fixed bond dimension."""

    def __init__(self, max_segment_length: int, chi: int) -> None:
        super().__init__()
        if max_segment_length <= 0:
            raise ValueError("max_segment_length must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")

        self.max_segment_length = int(max_segment_length)
        self.chi = int(chi)
        self.projection = nn.Linear(2 * self.max_segment_length, self.chi)

    def forward(self, embedded_segments: Tensor, segment_mask: Tensor | None = None) -> Tensor:
        """Compress padded embedded segments."""
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

            if segment_mask.size(1) != embedded_segments.size(1):
                raise ValueError("segment_mask has an unexpected num_segments dimension.")
            if segment_mask.size(2) != self.max_segment_length:
                raise ValueError("segment_mask has an unexpected max_segment_length dimension.")

            masked_segments = embedded_segments * segment_mask.to(
                dtype=embedded_segments.dtype,
                device=embedded_segments.device,
            )

        flattened = masked_segments.flatten(start_dim=2)
        compressed = self.projection(flattened)
        return F.normalize(compressed, dim=-1, eps=1e-8)


class TensorMerge(nn.Module):
    """Merge two bond vectors with a learnable rank-3 tensor contraction."""

    def __init__(self, chi: int, use_bias: bool = True) -> None:
        super().__init__()
        if chi <= 0:
            raise ValueError("chi must be positive.")

        self.chi = int(chi)
        self.tensor = nn.Parameter(torch.empty(self.chi, self.chi, self.chi))
        nn.init.normal_(self.tensor, mean=0.0, std=1.0 / self.chi)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.chi))
        else:
            self.register_parameter("bias", None)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Merge two tensors whose last dimension is ``chi``."""
        if left.ndim != right.ndim:
            raise ValueError("left and right must have the same number of dimensions.")
        if left.shape != right.shape:
            raise ValueError("left and right must have matching shapes.")
        if left.size(-1) != self.chi:
            raise ValueError(f"Expected last dimension to be chi={self.chi}, got {left.size(-1)}.")

        merged = torch.einsum("...i,...j,rij->...r", left, right, self.tensor)
        if self.bias is not None:
            merged = merged + self.bias
        return F.normalize(merged, dim=-1, eps=1e-8)


class LegacyTTNIRClassifier(nn.Module):
    """Legacy TTN-inspired multi-label classifier for IR spectra."""

    def __init__(
        self,
        num_labels: int,
        chi: int,
        num_segments: int = 32,
        embedding_scale: float = 0.1,
        x_max_mode: str = "per_sample",
        global_x_max: float | None = None,
        use_bias: bool = True,
        input_dim: int = 1800,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_segments <= 0:
            raise ValueError("num_segments must be positive.")
        if num_segments > input_dim:
            raise ValueError("num_segments cannot exceed input_dim.")
        if num_segments & (num_segments - 1):
            raise ValueError("num_segments must be a power of two for a binary tree.")

        self.num_labels = int(num_labels)
        self.chi = int(chi)
        self.num_segments = int(num_segments)
        self.embedding_scale = float(embedding_scale)
        self.x_max_mode = x_max_mode
        self.use_bias = bool(use_bias)
        self.input_dim = int(input_dim)

        segment_slices = self._compute_segment_slices(self.input_dim, self.num_segments)
        self.segment_slices = segment_slices
        self.segment_lengths = [end - start for start, end in segment_slices]
        self.max_segment_length = max(self.segment_lengths)

        mask = torch.zeros(self.num_segments, self.max_segment_length, dtype=torch.float32)
        for index, length in enumerate(self.segment_lengths):
            mask[index, :length] = 1.0
        self.register_buffer("segment_mask", mask, persistent=False)

        self.feature_map = QuantumFeatureMap(
            scale=self.embedding_scale,
            x_max_mode=self.x_max_mode,
            global_x_max=global_x_max,
        )
        self.segment_compressor = SegmentCompressor(
            max_segment_length=self.max_segment_length,
            chi=self.chi,
        )

        num_levels = int(math.log2(self.num_segments))
        self.merge_levels = nn.ModuleList(
            [TensorMerge(chi=self.chi, use_bias=self.use_bias) for _ in range(num_levels)]
        )
        head_hidden_dim = max(128, 2 * self.chi)
        self.output_norm = nn.LayerNorm(self.chi)
        self.output_head = nn.Sequential(
            nn.Linear(self.chi, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, self.num_labels),
        )

    @staticmethod
    def _compute_segment_slices(input_dim: int, num_segments: int) -> list[tuple[int, int]]:
        """Compute ordered contiguous slices with near-equal lengths."""
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
        """Split spectra into padded contiguous segments."""
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
        """Run the legacy TTN classifier."""
        segments = self._segment_input(x)
        embedded_segments = self.feature_map(segments, mask=self.segment_mask.unsqueeze(0))
        node_states = self.segment_compressor(embedded_segments, self.segment_mask)

        for level_index, merge in enumerate(self.merge_levels):
            num_nodes = node_states.size(1)
            if num_nodes % 2 != 0:
                raise RuntimeError(
                    f"Tree level {level_index} received an odd number of nodes: {num_nodes}."
                )

            left = node_states[:, 0::2, :]
            right = node_states[:, 1::2, :]
            node_states = merge(left, right)

        if node_states.size(1) != 1:
            raise RuntimeError("Tree reduction did not produce a single root node.")

        root = node_states[:, 0, :]
        logits = self.output_head(self.output_norm(root))

        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
