"""Experiment 10.2 TTN classifier with analytical Lorentzian derivative feature map.

The Lorentzian kernel L(x) = 1 / (1 + (x/gamma)^2) and its first and second
analytical derivatives are pre-computed as conv1d filters:

    L'(x)  = -2x/gamma^2 * L(x)^2
    L''(x) = 2/gamma^2 * (3*(x/gamma)^2 - 1) / (1 + (x/gamma)^2)^3

Convolving the raw spectrum directly with these kernels avoids the two-step
smooth-then-finite-difference approach, which would re-introduce noise at the
finite-difference step.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.msms_pos.tree_tensor_network.experiment.experiment6.model import (
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    validate_target_segment_overlap,
)
from spectroscopy_qml.msms_pos.tree_tensor_network.experiment.experiment10.model import (
    FastDirectIsometricMerge,
)


class LorentzianFeatureMap(nn.Module):
    """Three-channel feature map using analytical Lorentzian derivative kernels.

    Channels:
        0. Lorentzian-smoothed spectrum  (conv with L)
        1. First analytical derivative   (conv with L')
        2. Second analytical derivative  (conv with L'')

    All three kernels are pre-computed as fixed conv1d buffers.
    Reflect-padding avoids edge discontinuities.

    Args:
        gamma: Half-width at half-maximum in data points.  Default: 3.0.
        kernel_half_width: Kernel truncated at ±kernel_half_width.  Default: 15.
        norm_mode: Per-channel normalisation: ``"max_abs"``, ``"z_score"``,
            or ``"percentile"``.  Default: ``"max_abs"``.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        gamma: float = 3.0,
        kernel_half_width: int = 15,
        norm_mode: str = "max_abs",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if kernel_half_width <= 0:
            raise ValueError("kernel_half_width must be positive.")
        if norm_mode not in {"max_abs", "z_score", "percentile"}:
            raise ValueError("norm_mode must be 'max_abs', 'z_score', or 'percentile'.")
        self.gamma = float(gamma)
        self.kernel_half_width = int(kernel_half_width)
        self.norm_mode = norm_mode
        self.eps = float(eps)

        positions = torch.arange(
            -kernel_half_width, kernel_half_width + 1, dtype=torch.float32
        )
        u = positions / gamma
        l = 1.0 / (1.0 + u ** 2)

        k0 = l / l.sum()
        k1 = -2.0 * u / gamma * l ** 2
        k2 = 2.0 / gamma ** 2 * (3.0 * u ** 2 - 1.0) / (1.0 + u ** 2) ** 3

        # conv1d computes cross-correlation; flip odd kernel k1 to get true convolution
        self.register_buffer("_k0", k0.view(1, 1, -1), persistent=True)
        self.register_buffer("_k1", k1.flip(0).view(1, 1, -1), persistent=True)
        self.register_buffer("_k2", k2.view(1, 1, -1), persistent=True)

    def _apply_kernel(self, x: Tensor, kernel: Tensor) -> Tensor:
        pad = self.kernel_half_width
        x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
        return F.conv1d(x_padded, kernel).squeeze(1)

    def _normalize_channel(self, channel: Tensor) -> Tensor:
        if self.norm_mode == "max_abs":
            scale = channel.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        elif self.norm_mode == "z_score":
            scale = channel.std(dim=-1, keepdim=True).clamp_min(self.eps)
        else:  # percentile
            idx = max(0, int(0.99 * channel.shape[-1]) - 1)
            scale = channel.abs().sort(dim=-1).values[:, idx:idx+1].clamp_min(self.eps)
        return channel / scale

    def forward(self, x: Tensor) -> Tensor:
        """Map a raw spectrum batch to three analytical derivative channels.

        Args:
            x: Shape (batch_size, spectrum_length).

        Returns:
            Shape (batch_size, spectrum_length, 3).
        """
        if not torch.is_tensor(x):
            raise TypeError("LorentzianFeatureMap expects a torch.Tensor input.")
        if x.ndim != 2:
            raise ValueError(
                f"Expected input shape (batch_size, spectrum_length), got {tuple(x.shape)}."
            )
        ch0 = self._normalize_channel(self._apply_kernel(x, self._k0))
        ch1 = self._normalize_channel(self._apply_kernel(x, self._k1))
        ch2 = self._normalize_channel(self._apply_kernel(x, self._k2))
        return torch.stack((ch0, ch1, ch2), dim=-1)


class TTNIRClassifier10_2(nn.Module):
    """Experiment 10.2 TTN classifier with Lorentzian-smoothed feature channels.

    Architecture is identical to Experiment 10.1 (direct segment states, linear
    readout) except that ``SpectralDerivativeFeatureMap`` is replaced by
    ``LorentzianFeatureMap``.  The Lorentzian smoothing suppresses the
    high-frequency noise that corrupted the finite-difference derivatives used
    in 10.1 while preserving absorption peak positions and shapes.
    """

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
        lorentz_gamma: float = 3.0,
        lorentz_kernel_half_width: int = 15,
        lorentz_norm_mode: str = "max_abs",
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
        self.segment_state_dim = self.max_segment_length * 3  # 3 feature channels

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
        nn.init.normal_(
            self.input_position_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.input_dim),
        )

        self.feature_map = LorentzianFeatureMap(
            gamma=lorentz_gamma,
            kernel_half_width=lorentz_kernel_half_width,
            norm_mode=lorentz_norm_mode,
        )

        num_levels = 0 if self.num_segments <= 1 else math.ceil(math.log2(self.num_segments))
        merge_input_dims = [self.segment_state_dim] + [self.chi] * max(0, num_levels - 1)
        self.merge_levels = nn.ModuleList(
            [
                FastDirectIsometricMerge(
                    input_dim=input_dim_level,
                    output_dim=self.chi,
                    mode=merge_mode,
                    residual_weight=merge_residual_weight,
                    renormalize_output=merge_renormalize_output,
                )
                for input_dim_level in merge_input_dims
            ]
        )

        readout_dim = self.segment_state_dim if self.num_segments == 1 else self.chi
        self.output_norm = nn.LayerNorm(readout_dim)
        self.output_head = nn.Linear(readout_dim, self.num_labels)

    # ------------------------------------------------------------------
    # Segment helpers (identical to 10.1)
    # ------------------------------------------------------------------

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

    def _segment_states(self, segmented_features: Tensor) -> Tensor:
        flattened = segmented_features.flatten(start_dim=2)
        if self.segment_state_normalize:
            flattened = F.normalize(flattened, dim=-1, eps=1e-8)
        return flattened

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}."
            )
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}.")

        spectral_positions = torch.arange(self.input_dim, device=x.device)
        pos_emb = self.input_position_embedding(spectral_positions).squeeze(-1)  # (input_dim,)

        # Derivative channels are computed from the clean spectrum x so that the
        # learnable positional bias does not inject artificial d1/d2 structure.
        feature_sequence = self.feature_map(x)  # (batch, input_dim, 3)

        # Positional embedding is added only to the raw channel (ch 0) so the
        # TTN still receives position information without corrupting derivatives.
        raw_with_pos = x + pos_emb.unsqueeze(0)
        scale = raw_with_pos.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        feature_sequence = torch.cat(
            [(raw_with_pos / scale).unsqueeze(-1), feature_sequence[:, :, 1:]], dim=-1
        )
        segmented_features = self._segment_feature_sequence(feature_sequence)
        node_states = self._segment_states(segmented_features)

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
