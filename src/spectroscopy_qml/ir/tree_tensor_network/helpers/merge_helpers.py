"""Experiment 10 TTN classifier without a separate segment leaf encoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.spectroscopy_qml.ir.tree_tensor_network.helpers.segment_helpers import (
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    validate_target_segment_overlap,
)


class FastDirectIsometricMerge(nn.Module):
    """Direct TTN merge that can accept arbitrary input state sizes."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mode: str = "relaxed",
        residual_weight: float = 0.1,
        renormalize_output: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if mode not in {"strict", "relaxed"}:
            raise ValueError("mode must be either 'strict' or 'relaxed'.")
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError("residual_weight must be in [0, 1].")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.mode = mode
        self.residual_weight = 0.0 if mode == "strict" else float(residual_weight)
        self.renormalize_output = bool(renormalize_output)

        self.raw_isometry = nn.Parameter(torch.empty(self.input_dim * self.input_dim, self.output_dim))
        nn.init.orthogonal_(self.raw_isometry)
        self.residual_proj = None
        if self.residual_weight > 0.0 and self.input_dim != self.output_dim:
            self.residual_proj = nn.Linear(self.input_dim, self.output_dim)

    @torch.compiler.disable
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("left and right must have matching shapes.")
        if left.size(-1) != self.input_dim:
            raise ValueError(f"Expected last dimension input_dim={self.input_dim}, got {left.size(-1)}.")

        Q, _ = torch.linalg.qr(self.raw_isometry.float())
        pair_state = torch.einsum("...i,...j->...ij", left, right).flatten(start_dim=-2)
        merged = torch.matmul(pair_state, Q.to(pair_state.dtype))

        if self.residual_weight > 0.0:
            residual = 0.5 * (left + right)
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            merged = (1.0 - self.residual_weight) * merged + self.residual_weight * residual

        if self.renormalize_output:
            merged = F.normalize(merged, dim=-1, eps=1e-8)
        return merged