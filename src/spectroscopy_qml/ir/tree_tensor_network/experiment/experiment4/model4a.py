"""Experiment4A: stricter isometric TTN merge variant for IR spectra."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .model import TTNIRClassifier as BaseTTNIRClassifier
except ImportError:
    from model import TTNIRClassifier as BaseTTNIRClassifier


class IsometricTensorMerge(nn.Module):
    """Merge two child states with a row-orthonormal TTN-style isometry."""

    def __init__(self, chi: int, renormalize_output: bool = True) -> None:
        super().__init__()
        if chi <= 0:
            raise ValueError("chi must be positive.")

        self.chi = int(chi)
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
            raise ValueError(f"Expected last dimension to be chi={self.chi}, got {left.size(-1)}.")

        pair_state = torch.einsum("...i,...j->...ij", left, right).flatten(start_dim=-2)
        isometry = self._orthonormalize_columns(self.raw_isometry)
        merged = torch.matmul(pair_state, isometry)

        if self.renormalize_output:
            return F.normalize(merged, dim=-1, eps=1e-8)
        return merged


class TTNIRClassifier4A(BaseTTNIRClassifier):
    """Experiment4 frontend with a stricter TTN-style merge operator."""

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
        merge_renormalize_output: bool = True,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            chi=chi,
            num_segments=num_segments,
            embedding_scale=embedding_scale,
            x_max_mode=x_max_mode,
            global_x_max=global_x_max,
            position_scale=position_scale,
            position_strength=position_strength,
            input_dim=input_dim,
        )
        self.merge_renormalize_output = bool(merge_renormalize_output)
        self.merge_levels = nn.ModuleList(
            [
                IsometricTensorMerge(self.chi, renormalize_output=self.merge_renormalize_output)
                for _ in range(len(self.merge_levels))
            ]
        )
