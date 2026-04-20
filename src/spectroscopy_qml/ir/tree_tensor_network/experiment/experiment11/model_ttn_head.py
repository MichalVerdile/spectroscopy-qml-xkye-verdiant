"""TTN Head model for Experiment 11 (Hybrid CNN-TTN).

Takes frozen CNN feature vectors (1574-dim) as input instead of raw spectra.
Projects them into a structured segment representation, then applies the same
TTN merge-tree used in Experiment 10.

Architecture:
    CNN features (1574)
        → Linear projection (1574 → num_segments × segment_dim)
        → reshape: (batch, num_segments, segment_dim)
        → L2-normalise each segment state
        → TTN merge tree (FastDirectIsometricMerge, log2 levels)
        → LayerNorm → Linear → num_specialist_classes
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10.model import (
    FastDirectIsometricMerge,
)

# Default CNN feature dimension (last hidden Dense layer in Jung CNN)
CNN_FEATURE_DIM = 1574


class TTNHead(nn.Module):
    """TTN classifier head operating on frozen CNN feature vectors.

    Args:
        cnn_feature_dim:      Dimensionality of the input CNN feature vector.
        num_specialist_classes: Number of output classes (specialist subset).
        chi:                  TTN bond dimension (merge output dimension).
        num_segments:         Number of virtual segments to project features into.
                              Must be a power of 2 for a balanced tree.
        segment_dim:          Dimension of each segment state after projection.
                              If None, defaults to chi.
        merge_mode:           "relaxed" or "strict" isometric merge.
        merge_residual_weight: Residual connection weight in merge layers.
        merge_renormalize_output: Whether to renormalise after each merge.
        segment_normalize:    L2-normalise each segment state before merging.
    """

    def __init__(
        self,
        cnn_feature_dim: int = CNN_FEATURE_DIM,
        num_specialist_classes: int = 5,
        chi: int = 64,
        num_segments: int = 16,
        segment_dim: int | None = None,
        merge_mode: str = "relaxed",
        merge_residual_weight: float = 0.1,
        merge_renormalize_output: bool = True,
        segment_normalize: bool = True,
    ) -> None:
        super().__init__()

        if num_segments < 1:
            raise ValueError("num_segments must be >= 1.")
        if not (num_segments & (num_segments - 1) == 0):
            raise ValueError("num_segments must be a power of 2 for a balanced TTN.")

        self.cnn_feature_dim = int(cnn_feature_dim)
        self.num_specialist_classes = int(num_specialist_classes)
        self.chi = int(chi)
        self.num_segments = int(num_segments)
        self.segment_dim = int(segment_dim) if segment_dim is not None else int(chi)
        self.segment_normalize = bool(segment_normalize)

        # Project flat CNN features → structured segment states
        proj_out_dim = self.num_segments * self.segment_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self.cnn_feature_dim, proj_out_dim),
            nn.LayerNorm(proj_out_dim),
            nn.GELU(),
        )

        # TTN merge levels: log2(num_segments) levels
        num_levels = int(math.log2(self.num_segments))
        merge_input_dims = [self.segment_dim] + [self.chi] * max(0, num_levels - 1)
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

        self.output_norm = nn.LayerNorm(self.chi)
        self.output_head = nn.Linear(self.chi, self.num_specialist_classes)

    def forward(self, cnn_features: Tensor, apply_sigmoid: bool = False) -> Tensor:
        """
        Args:
            cnn_features: (batch_size, cnn_feature_dim)
        Returns:
            logits or probs: (batch_size, num_specialist_classes)
        """
        if cnn_features.ndim != 2:
            raise ValueError(
                f"Expected (batch, {self.cnn_feature_dim}), got {tuple(cnn_features.shape)}."
            )

        # Project and reshape to segment states
        projected = self.input_proj(cnn_features)
        node_states = projected.view(-1, self.num_segments, self.segment_dim)

        if self.segment_normalize:
            node_states = F.normalize(node_states, dim=-1, eps=1e-8)

        # TTN merge tree
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
