"""Experiment 10.2.1: Specialist MLP heads on frozen TTN 10.2 features.

Architecture:
    Raw spectrum (1800)
        → Frozen TTN 10.2 backbone  →  64-dim readout vector
        → 10 independent MLP Binary Heads (one per hard class)

The TTN backbone is fully frozen. Only the specialist heads are trained.
Specialist classes: Ketone(24), Sulfide(28), Enamine(13), Imine(21),
                    Thioamide(35), Hydrazone(19), Sulfoxide(33),
                    Acyl halide(1), Hydrazine(18), Enol(14)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (
    TTNIRClassifier10_2,
)

SPECIALIST_INDICES = [1, 13, 14, 18, 19, 21, 24, 28, 33, 35]


class TTN102FeatureExtractor(nn.Module):
    """Frozen TTN 10.2 backbone that returns the 64-dim readout vector."""

    def __init__(self, ttn: TTNIRClassifier10_2) -> None:
        super().__init__()
        self.ttn = ttn
        for p in self.ttn.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.ttn.output_norm.normalized_shape[0]

    def forward(self, x: Tensor) -> Tensor:
        """Return normalised TTN readout vector (batch, feature_dim)."""
        ttn = self.ttn
        spectral_positions = torch.arange(ttn.input_dim, device=x.device)
        pos_emb = ttn.input_position_embedding(spectral_positions).squeeze(-1)

        feature_sequence = ttn.feature_map(x)
        raw_with_pos = x + pos_emb.unsqueeze(0)
        scale = raw_with_pos.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        feature_sequence = torch.cat(
            [(raw_with_pos / scale).unsqueeze(-1), feature_sequence[:, :, 1:]], dim=-1
        )
        segmented_features = ttn._segment_feature_sequence(feature_sequence)
        node_states = ttn._segment_states(segmented_features)

        level_index = 0
        while node_states.size(1) > 1:
            merge = ttn.merge_levels[level_index]
            num_nodes = node_states.size(1)
            paired_nodes = num_nodes // 2
            left  = node_states[:, : 2 * paired_nodes : 2, :]
            right = node_states[:, 1 : 2 * paired_nodes : 2, :]
            merged = merge(left, right)
            if num_nodes % 2 == 1:
                carry = node_states[:, -1:, :]
                node_states = torch.cat((merged, carry), dim=1)
            else:
                node_states = merged
            level_index += 1

        readout = node_states[:, 0, :]
        return ttn.output_norm(readout)  # (batch, feature_dim)


class MLPBinaryHead(nn.Module):
    """Small MLP binary classifier for one specialist class."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TTN102SpecialistEnsemble(nn.Module):
    """Frozen TTN 10.2 + 10 trainable MLP specialist heads.

    Args:
        ttn:              Loaded TTNIRClassifier10_2 instance.
        specialist_indices: Global class indices (default: the 10 hard classes).
        hidden_dim:       Hidden size of each MLP head.
        dropout:          Dropout rate in MLP heads.
    """

    def __init__(
        self,
        ttn: TTNIRClassifier10_2,
        specialist_indices: list[int] = SPECIALIST_INDICES,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.specialist_indices = list(specialist_indices)
        self.backbone = TTN102FeatureExtractor(ttn)
        feat_dim = self.backbone.feature_dim
        self.heads = nn.ModuleList([
            MLPBinaryHead(feat_dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in specialist_indices
        ])

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        """
        Args:
            x: (batch, 1800) raw SNV-normalised spectrum
        Returns:
            (batch, num_specialist_classes) logits or probabilities
        """
        features = self.backbone(x)                              # (batch, 64)
        logits = torch.cat([h(features) for h in self.heads], dim=-1)  # (batch, 10)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


def load_ttn102(checkpoint_path: Path, config: dict, device: torch.device) -> TTNIRClassifier10_2:
    """Load a trained TTNIRClassifier10_2 from a checkpoint."""
    model = TTNIRClassifier10_2(
        input_dim=config["input_dim"],
        num_labels=config["num_labels"],
        chi=config["chi"],
        segment_window_size=config["segment_window_size"],
        segment_stride=config["segment_stride"],
        segment_mode=config["segment_mode"],
        segment_state_normalize=config["segment_state_normalize"],
        merge_mode=config["merge_mode"],
        merge_residual_weight=config["merge_residual_weight"],
        merge_renormalize_output=config["merge_renormalize_output"],
        lorentz_gamma=config["lorentz_gamma"],
        lorentz_kernel_half_width=config["lorentz_kernel_half_width"],
        lorentz_norm_mode=config["lorentz_norm_mode"],
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
