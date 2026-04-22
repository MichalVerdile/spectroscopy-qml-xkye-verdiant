"""Experiment 10.2.3: Specialist heads with partial TTN fine-tuning.

Same architecture as 10.2.2 (TTN readout + spectral window → MLP head) but
the top N TTN merge levels and output_norm are unfrozen and trained with a
small backbone LR alongside the specialist heads.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (
    TTNIRClassifier10_2,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_2.spectral_windows import (
    get_window_indices,
    total_window_size,
    SPECTRAL_WINDOWS,
)

SPECIALIST_INDICES = [1, 13, 14, 18, 19, 21, 24, 28, 33, 35]


class TTN102FeatureExtractor(nn.Module):
    """TTN 10.2 backbone with optional partial unfreezing of top merge levels."""

    def __init__(self, ttn: TTNIRClassifier10_2, finetune_layers: int = 1) -> None:
        super().__init__()
        self.ttn = ttn
        # Freeze everything first
        for p in self.ttn.parameters():
            p.requires_grad_(False)
        # Unfreeze top N merge levels + output_norm
        if finetune_layers > 0:
            for layer in self.ttn.merge_levels[-finetune_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            for p in self.ttn.output_norm.parameters():
                p.requires_grad_(True)

    @property
    def feature_dim(self) -> int:
        return self.ttn.output_norm.normalized_shape[0]

    def forward(self, x: Tensor) -> Tensor:
        ttn = self.ttn
        spectral_positions = torch.arange(ttn.input_dim, device=x.device)
        pos_emb = ttn.input_position_embedding(spectral_positions).squeeze(-1)
        feature_sequence = ttn.feature_map(x)
        raw_with_pos = x + pos_emb.unsqueeze(0)
        scale = raw_with_pos.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        feature_sequence = torch.cat(
            [(raw_with_pos / scale).unsqueeze(-1), feature_sequence[:, :, 1:]], dim=-1
        )
        segmented = ttn._segment_feature_sequence(feature_sequence)
        node_states = ttn._segment_states(segmented)
        level_index = 0
        while node_states.size(1) > 1:
            merge = ttn.merge_levels[level_index]
            n = node_states.size(1)
            p = n // 2
            merged = merge(node_states[:, :2*p:2], node_states[:, 1:2*p:2])
            node_states = torch.cat((merged, node_states[:, -1:]), 1) if n % 2 else merged
            level_index += 1
        return ttn.output_norm(node_states[:, 0, :])


class CombinedSpecialistHead(nn.Module):
    """MLP head: concat(TTN readout, spectral window projection) → binary logit."""

    def __init__(
        self,
        class_idx: int,
        ttn_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        window_proj_dim: int = 32,
    ) -> None:
        super().__init__()
        self.class_idx = class_idx

        win_indices = get_window_indices(class_idx)
        self.register_buffer("window_starts", torch.tensor([lo for lo, _ in win_indices], dtype=torch.long))
        self.register_buffer("window_ends",   torch.tensor([hi for _, hi in win_indices], dtype=torch.long))

        win_dim = total_window_size(class_idx)
        self.window_proj = nn.Sequential(
            nn.Linear(win_dim, window_proj_dim),
            nn.LayerNorm(window_proj_dim),
            nn.GELU(),
        )

        in_dim = ttn_dim + window_proj_dim
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, ttn_feat: Tensor, x_raw: Tensor) -> Tensor:
        segs = [x_raw[:, lo:hi] for lo, hi in zip(self.window_starts, self.window_ends)]
        window = torch.cat(segs, dim=-1)
        combined = torch.cat([ttn_feat, self.window_proj(window)], dim=-1)
        return self.mlp(combined)


class TTN102CombinedEnsemble(nn.Module):
    """TTN 10.2 (partially trainable) + 10 specialist heads."""

    def __init__(
        self,
        ttn: TTNIRClassifier10_2,
        specialist_indices: list[int] = SPECIALIST_INDICES,
        hidden_dims: list[int] | None = None,
        window_proj_dim: int = 32,
        dropout: float = 0.2,
        finetune_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.specialist_indices = list(specialist_indices)
        self.backbone = TTN102FeatureExtractor(ttn, finetune_layers=finetune_layers)
        ttn_dim = self.backbone.feature_dim
        self.heads = nn.ModuleList([
            CombinedSpecialistHead(
                class_idx=idx,
                ttn_dim=ttn_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                window_proj_dim=window_proj_dim,
            )
            for idx in specialist_indices
        ])

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        ttn_feat = self.backbone(x)
        logits = torch.cat([h(ttn_feat, x) for h in self.heads], dim=-1)
        return torch.sigmoid(logits) if apply_sigmoid else logits


def load_ttn102(checkpoint_path: Path, config: dict, device: torch.device) -> TTNIRClassifier10_2:
    model = TTNIRClassifier10_2(
        input_dim=config["input_dim"], num_labels=config["num_labels"],
        chi=config["chi"], segment_window_size=config["segment_window_size"],
        segment_stride=config["segment_stride"], segment_mode=config["segment_mode"],
        segment_state_normalize=config["segment_state_normalize"],
        merge_mode=config["merge_mode"], merge_residual_weight=config["merge_residual_weight"],
        merge_renormalize_output=config["merge_renormalize_output"],
        lorentz_gamma=config["lorentz_gamma"],
        lorentz_kernel_half_width=config["lorentz_kernel_half_width"],
        lorentz_norm_mode=config["lorentz_norm_mode"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)
