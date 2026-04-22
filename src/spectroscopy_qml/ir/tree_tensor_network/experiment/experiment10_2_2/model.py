"""Experiment 10.2.2: Specialist heads with combined TTN + spectral window input.

Per-class input = [TTN 10.2 readout (64-dim)] + [class-specific spectral window]
                → MLP binary head

The TTN backbone is fully frozen. Each specialist head has its own window
extractor that directly reads the raw spectrum at the diagnostic wavenumbers.
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
    """Frozen TTN 10.2 backbone → 64-dim normalised readout."""

    def __init__(self, ttn: TTNIRClassifier10_2) -> None:
        super().__init__()
        self.ttn = ttn
        for p in self.ttn.parameters():
            p.requires_grad_(False)

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
    """MLP head that reads TTN features + raw spectral window for one class.

    Input: concat([ttn_features (64-dim)], [spectral_window (window_dim)])
    """

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
        # Project spectral window to fixed dim before concat
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
        """
        Args:
            ttn_feat: (batch, 64) frozen TTN readout
            x_raw:    (batch, 1800) raw spectrum
        Returns:
            (batch, 1) logit
        """
        segs = [x_raw[:, lo:hi] for lo, hi in zip(self.window_starts, self.window_ends)]
        window = torch.cat(segs, dim=-1)
        combined = torch.cat([ttn_feat, self.window_proj(window)], dim=-1)
        return self.mlp(combined)


class TTN102CombinedEnsemble(nn.Module):
    """Frozen TTN 10.2 + 10 combined specialist heads."""

    def __init__(
        self,
        ttn: TTNIRClassifier10_2,
        specialist_indices: list[int] = SPECIALIST_INDICES,
        hidden_dims: list[int] | None = None,
        window_proj_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.specialist_indices = list(specialist_indices)
        self.backbone = TTN102FeatureExtractor(ttn)
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
