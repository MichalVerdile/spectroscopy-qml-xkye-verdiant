"""Experiment 10.2.4: 1D-CNN specialist heads on diagnostic spectral windows.

Per-class input = [TTN 10.2 readout (64-dim)] + [1D-CNN over diagnostic windows]
                → Linear → binary logit

The CNN learns local spectral patterns (peak shapes, shoulders) rather than
projecting the window with a flat MLP. TTN backbone is fully frozen.
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
    """Fully frozen TTN 10.2 backbone → 64-dim normalised readout."""

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


class CNNSpecialistHead(nn.Module):
    """1D-CNN head over diagnostic spectral windows + TTN readout → binary logit.

    Architecture:
        spectrum → extract diagnostic windows → concat segments → (B, 1, win_len)
        → Conv1d(1, cnn_channels, 7) → GELU
        → Conv1d(cnn_channels, cnn_channels*2, 5) → GELU
        → AdaptiveAvgPool1d(pool_size) → flatten → (B, cnn_channels*2*pool_size)
        → concat with ttn_feat (B, 64)
        → Linear → LayerNorm → GELU → Dropout → Linear(1)
    """

    def __init__(
        self,
        class_idx: int,
        ttn_dim: int,
        cnn_channels: int = 32,
        pool_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.class_idx = class_idx

        win_indices = get_window_indices(class_idx)
        self.register_buffer("window_starts", torch.tensor([lo for lo, _ in win_indices], dtype=torch.long))
        self.register_buffer("window_ends",   torch.tensor([hi for _, hi in win_indices], dtype=torch.long))

        self.conv_body = nn.Sequential(
            nn.Conv1d(1, cnn_channels,     kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        cnn_out_dim = cnn_channels * 2 * pool_size

        in_dim = ttn_dim + cnn_out_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, ttn_feat: Tensor, x_raw: Tensor) -> Tensor:
        segs = [x_raw[:, lo:hi] for lo, hi in zip(self.window_starts, self.window_ends)]
        window = torch.cat(segs, dim=-1).unsqueeze(1)       # (B, 1, win_len)
        x = self.conv_body(window)
        # AdaptiveAvgPool1d requires divisible sizes on MPS — run on CPU
        cnn_feat = self.pool(x.cpu()).to(x.device).flatten(1)
        combined = torch.cat([ttn_feat, cnn_feat], dim=-1)
        return self.head(combined)


class TTN102CNNEnsemble(nn.Module):
    """Frozen TTN 10.2 + 10 CNN specialist heads."""

    def __init__(
        self,
        ttn: TTNIRClassifier10_2,
        specialist_indices: list[int] = SPECIALIST_INDICES,
        cnn_channels: int = 32,
        pool_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.specialist_indices = list(specialist_indices)
        self.backbone = TTN102FeatureExtractor(ttn)
        ttn_dim = self.backbone.feature_dim
        self.heads = nn.ModuleList([
            CNNSpecialistHead(
                class_idx=idx,
                ttn_dim=ttn_dim,
                cnn_channels=cnn_channels,
                pool_size=pool_size,
                dropout=dropout,
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
