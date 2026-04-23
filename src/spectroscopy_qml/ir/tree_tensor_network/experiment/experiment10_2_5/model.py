"""Experiment 10.2.5: frozen TTN 10.2 + quanvolutional specialist heads.

Each specialist head is trained only for one TTN-weak functional group.  The
head reads the frozen 10.2 TTN readout plus class-specific diagnostic IR
windows.  The window branch starts with the fixed 1-D quanvolutional layer from
``spectroscopy_qml.ir.qml.model_quanv1d`` and then uses a small trainable
classical readout.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from spectroscopy_qml.ir.qml.model_quanv1d import Quanvolution1D
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (
    TTNIRClassifier10_2,
)

SPEC_LEN = 1800
WN_MIN = 400
WN_MAX = 4000

# Weak TTN groups from the supplied error table, roughly F1_TTN < 0.60.
WEAK_TTN_SPECIALIST_INDICES = [
    0,   # Acid anhydride
    1,   # Acyl halide
    3,   # Aldehyde
    5,   # Alkene
    10,  # Azo compound
    13,  # Enamine
    14,  # Enol
    18,  # Hydrazine
    19,  # Hydrazone
    21,  # Imine
    24,  # Ketone
    27,  # Phosphine
    28,  # Sulfide
    33,  # Sulfoxide
    34,  # Thial
    35,  # Thioamide
]

SPECIALIST_INDICES = WEAK_TTN_SPECIALIST_INDICES


def wn_to_idx(wn: float) -> int:
    step = (WN_MAX - WN_MIN) / (SPEC_LEN - 1)
    return int(round((wn - WN_MIN) / step))


# class_index -> diagnostic windows in cm^-1.  Ranges follow the table where
# available and are widened slightly so the quanvolution sees peak shape.
SPECTRAL_WINDOWS: dict[int, list[tuple[int, int]]] = {
    0: [(1740, 1860)],                 # Acid anhydride: double C=O
    1: [(1770, 1830)],                 # Acyl halide: C=O
    3: [(1700, 1740), (2700, 2840)],   # Aldehyde: C=O + aldehydic C-H
    5: [(1610, 1690)],                 # Alkene: C=C
    10: [(1400, 1500)],                # Azo compound: N=N
    13: [(1560, 1680), (1000, 1200)],  # Enamine: C=C-N + C-N
    14: [(1580, 1660), (3100, 3600)],  # Enol: C=C + O-H
    18: [(3100, 3500), (1000, 1200)],  # Hydrazine: N-H + N-N
    19: [(1580, 1700), (3100, 3400)],  # Hydrazone: C=N + N-H
    21: [(1640, 1660)],                # Imine: C=N
    24: [(1700, 1750)],                # Ketone: C=O
    27: [(2300, 2400)],                # Phosphine: P-H
    28: [(580, 720)],                  # Sulfide: C-S
    33: [(980, 1090)],                 # Sulfoxide: S=O
    34: [(1600, 1700)],                # Thial: C=S
    35: [(1050, 1200), (3100, 3400)],  # Thioamide: C=S + N-H
}


def get_window_indices(class_idx: int) -> list[tuple[int, int]]:
    if class_idx not in SPECTRAL_WINDOWS:
        raise KeyError(f"No diagnostic windows configured for class index {class_idx}.")
    indices: list[tuple[int, int]] = []
    for lo, hi in SPECTRAL_WINDOWS[class_idx]:
        start = max(0, min(SPEC_LEN, wn_to_idx(lo)))
        end = max(0, min(SPEC_LEN, wn_to_idx(hi)))
        if end <= start:
            end = min(SPEC_LEN, start + 1)
        indices.append((start, end))
    return indices


def total_window_size(class_idx: int) -> int:
    return sum(hi - lo for lo, hi in get_window_indices(class_idx))


class TTN102FeatureExtractor(nn.Module):
    """Fully frozen TTN 10.2 backbone -> normalized readout."""

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
            merged = merge(node_states[:, : 2 * p : 2], node_states[:, 1 : 2 * p : 2])
            node_states = torch.cat((merged, node_states[:, -1:]), 1) if n % 2 else merged
            level_index += 1
        return ttn.output_norm(node_states[:, 0, :])


class QuanvolutionalSpecialistHead(nn.Module):
    """Per-class quanvolutional diagnostic-window head."""

    def __init__(
        self,
        class_idx: int,
        ttn_dim: int,
        *,
        patch_size: int = 4,
        stride: int = 2,
        n_filters: int = 16,
        quanv_seed: int = 42,
        conv_channels: int = 32,
        pool_size: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.class_idx = int(class_idx)

        win_indices = get_window_indices(class_idx)
        self.register_buffer(
            "window_starts", torch.tensor([lo for lo, _ in win_indices], dtype=torch.long)
        )
        self.register_buffer(
            "window_ends", torch.tensor([hi for _, hi in win_indices], dtype=torch.long)
        )

        win_dim = total_window_size(class_idx)
        if patch_size > win_dim:
            raise ValueError(f"patch_size={patch_size} is larger than window size {win_dim}.")

        self.quanv = Quanvolution1D(
            input_dim=win_dim,
            patch_size=patch_size,
            stride=stride,
            n_filters=n_filters,
            seed=quanv_seed + class_idx,
            decode="count_normalized",
        )
        self.window_features = nn.Sequential(
            nn.Conv1d(n_filters, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(pool_size),
            nn.Flatten(),
        )
        in_dim = ttn_dim + conv_channels * pool_size
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ttn_feat: Tensor, x_raw: Tensor) -> Tensor:
        segs = [
            x_raw[:, int(lo.item()) : int(hi.item())]
            for lo, hi in zip(self.window_starts, self.window_ends)
        ]
        window = torch.cat(segs, dim=-1)
        quanv_features = self.quanv(window)
        window_features = self.window_features(quanv_features)
        combined = torch.cat([ttn_feat, window_features], dim=-1)
        return self.head(combined)


class TTN102QuanvEnsemble(nn.Module):
    """Frozen TTN 10.2 + quanvolutional specialist heads."""

    def __init__(
        self,
        ttn: TTNIRClassifier10_2,
        specialist_indices: list[int] = SPECIALIST_INDICES,
        *,
        patch_size: int = 4,
        stride: int = 2,
        n_filters: int = 16,
        quanv_seed: int = 42,
        conv_channels: int = 32,
        pool_size: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.specialist_indices = list(specialist_indices)
        self.backbone = TTN102FeatureExtractor(ttn)
        ttn_dim = self.backbone.feature_dim
        self.heads = nn.ModuleList(
            [
                QuanvolutionalSpecialistHead(
                    class_idx=idx,
                    ttn_dim=ttn_dim,
                    patch_size=patch_size,
                    stride=stride,
                    n_filters=n_filters,
                    quanv_seed=quanv_seed,
                    conv_channels=conv_channels,
                    pool_size=pool_size,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for idx in specialist_indices
            ]
        )

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        ttn_feat = self.backbone(x)
        logits = torch.cat([h(ttn_feat, x) for h in self.heads], dim=-1)
        return torch.sigmoid(logits) if apply_sigmoid else logits


def load_ttn102(checkpoint_path: Path, config: dict, device: torch.device) -> TTNIRClassifier10_2:
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)
