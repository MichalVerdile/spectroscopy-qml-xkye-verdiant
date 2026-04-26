"""TTN-guided Shpetim VQC specialist model.

Frozen TTN 10.2 backbone provides 64-dim features and per-class logits.
A small CNN compresses the raw spectrum to n_qubits values which are fed
into a data-reuploading VQC.  The combined TTN features + VQC output passes
through a classical head.  The TTN logits for the 10 specialist classes act
as a residual skip connection so the head only needs to learn corrections.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import pennylane as qml
from torch import Tensor

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_5.model import (
    TTN102FeatureExtractor,
    load_ttn102,
)
from spectroscopy_qml.ir.qml.quantum_model_shpetim import TTN10_2_HARD_CLASS_INDICES


class _SpectralMini(nn.Module):
    """Light 1-D CNN: [B,1,L] → [B, out_features].  AdaptiveAvgPool1d(1) is
    always MPS-safe and avoids the divisibility constraint."""

    def __init__(self, out_features: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [B, 64, 1] — always divisible
            nn.Flatten(),             # [B, 64]
        )
        self.proj = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.backbone(x))


class TTNShpetimEnsemble(nn.Module):
    """Frozen TTN 10.2 + data-reuploading VQC for the 10 hardest TTN classes."""

    def __init__(
        self,
        ttn,
        specialist_indices: tuple[int, ...] = TTN10_2_HARD_CLASS_INDICES,
        *,
        n_qubits: int = 4,
        n_layers: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.specialist_indices = list(specialist_indices)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.backbone = TTN102FeatureExtractor(ttn)
        ttn_dim = self.backbone.feature_dim  # 64

        self.extractor = _SpectralMini(out_features=n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="backprop",
        )
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Head: TTN features + VQC output → specialist logits
        head_in = ttn_dim + n_qubits
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(specialist_indices)),
        )

    def _circuit(self, features: Tensor, weights: Tensor) -> list[Tensor]:
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(features[:, q], wires=q)
            for q in range(self.n_qubits):
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, 1800] raw spectrum → logits [B, n_specialist]."""
        # Frozen TTN branch
        ttn_feat, ttn_logits = self.backbone(x)                          # [B,64], [B,37]
        ttn_spec = ttn_logits[:, self.specialist_indices]                 # [B,10]

        # VQC branch (always on CPU)
        feat = self.extractor(x.unsqueeze(1))                            # [B, n_qubits]
        feat = torch.tanh(feat) * math.pi
        q_out = torch.stack(
            self.qnode(feat.cpu(), self.q_weights.cpu()), dim=1
        ).float().to(x.device)                                           # [B, n_qubits]

        # Head + TTN skip connection
        combined = torch.cat([ttn_feat, q_out], dim=-1)                  # [B, 64+n_qubits]
        return self.head(combined) + ttn_spec                            # residual


def load_ttn_shpetim_ensemble(
    ttn_checkpoint: Path,
    ttn_config: dict,
    device: torch.device,
    specialist_indices: tuple[int, ...] = TTN10_2_HARD_CLASS_INDICES,
    n_qubits: int = 4,
    n_layers: int = 12,
    hidden_dim: int = 64,
    dropout: float = 0.0,
) -> TTNShpetimEnsemble:
    ttn = load_ttn102(ttn_checkpoint, ttn_config, device)
    return TTNShpetimEnsemble(
        ttn=ttn,
        specialist_indices=specialist_indices,
        n_qubits=n_qubits,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
