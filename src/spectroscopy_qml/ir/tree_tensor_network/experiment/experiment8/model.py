"""
Experiment 8 — Quantum Transfer Learning für IR-Spektren-Klassifikation.

Referenz: Mari et al. (2020) "Transfer learning in hybrid classical-quantum
neural networks" — Quantum 4, 340.

Architektur
-----------

    IR-Spektrum (1800,)
          │
          ▼
    ┌─────────────────────────────────────────────┐
    │  TTN Exp6 Backbone  (eingefroren)           │
    │  SpectralDerivativeFeatureMap               │
    │  SegmentLeafEncoder  (chi=64)               │
    │  RelaxedIsometricMerge ×6                   │
    │  LayerNorm                                  │
    │                                             │
    │  Output: (batch, 64)  ← chi-dim Feature    │
    └─────────────────────────────────────────────┘
          │  eingefroren — keine Gradienten
          ▼
    ┌─────────────────────────────────────────────┐
    │  Klassischer Adapter  (trainierbar)         │
    │  Linear(64 → N_QUBITS) → Tanh → [-1, +1]  │
    └─────────────────────────────────────────────┘
          │  x₀...x₃  (4 Zahlen)
          ▼
    ┌─────────────────────────────────────────────┐
    │  Quantenschaltkreis (4 Qubits, 3 Schichten) │
    │  Data Re-uploading (Pérez-Salinas 2020):    │
    │   pro Schicht:                              │
    │     RY(π·xᵢ) auf Qubit i  ← Re-encode      │
    │     RX RY RZ auf Qubit i  ← trainierbar    │
    │     CNOT Ring 0→1→2→3→0   ← Verschränkung  │
    │                                             │
    │  Messung: ⟨Z₀⟩...⟨Z₃⟩ ∈ [-1, +1]          │
    └─────────────────────────────────────────────┘
          │  (batch, 4)
          ▼
    ┌─────────────────────────────────────────────┐
    │  Klassischer Kopf  (trainierbar)            │
    │  Linear(4 → 32) → GELU → Linear(32 → 37)   │
    └─────────────────────────────────────────────┘
          │
          ▼
    37 Logits (BCEWithLogitsLoss)

Warum 4 Qubits
--------------
4 Qubits = 16-dim Hilbert-Raum. Mit 3 Re-uploading-Schichten und Ring-CNOT
lernt der Schaltkreis nichtlineare Transformationen der 64-dim TTN-Features
ohne Barren-Plateau-Problem (tritt erst ab ~10 Qubits signifikant auf).

Trainierbare Parameter
----------------------
  Adapter:         64×4 + 4 = 260
  Quantenkreis:    3 Schichten × 4 Qubits × 3 Rotationen = 36
  Kopf:            4×32 + 32 + 32×37 + 37 = 1'349
  ─────────────────────────────────────────────
  Total (neu):     1'645  (Backbone ist eingefroren)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import pennylane as qml

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.model import (
    TTNIRClassifier6,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    DEFAULT_SEGMENT_STRIDE,
)

# ── Quantum-Hyperparameter ─────────────────────────────────────────────────────
N_QUBITS:  int  = 4
N_LAYERS:  int  = 3
N_CLASSES: int  = 37
USE_GPU:   bool = False

_dev = qml.device(
    "lightning.gpu" if USE_GPU else "lightning.qubit",
    wires=N_QUBITS,
)


# ── Quantenschaltkreis ─────────────────────────────────────────────────────────

@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def _circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
    """
    4-Qubit Re-uploading Schaltkreis.

    Args:
        inputs:  (N_QUBITS,) — Adapter-Output, Werte in [-1, +1]
        weights: (N_LAYERS, N_QUBITS, 3) — trainierbare Rotationswinkel

    Returns:
        Liste von N_QUBITS Pauli-Z Erwartungswerten in [-1, +1].
    """
    for layer in range(N_LAYERS):
        # Re-uploading: Daten in jeder Schicht neu einbetten
        for i in range(N_QUBITS):
            qml.RY(torch.pi * inputs[i], wires=i)
        # Trainierbare Rotationen
        for i in range(N_QUBITS):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        # Ring-Verschränkung
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ── Backbone-Lader ─────────────────────────────────────────────────────────────

def load_frozen_backbone(
    checkpoint_path: Path,
    device: torch.device,
) -> TTNIRClassifier6:
    """
    Lädt TTNIRClassifier6 aus einem Exp6-Checkpoint, entfernt den
    Output-Head und friert alle verbleibenden Parameter ein.

    Der Output der veränderten Backbone-Forwards-Methode ist der
    chi=64-dimensionale Zustandsvektor nach output_norm — also genau
    die komprimierten TTN-Features vor der ursprünglichen Klassifikation.

    Args:
        checkpoint_path: Pfad zur gespeicherten .pt-Datei.
        device:          Ziel-Device.

    Returns:
        Eingefriertes TTNIRClassifier6-Modell mit nn.Identity als output_head.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Modell mit den Exp6-Defaults bauen (chi=64 ist der dominante Run)
    args = ckpt.get("args", None)
    chi               = getattr(args, "chi",                64)
    segment_window    = getattr(args, "segment_window_size", DEFAULT_SEGMENT_WINDOW_SIZE)
    segment_stride    = getattr(args, "segment_stride",      DEFAULT_SEGMENT_STRIDE)
    segment_mode      = getattr(args, "segment_mode",        "overlap")
    leaf_dropout      = getattr(args, "leaf_dropout",        0.05)
    merge_residual_w  = getattr(args, "merge_residual_weight", 0.1)
    readout_hidden    = getattr(args, "readout_hidden_dim",  None)
    readout_dropout   = getattr(args, "readout_dropout",     0.0)

    backbone = TTNIRClassifier6(
        num_labels=N_CLASSES,
        chi=chi,
        input_dim=1800,
        segment_window_size=segment_window,
        segment_stride=segment_stride,
        segment_mode=segment_mode,
        leaf_dropout=leaf_dropout,
        merge_residual_weight=merge_residual_w,
        readout_hidden_dim=readout_hidden,
        readout_dropout=readout_dropout,
    )

    # Gewichte laden
    backbone.load_state_dict(ckpt["model_state_dict"])

    # Output-Head durch Identity ersetzen → Backbone gibt jetzt (batch, chi) aus
    backbone.output_head = nn.Identity()

    # Alle Parameter einfrieren
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval()

    return backbone.to(device)


# ── Hauptmodell ────────────────────────────────────────────────────────────────

class QuantumTransferModel(nn.Module):
    """
    Quantum Transfer Learning Modell für IR-Spektren-Klassifikation.

    Kombiniert den eingefrorenen TTN-Exp6-Backbone als Feature-Extraktor
    mit einem trainierbaren 4-Qubit-Quantenschaltkreis als Klassifikator.

    Args:
        checkpoint_path: Pfad zur besten Exp6 .pt-Datei.
        device:          Ziel-Device für Backbone und Quantenkreis.
        n_qubits:        Anzahl Qubits (Standard 4).
        n_layers:        Re-uploading Schichten (Standard 3).
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # ── 1. Eingefrierter Backbone ──────────────────────────────────────────
        self.backbone = load_frozen_backbone(checkpoint_path, device)
        backbone_out_dim = self.backbone.chi  # 64

        # ── 2. Klassischer Adapter  64 → N_QUBITS ─────────────────────────────
        self.adapter = nn.Sequential(
            nn.Linear(backbone_out_dim, n_qubits),
            nn.Tanh(),   # Output in [-1, +1] für Winkel-Encoding
        )

        # ── 3. Quantenschaltkreis ─────────────────────────────────────────────
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # ── 4. Klassischer Kopf ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.GELU(),
            nn.Linear(32, N_CLASSES),
        )

    # ── Backbone im Eval-Modus halten ─────────────────────────────────────────

    def train(self, mode: bool = True) -> "QuantumTransferModel":
        """Backbone bleibt immer in eval() — nur Adapter, qlayer, head trainieren."""
        super().train(mode)
        self.backbone.eval()
        return self

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1800) SNV-normalisiertes IR-Spektrum

        Returns:
            (batch, 37) rohe Logits für BCEWithLogitsLoss
        """
        # Backbone (eingefroren) → (batch, 64)
        with torch.no_grad():
            features = self.backbone(x)      # (batch, chi=64)

        # Adapter → (batch, 4), Werte in [-1, +1]
        angles = self.adapter(features)      # (batch, 4)

        # Quantenschaltkreis: pro Sample einmal, dann stapeln
        q_out = torch.stack(
            [self.qlayer(angles[i]) for i in range(angles.shape[0])]
        )                                    # (batch, 4)

        return self.head(q_out)              # (batch, 37)

    # ── Parameter-Übersicht ───────────────────────────────────────────────────

    def param_summary(self) -> None:
        """Gibt trainierbare vs. eingefrorene Parameter aus."""
        trainable   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen      = sum(p.numel() for p in self.backbone.parameters())
        q_params    = sum(p.numel() for p in self.qlayer.parameters() if p.requires_grad)
        c_params    = trainable - q_params

        print("QuantumTransferModel — Parameter")
        print(f"  Eingefroren (TTN Backbone): {frozen:>10,}")
        print(f"  Trainierbar gesamt:         {trainable:>10,}")
        print(f"    davon Quantum ({self.n_layers}×{self.n_qubits}×3): {q_params:>7,}")
        print(f"    davon Klassisch (Adapter+Kopf): {c_params:>4,}")
