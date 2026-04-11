"""
Data Re-uploading QML model for IR spectroscopy functional group classification.

Based on: Pérez-Salinas et al. (2020) "Data re-uploading for a universal
quantum classifier" — Quantum 4, 226.

Key idea: instead of encoding the input only once before the PQC, the same
input is re-encoded at the start of EVERY layer.  This makes the quantum
circuit exponentially more expressive without adding qubits, because the
circuit can learn arbitrary Fourier-like decompositions of the input.

Architecture:
    1. Thin encoder   : Linear(1800→64) → BN → ReLU → Dropout(0.2)
                        → Linear(64→N_QUBITS) → Tanh
                        (~116 k params — small enough to avoid overfitting)
    2. Re-uploading QNN (N_LAYERS=5):
         for each layer ℓ:
           RY(π·xᵢ)  on qubit i  ← re-encode input every layer
           RX(θ) RY(φ) RZ(λ)     on qubit i  ← trainable
           ring CNOT  0→1→…→7→0  ← entanglement
    3. Measurement: ⟨Z⟩ on all N_QUBITS qubits → (batch, N_QUBITS)
    4. Head: Linear(N_QUBITS→32) → ReLU → Dropout(0.2) → Linear(32→37)

Loss : BCEWithLogitsLoss (raw logits, no sigmoid in model)
Grad : adjoint differentiation (lightning.qubit)

Input : (batch, 1800) float32 — SNV-normalised IR spectrum
Output: (batch, 37)   float32 — raw logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS:  int  = 8
N_LAYERS:  int  = 5      # more layers → richer Fourier decomposition
N_CLASSES: int  = 37
INPUT_DIM: int  = 1800
USE_GPU:   bool = False

# ── PennyLane device ──────────────────────────────────────────────────────────
_dev = qml.device("lightning.gpu" if USE_GPU else "lightning.qubit",
                  wires=N_QUBITS)


@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def _circuit_reupload(inputs: torch.Tensor, weights: torch.Tensor) -> list:
    """
    Data re-uploading quantum circuit for a single sample.

    At every layer the full input is re-encoded via RY(π·xᵢ), followed by
    trainable RX/RY/RZ rotations and a ring of CNOT gates.

    Args:
        inputs:  shape (N_QUBITS,) — thin-encoder output in [-1, 1]
        weights: shape (N_LAYERS, N_QUBITS, 3) — trainable PQC angles

    Returns:
        List of N_QUBITS Pauli-Z expectation values in [-1, 1].
    """
    for layer in range(N_LAYERS):
        # ── Re-upload input ──────────────────────────────────────────────────
        for i in range(N_QUBITS):
            qml.RY(torch.pi * inputs[i], wires=i)

        # ── Trainable rotations ──────────────────────────────────────────────
        for i in range(N_QUBITS):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)

        # ── Ring entanglement ────────────────────────────────────────────────
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ── Model ─────────────────────────────────────────────────────────────────────

class QMLReuploadModel(nn.Module):
    """
    Hybrid classical-quantum classifier using data re-uploading.

    The thin classical encoder avoids overfitting; the quantum circuit with
    re-uploading provides a richer feature representation than a single-encoding
    PQC of the same depth.

    Input : (batch, 1800) float32 — SNV-normalised spectrum
    Output: (batch, 37)   float32 — raw logits for BCEWithLogitsLoss
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Thin encoder  1800 → N_QUBITS  (~116 k params) ───────────────────
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_QUBITS),
            nn.Tanh(),                  # output in [-1, 1] for angle encoding
        )

        # ── Re-uploading quantum layer ────────────────────────────────────────
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(_circuit_reupload, weight_shapes)

        # ── Classical head  N_QUBITS → 37 ────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 1800) SNV-normalised spectrum.

        Returns:
            (batch, 37) raw logits for BCEWithLogitsLoss.
        """
        encoded = self.encoder(x)                                   # (B, 8)
        q_out = torch.stack(
            [self.qlayer(encoded[i]) for i in range(encoded.shape[0])]
        )                                                            # (B, 8)
        return self.head(q_out)                                      # (B, 37)


# ── Parameter summary ─────────────────────────────────────────────────────────

def _param_summary(model: QMLReuploadModel) -> None:
    total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_par   = sum(p.numel() for p in model.qlayer.parameters() if p.requires_grad)
    cl_par  = total - q_par
    print("QMLReuploadModel parameter summary")
    print(f"  Total parameters    : {total:,}")
    print(f"  Quantum parameters  : {q_par}  "
          f"(N_LAYERS={N_LAYERS} × N_QUBITS={N_QUBITS} × 3 = {N_LAYERS*N_QUBITS*3})")
    print(f"  Classical parameters: {cl_par:,}")


if __name__ == "__main__":
    _param_summary(QMLReuploadModel())
