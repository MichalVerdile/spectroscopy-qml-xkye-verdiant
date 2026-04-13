"""
Quantum Machine Learning model for IR spectroscopy functional group classification.

This module implements a hybrid classical-quantum multi-label classifier that fits
into the existing model comparison framework alongside CNN, MPS, and TTN models.

Architecture overview:
    1. Classical encoder  : Linear(1800→256) → BN → ReLU → Linear(256→64) → BN → ReLU
                            → Linear(64→8) → Tanh
    2. Angle encoding     : RY(π·xᵢ) on qubit i  (i = 0..7)
    3. PQC (N_LAYERS=3)   : per layer — RX+RY+RZ on each qubit → ring CNOT
    4. Measurement        : ⟨Z⟩ on all 8 qubits  →  (batch, 8)  in [-1, 1]
    5. Classical head     : Linear(8→64) → ReLU → Dropout(0.3) → Linear(64→37)

Loss: BCEWithLogitsLoss  (model outputs raw logits, no sigmoid).
Gradient method: parameter-shift rule via PennyLane's ``diff_method="parameter-shift"``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
N_QUBITS: int = 8
N_LAYERS: int = 3
N_CLASSES: int = 37
INPUT_DIM: int = 1800
USE_GPU: bool = False  # set True to switch device to lightning.gpu


# ---------------------------------------------------------------------------
# PennyLane device & QNode
# ---------------------------------------------------------------------------
_device_name = "lightning.gpu" if USE_GPU else "lightning.qubit"
_dev = qml.device(_device_name, wires=N_QUBITS)


@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def _circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
    """
    Parametrised quantum circuit for a single sample.

    Args:
        inputs:  1-D tensor of shape (N_QUBITS,) — encoder output, values in [-1, 1].
        weights: Tensor of shape (N_LAYERS, N_QUBITS, 3) — trainable PQC parameters.

    Returns:
        List of N_QUBITS Pauli-Z expectation values.
    """
    # --- angle encoding: RY(π · xᵢ) on qubit i ---
    for i in range(N_QUBITS):
        qml.RY(torch.pi * inputs[i], wires=i)

    # --- PQC: N_LAYERS of single-qubit rotations + ring CNOT ---
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)
            qml.RZ(weights[layer, qubit, 2], wires=qubit)
        # ring CNOT: 0→1→2→…→7→0
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ---------------------------------------------------------------------------
# QMLModel
# ---------------------------------------------------------------------------

class QMLModel(nn.Module):
    """
    Hybrid classical-quantum multi-label classifier for IR functional group prediction.

    The model encodes an SNV-normalised IR spectrum (length 1800) into 8 real values
    via a classical MLP encoder, feeds them into a parametrised quantum circuit using
    angle encoding, measures Pauli-Z expectations on all qubits, and maps the quantum
    output to 37 class logits with a small classical head.

    Input : (batch, 1800) float tensor  — SNV-normalised spectrum
    Output: (batch, 37)   float tensor  — raw logits for BCEWithLogitsLoss
    """

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Classical encoder  1800 → 8  (output in [-1, 1] via Tanh)
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, N_QUBITS),
            nn.Tanh(),
        )

        # ------------------------------------------------------------------
        # 2 + 3 + 4.  Quantum layer  (TorchLayer wraps the QNode)
        # ------------------------------------------------------------------
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # ------------------------------------------------------------------
        # 5. Classical head  8 → 37
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Args:
            x: SNV-normalised IR spectrum of shape (batch_size, 1800).

        Returns:
            Raw logits of shape (batch_size, 37) suitable for BCEWithLogitsLoss.
        """
        # Classical encoding → (batch, 8), values in [-1, 1]
        encoded = self.encoder(x)                           # (batch, 8)

        # Quantum layer: process each sample individually, then stack
        # (PennyLane QNode handles a single sample at a time)
        q_out = torch.stack(
            [self.qlayer(encoded[i]) for i in range(encoded.shape[0])]
        )                                                    # (batch, 8)

        # Classical head → raw logits
        logits = self.head(q_out)                           # (batch, 37)
        return logits


# ---------------------------------------------------------------------------
# Parameter summary (printed when module is executed directly or imported)
# ---------------------------------------------------------------------------

def _param_summary(model: QMLModel) -> None:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    quantum_params = sum(
        p.numel() for p in model.qlayer.parameters() if p.requires_grad
    )
    classical_params = total - quantum_params

    print(f"QMLModel parameter summary")
    print(f"  Total parameters    : {total:,}")
    print(f"  Quantum parameters  : {quantum_params:,}  "
          f"(N_LAYERS={N_LAYERS} × N_QUBITS={N_QUBITS} × 3 = {N_LAYERS*N_QUBITS*3})")
    print(f"  Classical parameters: {classical_params:,}")


if __name__ == "__main__":
    model = QMLModel()
    _param_summary(model)
