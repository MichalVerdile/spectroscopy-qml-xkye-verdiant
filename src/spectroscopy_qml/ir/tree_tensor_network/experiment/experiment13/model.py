"""Experiment 13: QCNN specialist heads on raw IR spectral windows.

Each hard class gets one SpectralWindowQCNN head:
    1. Extract diagnostic spectral window(s) from 1800-point raw spectrum
    2. Learnable linear projection → n_qubits angles
    3. QCNN-style PennyLane circuit  (batched via parameter broadcasting)
    4. Learnable affine → scalar logit

Batching strategy: PennyLane parameter broadcasting.
  angles shape (batch, n_qubits) → inputs[..., wire] has shape (batch,).
  PennyLane treats this as a batched parameter and simulates all samples
  in one vectorised call — no Python loop over samples.
"""

from __future__ import annotations

import math
from typing import Callable

import pennylane as qml
import torch
from torch import Tensor, nn

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment13.spectral_windows import (
    get_window_indices,
    total_window_size,
)

SPEC_LEN = 1800


# ---------------------------------------------------------------------------
# Quantum circuit helpers
# ---------------------------------------------------------------------------

def _two_qubit_block(wires: tuple[int, int], params: Tensor) -> None:
    wi, wj = wires
    qml.RY(params[0], wires=wi)
    qml.RZ(params[1], wires=wi)
    qml.CNOT(wires=[wi, wj])
    qml.RY(params[2], wires=wj)
    qml.RZ(params[3], wires=wj)
    qml.CNOT(wires=[wj, wi])


def _build_batched_qcnn(n_qubits: int) -> Callable:
    """Return a QNode that accepts batched inputs via parameter broadcasting.

    ``inputs`` shape: ``(batch, n_qubits)`` — each ``inputs[..., wire]`` is a
    1-D tensor of length ``batch``.  PennyLane executes all batch elements in
    a single vectorised simulation call instead of a Python loop.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    level1_pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    level2_pairs = [(1, 3), (5, 7)] if n_qubits >= 8 else [(1, 3)]
    final_pair   = [(3, 7)] if n_qubits >= 8 else [(1, 3)]
    readout_wire = n_qubits - 1 if n_qubits >= 8 else 3

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def _circuit(inputs: Tensor, conv1: Tensor, conv2: Tensor, final: Tensor) -> Tensor:
        # inputs: (batch, n_qubits) — use [..., wire] for broadcasting
        for wire in range(n_qubits):
            qml.RY(inputs[..., wire], wires=wire)
        for idx, wires in enumerate(level1_pairs):
            _two_qubit_block(wires, conv1[idx])
        for idx, wires in enumerate(level2_pairs):
            _two_qubit_block(wires, conv2[idx])
        _two_qubit_block(final_pair[0], final[0])
        return qml.expval(qml.PauliZ(readout_wire))

    return _circuit, level1_pairs, level2_pairs


# ---------------------------------------------------------------------------
# Per-class head
# ---------------------------------------------------------------------------

class SpectralWindowQCNN(nn.Module):
    """QCNN head that reads one or more diagnostic spectral windows.

    Uses PennyLane parameter broadcasting for efficient batched execution —
    no Python loop over samples.
    """

    def __init__(
        self,
        class_idx: int,
        n_qubits: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_qubits not in {4, 6, 8}:
            raise ValueError("n_qubits must be one of {4, 6, 8}.")

        self.class_idx = class_idx
        self.n_qubits  = n_qubits

        window_indices = get_window_indices(class_idx)
        self.register_buffer(
            "window_starts",
            torch.tensor([lo for lo, _ in window_indices], dtype=torch.long),
        )
        self.register_buffer(
            "window_ends",
            torch.tensor([hi for _, hi in window_indices], dtype=torch.long),
        )

        in_dim = total_window_size(class_idx)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # Build batched QNode and store circuit weights as nn.Parameters
        _circuit, level1_pairs, level2_pairs = _build_batched_qcnn(n_qubits)
        self._circuit = _circuit
        self.conv1 = nn.Parameter(torch.empty(len(level1_pairs), 4).uniform_(-0.1, 0.1))
        self.conv2 = nn.Parameter(torch.empty(len(level2_pairs), 4).uniform_(-0.1, 0.1))
        self.final = nn.Parameter(torch.empty(1, 4).uniform_(-0.1, 0.1))

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, SPEC_LEN) raw SNV-normalised spectrum
        Returns:
            logits: (batch, 1)
        """
        segments = [x[:, lo:hi] for lo, hi in zip(self.window_starts, self.window_ends)]
        window = torch.cat(segments, dim=-1)         # (batch, in_dim)
        angles = math.pi * self.proj(window)         # (batch, n_qubits)

        # Single batched circuit call — no Python loop
        q_out = self._circuit(angles, self.conv1, self.conv2, self.final)  # (batch,)

        return (self.output_scale * q_out + self.output_bias).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class SpectralWindowEnsemble(nn.Module):
    """One SpectralWindowQCNN per specialist class."""

    def __init__(
        self,
        specialist_indices: list[int],
        n_qubits: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.specialist_indices = list(specialist_indices)
        self.heads = nn.ModuleList([
            SpectralWindowQCNN(idx, n_qubits=n_qubits, dropout=dropout)
            for idx in specialist_indices
        ])

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        logits = torch.cat([head(x) for head in self.heads], dim=-1)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
