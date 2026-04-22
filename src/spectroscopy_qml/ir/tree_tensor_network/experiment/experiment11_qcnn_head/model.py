"""Specialist head variants on frozen CNN features.

This module supports three directly comparable head types for the hard-class
specialist setup on top of the 1574-dim frozen CNN features:

1. ``linear``: one linear one-vs-rest head per specialist class.
2. ``mlp``: a small MLP one-vs-rest head per specialist class.
3. ``qcnn``: a compact per-class QCNN-style PennyLane head.

All variants expose the same batched interface:
    (batch, 1574) -> (batch, num_specialist_classes)
"""

from __future__ import annotations

import math
from typing import Callable

import pennylane as qml
import torch
from torch import Tensor, nn

CNN_FEATURE_DIM = 1574


def _two_qubit_block(wires: tuple[int, int], params: Tensor) -> None:
    """Small hardware-efficient 2-qubit block."""
    wi, wj = wires
    qml.RY(params[0], wires=wi)
    qml.RZ(params[1], wires=wi)
    qml.CNOT(wires=[wi, wj])
    qml.RY(params[2], wires=wj)
    qml.RZ(params[3], wires=wj)
    qml.CNOT(wires=[wj, wi])


def _build_qcnn_torch_layer(n_qubits: int) -> qml.qnn.TorchLayer:
    """Return a compact QCNN-style TorchLayer with one scalar measurement."""
    dev = qml.device("default.qubit", wires=n_qubits)

    level1_pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    level2_pairs = [(1, 3), (5, 7)] if n_qubits >= 8 else [(1, 3)]
    final_pair = [(3, 7)] if n_qubits >= 8 else [(1, 3)]
    readout_wire = n_qubits - 1 if n_qubits >= 8 else 3

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def _circuit(
        inputs: Tensor,
        conv1: Tensor,
        conv2: Tensor,
        final: Tensor,
    ) -> Tensor:
        for wire in range(n_qubits):
            qml.RY(inputs[wire], wires=wire)

        for idx, wires in enumerate(level1_pairs):
            _two_qubit_block(wires, conv1[idx])

        for idx, wires in enumerate(level2_pairs):
            _two_qubit_block(wires, conv2[idx])

        _two_qubit_block(final_pair[0], final[0])
        return qml.expval(qml.PauliZ(readout_wire))

    weight_shapes = {
        "conv1": (len(level1_pairs), 4),
        "conv2": (len(level2_pairs), 4),
        "final": (1, 4),
    }
    return qml.qnn.TorchLayer(_circuit, weight_shapes)


class LinearBinaryHead(nn.Module):
    """One-vs-rest linear classifier for a single specialist class."""

    def __init__(self, input_dim: int = CNN_FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MLPBinaryHead(nn.Module):
    """One-vs-rest MLP classifier for a single specialist class.

    Supports multiple hidden layers via ``hidden_dims`` list.
    E.g. hidden_dims=[512, 128] → input→512→128→1.
    """

    def __init__(
        self,
        input_dim: int = CNN_FEATURE_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        dims = hidden_dims if hidden_dims is not None else [hidden_dim]
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class QCNNBinaryHead(nn.Module):
    """Per-class QCNN head on frozen CNN features.

    The classical front-end maps the 1574-dim feature vector to ``n_qubits``
    normalized angles. The quantum back-end is a small QCNN-style circuit with
    two local convolution levels and a final merge block. A learned affine map
    turns the scalar expectation value into a logit.
    """

    def __init__(
        self,
        input_dim: int = CNN_FEATURE_DIM,
        n_qubits: int = 8,
        projection_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_qubits not in {4, 6, 8}:
            raise ValueError("n_qubits must currently be one of {4, 6, 8}.")

        self.n_qubits = int(n_qubits)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, projection_hidden_dim),
            nn.LayerNorm(projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, self.n_qubits),
            nn.LayerNorm(self.n_qubits),
            nn.Tanh(),
        )
        self.qlayer = _build_qcnn_torch_layer(self.n_qubits)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        angles = math.pi * self.input_proj(x)
        q_values = torch.stack([self.qlayer(angles[i]) for i in range(angles.shape[0])], dim=0)
        return (self.output_scale * q_values + self.output_bias).unsqueeze(-1)


def _head_factory(
    head_type: str,
    *,
    input_dim: int,
    mlp_hidden_dim: int,
    mlp_hidden_dims: list[int] | None = None,
    qcnn_qubits: int,
    qcnn_projection_hidden_dim: int,
    dropout: float,
) -> Callable[[], nn.Module]:
    if head_type == "linear":
        return lambda: LinearBinaryHead(input_dim=input_dim)
    if head_type == "mlp":
        return lambda: MLPBinaryHead(
            input_dim=input_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout,
        )
    if head_type == "qcnn":
        return lambda: QCNNBinaryHead(
            input_dim=input_dim,
            n_qubits=qcnn_qubits,
            projection_hidden_dim=qcnn_projection_hidden_dim,
            dropout=dropout,
        )
    raise ValueError("head_type must be one of {'linear', 'mlp', 'qcnn'}.")


class SpecialistHeadEnsemble(nn.Module):
    """Multi-output specialist model with one binary head per hard class."""

    def __init__(
        self,
        num_specialist_classes: int,
        *,
        head_type: str = "qcnn",
        input_dim: int = CNN_FEATURE_DIM,
        mlp_hidden_dim: int = 128,
        mlp_hidden_dims: list[int] | None = None,
        qcnn_qubits: int = 8,
        qcnn_projection_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_specialist_classes <= 0:
            raise ValueError("num_specialist_classes must be positive.")

        factory = _head_factory(
            head_type,
            input_dim=input_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            qcnn_qubits=qcnn_qubits,
            qcnn_projection_hidden_dim=qcnn_projection_hidden_dim,
            dropout=dropout,
        )
        self.head_type = head_type
        self.input_dim = int(input_dim)
        self.num_specialist_classes = int(num_specialist_classes)
        self.heads = nn.ModuleList([factory() for _ in range(self.num_specialist_classes)])

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}."
            )
        logits = torch.cat([head(x) for head in self.heads], dim=-1)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
