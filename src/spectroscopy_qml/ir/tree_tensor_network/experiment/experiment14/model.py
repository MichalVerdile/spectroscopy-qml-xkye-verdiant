"""Experiment 14 models for reduction-before-quantum ablations.

This experiment compares four variants on the same task:

1. Raw subsampling -> shared quantum head
2. PCA -> shared quantum head
3. TN compression -> shared quantum head
4. TN compression -> classical control head

The shared quantum head is identical across (1)-(3).  Only the compression
stage changes, which makes the ablation directly interpretable.
"""

from __future__ import annotations

import math

import pennylane as qml
import torch
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment12.model import (
    Experiment12FeatureExtractor,
    HierarchicalTNCompressor,
)


class Experiment14FeatureExtractor(nn.Module):
    """Shared sequence extractor for Experiment 14.

    ``feature_source="raw"`` keeps the SNV-normalized spectrum as a single
    channel sequence.  Other values reuse the engineered feature maps from
    Experiment 12.
    """

    def __init__(
        self,
        feature_source: str = "raw",
        *,
        include_raw_channel: bool = False,
        sg_window_length: int = 11,
        sg_polyorder: int = 3,
        voigt_gamma_l: float = 3.0,
        voigt_gamma_g: float = 2.0,
        voigt_eta: float = 0.5,
        voigt_kernel_half_width: int = 20,
    ) -> None:
        super().__init__()
        self.feature_source = feature_source
        if feature_source == "raw":
            self.feature_map = None
            self.output_channels = 1
        else:
            self.feature_map = Experiment12FeatureExtractor(
                feature_source=feature_source,
                include_raw_channel=include_raw_channel,
                sg_window_length=sg_window_length,
                sg_polyorder=sg_polyorder,
                voigt_gamma_l=voigt_gamma_l,
                voigt_gamma_g=voigt_gamma_g,
                voigt_eta=voigt_eta,
                voigt_kernel_half_width=voigt_kernel_half_width,
            )
            self.output_channels = self.feature_map.output_channels

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected shape (batch, spectrum_length), got {tuple(x.shape)}.")
        if self.feature_map is None:
            return x.unsqueeze(-1)
        return self.feature_map(x)


class RawSubsampleCompressor(nn.Module):
    """Fixed subsampling baseline.

    The full ordered sequence is flattened and reduced to ``output_dim`` by
    adaptive average pooling.  This is intentionally simple and structure-agnostic.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self.output_dim = int(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected shape (batch, length, channels), got {tuple(x.shape)}.")
        pooled = F.adaptive_avg_pool1d(x.flatten(start_dim=1).unsqueeze(1), self.output_dim)
        return pooled.squeeze(1)


class PCACompressor(nn.Module):
    """Frozen PCA projection fit on the training split."""

    def __init__(self, mean: Tensor, components: Tensor) -> None:
        super().__init__()
        if mean.ndim != 1:
            raise ValueError("mean must be a vector.")
        if components.ndim != 2:
            raise ValueError("components must be a matrix.")
        if components.shape[1] != mean.shape[0]:
            raise ValueError("components and mean shapes are incompatible.")
        self.register_buffer("mean", mean.to(torch.float32), persistent=True)
        self.register_buffer("components", components.to(torch.float32), persistent=True)

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    @classmethod
    def from_sklearn(cls, pca: PCA) -> "PCACompressor":
        return cls(
            mean=torch.from_numpy(pca.mean_.astype("float32")),
            components=torch.from_numpy(pca.components_.astype("float32")),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected shape (batch, length, channels), got {tuple(x.shape)}.")
        flat = x.flatten(start_dim=1)
        centered = flat - self.mean.unsqueeze(0)
        return centered @ self.components.t()


class TNSequenceCompressor(nn.Module):
    """Wrapper around the Experiment 12 hierarchical TN compressor."""

    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        output_dim: int,
        *,
        site_length: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.compressor = HierarchicalTNCompressor(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            bond_dim=output_dim,
            site_length=site_length,
            dropout=dropout,
        )
        self.output_dim = int(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.compressor(x)


def _build_reupload_torch_layer(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """Build a compact re-uploading PennyLane TorchLayer."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def _circuit(inputs: Tensor, weights: Tensor) -> list[Tensor]:
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.RY(torch.pi * inputs[qubit], wires=qubit)
            for qubit in range(n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(_circuit, weight_shapes)


class SharedQuantumHead(nn.Module):
    """Shared quantum head used for raw/PCA/TN variants."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        *,
        n_qubits: int = 4,
        n_layers: int = 3,
        adapter_hidden_dim: int = 32,
        head_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        self.input_dim = int(input_dim)
        self.num_labels = int(num_labels)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_dim, self.n_qubits),
            nn.LayerNorm(self.n_qubits),
            nn.Tanh(),
        )
        self.qlayer = _build_reupload_torch_layer(self.n_qubits, self.n_layers)
        self.readout = nn.Sequential(
            nn.Linear(self.n_qubits, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, self.num_labels),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected shape (batch, {self.input_dim}), got {tuple(x.shape)}.")
        encoded = self.adapter(x)
        q_out = torch.stack([self.qlayer(encoded[i]) for i in range(encoded.shape[0])], dim=0)
        q_out = q_out.to(encoded.device)
        return self.readout(q_out)


class SharedClassicalHead(nn.Module):
    """Classical control head used for TN -> classical."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(num_labels)]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=False):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected shape (batch, {self.input_dim}), got {tuple(x.shape)}.")
        return self.network(x)


class Experiment14Classifier(nn.Module):
    """Compressor + head composition used by Experiment 14."""

    def __init__(self, compressor: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.compressor = compressor
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        compressed = self.compressor(x)
        return self.head(compressed)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))
