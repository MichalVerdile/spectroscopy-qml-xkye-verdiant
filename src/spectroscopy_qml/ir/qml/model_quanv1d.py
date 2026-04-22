"""
Quanvolutional neural network for IR spectroscopy.

This module adapts Henderson et al. (arXiv:1904.04767), "Quanvolutional
Neural Networks", from 2-D image patches to 1-D IR spectra.

The paper's key design choice is a fixed quanvolutional layer: local input
patches are encoded into basis states, transformed by random quantum circuits,
decoded to scalar feature values, and then fed to an otherwise classical neural
network.  The quantum filters are not variationally trained.

For IR spectra the adapted stack is:

    IR spectrum -> 1-D quanvolution -> Conv1d -> Pool -> Conv1d -> Pool -> FC

The quanvolutional layer follows the paper's lookup-table idea. Since basis
encoding maps each local patch to one of 2**patch_size binary states, all
possible random-circuit outputs are precomputed once at initialization. Forward
passes then use fast tensor indexing instead of re-running a simulator for every
training sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
import torch
from torch import Tensor, nn

INPUT_DIM: int = 1800
N_CLASSES: int = 37
DEFAULT_PATCH_SIZE: int = 9
DEFAULT_STRIDE: int = 4
DEFAULT_N_FILTERS: int = 25


@dataclass(frozen=True)
class QuantumGate:
    """A fixed gate in one random quanvolutional filter."""

    name: str
    wires: tuple[int, ...]
    params: tuple[float, ...] = ()


def _rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.asarray([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.asarray([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.asarray(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def _phase(theta: float) -> np.ndarray:
    return np.asarray([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.asarray(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def _single_qubit_matrix(gate: QuantumGate) -> np.ndarray:
    if gate.name == "rx":
        return _rx(gate.params[0])
    if gate.name == "ry":
        return _ry(gate.params[0])
    if gate.name == "rz":
        return _rz(gate.params[0])
    if gate.name == "u3":
        return _u3(gate.params[0], gate.params[1], gate.params[2])
    if gate.name == "phase":
        return _phase(gate.params[0])
    if gate.name == "t":
        return _phase(pi / 4.0)
    if gate.name == "h":
        return np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / sqrt(2.0)
    raise ValueError(f"Unsupported single-qubit gate: {gate.name}")


def _two_qubit_matrix(gate: QuantumGate) -> np.ndarray:
    if gate.name == "cnot":
        return np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.complex128,
        )
    if gate.name == "swap":
        return np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.complex128,
        )
    if gate.name == "sqrtswap":
        a = (1.0 + 1.0j) / 2.0
        b = (1.0 - 1.0j) / 2.0
        return np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, a, b, 0.0],
             [0.0, b, a, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.complex128,
        )
    if gate.name == "crz":
        theta = gate.params[0]
        return np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, np.exp(-0.5j * theta), 0.0],
                [0.0, 0.0, 0.0, np.exp(0.5j * theta)],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported two-qubit gate: {gate.name}")


def _apply_unitary(
    state: np.ndarray,
    unitary: np.ndarray,
    wires: tuple[int, ...],
    n_qubits: int,
) -> np.ndarray:
    """Apply a one- or two-qubit unitary to a statevector."""
    axes = list(wires) + [axis for axis in range(n_qubits) if axis not in wires]
    inverse_axes = np.argsort(axes)
    tensor = state.reshape((2,) * n_qubits)
    transposed = np.transpose(tensor, axes)
    front_dim = 2 ** len(wires)
    rest_dim = 2 ** (n_qubits - len(wires))
    updated = unitary @ transposed.reshape(front_dim, rest_dim)
    return np.transpose(updated.reshape((2,) * n_qubits), inverse_axes).reshape(-1)


def _basis_index_to_count(index: int) -> int:
    return int(index).bit_count()


def _generate_random_filter(
    patch_size: int,
    rng: np.random.Generator,
    connection_probability: float,
    max_single_qubit_gates: int,
) -> tuple[QuantumGate, ...]:
    """Generate one fixed random quantum filter, following the paper's recipe."""
    if not 0.0 <= connection_probability <= 1.0:
        raise ValueError("connection_probability must be in [0, 1].")

    gates: list[QuantumGate] = []
    two_qubit_names = ("cnot", "swap", "sqrtswap", "crz")
    for left in range(patch_size):
        for right in range(left + 1, patch_size):
            if rng.random() > connection_probability:
                continue
            name = str(rng.choice(two_qubit_names))
            wires = (left, right)
            if name in {"cnot", "crz"} and rng.random() < 0.5:
                wires = (right, left)
            params = (float(rng.uniform(0.0, 2.0 * pi)),) if name == "crz" else ()
            gates.append(QuantumGate(name=name, wires=wires, params=params))

    single_qubit_names = ("rx", "ry", "rz", "u3", "phase", "t", "h")
    n_single = int(rng.integers(0, max_single_qubit_gates + 1))
    for _ in range(n_single):
        name = str(rng.choice(single_qubit_names))
        wire = int(rng.integers(0, patch_size))
        if name in {"rx", "ry", "rz", "phase"}:
            params = (float(rng.uniform(0.0, 2.0 * pi)),)
        elif name == "u3":
            params = tuple(float(v) for v in rng.uniform(0.0, 2.0 * pi, size=3))
        else:
            params = ()
        gates.append(QuantumGate(name=name, wires=(wire,), params=params))

    rng.shuffle(gates)
    return tuple(gates)


def _simulate_filter_lookup(
    gates: tuple[QuantumGate, ...],
    patch_size: int,
    decode: str,
) -> np.ndarray:
    """Precompute one filter's output for every binary encoded patch."""
    if decode not in {"count", "count_normalized", "count_centered"}:
        raise ValueError("decode must be 'count', 'count_normalized', or 'count_centered'.")

    n_states = 2 ** patch_size
    lookup = np.empty(n_states, dtype=np.float32)
    for basis_index in range(n_states):
        state = np.zeros(n_states, dtype=np.complex128)
        state[basis_index] = 1.0
        for gate in gates:
            matrix = (
                _single_qubit_matrix(gate)
                if len(gate.wires) == 1
                else _two_qubit_matrix(gate)
            )
            state = _apply_unitary(state, matrix, gate.wires, patch_size)

        decoded_count = _basis_index_to_count(int(np.argmax(np.abs(state) ** 2)))
        if decode == "count":
            lookup[basis_index] = float(decoded_count)
        elif decode == "count_normalized":
            lookup[basis_index] = float(decoded_count) / float(patch_size)
        else:
            lookup[basis_index] = 2.0 * float(decoded_count) / float(patch_size) - 1.0
    return lookup


def build_quanvolution_lookup(
    patch_size: int = DEFAULT_PATCH_SIZE,
    n_filters: int = DEFAULT_N_FILTERS,
    seed: int = 42,
    connection_probability: float = 0.5,
    max_single_qubit_gates: int | None = None,
    decode: str = "count_normalized",
) -> tuple[np.ndarray, tuple[tuple[QuantumGate, ...], ...]]:
    """Build fixed quanvolutional lookup tables and retain filter descriptions."""
    if patch_size <= 1:
        raise ValueError("patch_size must be greater than 1.")
    if patch_size > 12:
        raise ValueError("patch_size > 12 creates large 2**patch_size lookup tables.")
    if n_filters <= 0:
        raise ValueError("n_filters must be positive.")

    rng = np.random.default_rng(seed)
    max_gates = 2 * patch_size if max_single_qubit_gates is None else max_single_qubit_gates
    filters = tuple(
        _generate_random_filter(
            patch_size=patch_size,
            rng=rng,
            connection_probability=connection_probability,
            max_single_qubit_gates=max_gates,
        )
        for _ in range(n_filters)
    )
    lookup = np.stack(
        [_simulate_filter_lookup(gates, patch_size=patch_size, decode=decode) for gates in filters],
        axis=0,
    )
    return lookup, filters


class Quanvolution1D(nn.Module):
    """Fixed 1-D quanvolutional layer for spectra."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        patch_size: int = DEFAULT_PATCH_SIZE,
        stride: int = DEFAULT_STRIDE,
        n_filters: int = DEFAULT_N_FILTERS,
        threshold: float = 0.0,
        seed: int = 42,
        connection_probability: float = 0.5,
        max_single_qubit_gates: int | None = None,
        decode: str = "count_normalized",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if patch_size <= 1 or patch_size > input_dim:
            raise ValueError("patch_size must be in [2, input_dim].")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.input_dim = int(input_dim)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.n_filters = int(n_filters)
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.decode = decode
        self.n_patches = 1 + (self.input_dim - self.patch_size) // self.stride

        lookup, filters = build_quanvolution_lookup(
            patch_size=self.patch_size,
            n_filters=self.n_filters,
            seed=self.seed,
            connection_probability=connection_probability,
            max_single_qubit_gates=max_single_qubit_gates,
            decode=decode,
        )
        self.filters = filters
        self.register_buffer("lookup", torch.as_tensor(lookup, dtype=torch.float32), persistent=True)
        powers = 2 ** torch.arange(self.patch_size - 1, -1, -1, dtype=torch.long)
        self.register_buffer("_bit_powers", powers, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}.")

        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        bits = (patches > self.threshold).to(torch.long)
        codes = torch.sum(bits * self._bit_powers.view(1, 1, -1), dim=-1)
        features = self.lookup[:, codes].permute(1, 0, 2).contiguous()
        return features


class QuanvolutionalIRClassifier(nn.Module):
    """
    IR classifier using a paper-style quanvolutional first layer.

    The quantum part is intentionally fixed, as in arXiv:1904.04767. Only the
    downstream classical 1-D CNN and fully connected head are trained.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_classes: int = N_CLASSES,
        patch_size: int = DEFAULT_PATCH_SIZE,
        stride: int = DEFAULT_STRIDE,
        n_filters: int = DEFAULT_N_FILTERS,
        threshold: float = 0.0,
        seed: int = 42,
        connection_probability: float = 0.5,
        conv1_channels: int = 50,
        conv2_channels: int = 64,
        fc_hidden_dim: int = 1024,
        fc_pool_bins: int = 16,
        dropout: float = 0.4,
        decode: str = "count_normalized",
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if fc_pool_bins <= 0:
            raise ValueError("fc_pool_bins must be positive.")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.quanv = Quanvolution1D(
            input_dim=input_dim,
            patch_size=patch_size,
            stride=stride,
            n_filters=n_filters,
            threshold=threshold,
            seed=seed,
            connection_probability=connection_probability,
            decode=decode,
        )
        self.features = nn.Sequential(
            nn.Conv1d(n_filters, conv1_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            nn.AdaptiveAvgPool1d(fc_pool_bins),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv2_channels * fc_pool_bins, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        q_features = self.quanv(x)
        classical_features = self.features(q_features)
        return self.head(classical_features)


def param_summary(model: QuanvolutionalIRClassifier) -> dict[str, int]:
    """Return trainable and fixed-lookup parameter counts."""
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    lookup_values = int(model.quanv.lookup.numel())
    return {
        "trainable_parameters": trainable,
        "fixed_lookup_values": lookup_values,
        "quanvolutional_filters": model.quanv.n_filters,
        "patch_size": model.quanv.patch_size,
        "n_patches": model.quanv.n_patches,
    }


if __name__ == "__main__":
    model = QuanvolutionalIRClassifier()
    print("QuanvolutionalIRClassifier parameter summary")
    for key, value in param_summary(model).items():
        print(f"  {key}: {value:,}")
