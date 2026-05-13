import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair_blocks(start: int, num_qubits: int) -> list[tuple[int, int]]:
    """Return adjacent qubit pairs used by the QCNN convolution blocks."""
    return [(left, left + 1) for left in range(start, num_qubits - 1, 2)]


class ShallowQCNN(nn.Module):
    """
    Small QCNN-style circuit for latent vectors that have already been compressed.

    The circuit uses angle encoding followed by shallow alternating convolution and
    pooling-style blocks. It is intentionally small so it can specialize a handful
    of hard labels without replacing the frozen classical MPS backbone.
    """

    def __init__(self, num_qubits: int = 4, circuit_layers: int = 2, compressed_dim: int = 8):
        super().__init__()
        if not 4 <= compressed_dim <= 16:
            raise ValueError("compressed_dim must be between 4 and 16")
        if not 4 <= num_qubits <= 6:
            raise ValueError("num_qubits must be between 4 and 6")
        if not 1 <= circuit_layers <= 2:
            raise ValueError("circuit_layers must be between 1 and 2")

        self.num_qubits = num_qubits
        self.circuit_layers = circuit_layers
        self.compressed_dim = compressed_dim
        self.max_features = 2 * num_qubits * circuit_layers

        if compressed_dim > self.max_features:
            raise ValueError(
                f"compressed_dim={compressed_dim} exceeds circuit capacity "
                f"{self.max_features} for num_qubits={num_qubits}, circuit_layers={circuit_layers}"
            )

        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "PennyLane is required for the QCNN specialist head. "
                "Install it with: pip install pennylane pennylane-lightning"
            ) from exc

        self._qml = qml
        self.even_pairs = _pair_blocks(0, num_qubits)
        self.odd_pairs = _pair_blocks(1, num_qubits)
        self.pool_pairs = self.even_pairs

        dev = qml.device("default.qubit", wires=num_qubits)

        even_count = len(self.even_pairs)
        odd_count = len(self.odd_pairs)
        pool_count = len(self.pool_pairs)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, conv_even, conv_odd, pooling):
            feature_idx = 0
            for layer in range(circuit_layers):
                for qubit in range(num_qubits):
                    qml.RY(inputs[..., feature_idx], wires=qubit)
                    feature_idx += 1
                    qml.RZ(inputs[..., feature_idx], wires=qubit)
                    feature_idx += 1

                for pair_idx, (left, right) in enumerate(self.even_pairs):
                    params = conv_even[layer, pair_idx]
                    qml.CNOT(wires=[left, right])
                    qml.RY(params[0], wires=left)
                    qml.RZ(params[1], wires=right)
                    qml.CNOT(wires=[right, left])
                    qml.RY(params[2], wires=right)

                for pair_idx, (left, right) in enumerate(self.odd_pairs):
                    params = conv_odd[layer, pair_idx]
                    qml.CNOT(wires=[left, right])
                    qml.RY(params[0], wires=left)
                    qml.RZ(params[1], wires=right)
                    qml.CNOT(wires=[right, left])
                    qml.RY(params[2], wires=left)

                for pair_idx, (keep, discard) in enumerate(self.pool_pairs):
                    params = pooling[layer, pair_idx]
                    qml.CRX(params[0], wires=[discard, keep])
                    qml.CRZ(params[1], wires=[discard, keep])
                    qml.RY(params[2], wires=keep)

            return [qml.expval(qml.PauliZ(qubit)) for qubit in range(num_qubits)]

        self.qlayer = qml.qnn.TorchLayer(
            circuit,
            {
                "conv_even": (circuit_layers, even_count, 3),
                "conv_odd": (circuit_layers, odd_count, 3),
                "pooling": (circuit_layers, pool_count, 3),
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(x.shape)}")
        if x.size(1) > self.max_features:
            x = x[:, : self.max_features]
        elif x.size(1) < self.max_features:
            x = F.pad(x, (0, self.max_features - x.size(1)))
        return self.qlayer(x)


class LatentQCNNHead(nn.Module):
    """Compress MPS latents, run a shallow QCNN, and map to rare-or-hard-label logits."""

    def __init__(
        self,
        latent_dim: int,
        num_labels: int,
        compressed_dim: int = 8,
        num_qubits: int = 4,
        circuit_layers: int = 2,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.compressed_dim = compressed_dim
        self.num_qubits = num_qubits
        self.circuit_layers = circuit_layers

        self.compressor = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, compressed_dim),
            nn.Tanh(),
        )
        self.qcnn = ShallowQCNN(
            num_qubits=num_qubits,
            circuit_layers=circuit_layers,
            compressed_dim=compressed_dim,
        )
        self.output = nn.Linear(num_qubits, num_labels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        angles = self.compressor(latent) * math.pi
        q_features = self.qcnn(angles)
        return self.output(q_features)
