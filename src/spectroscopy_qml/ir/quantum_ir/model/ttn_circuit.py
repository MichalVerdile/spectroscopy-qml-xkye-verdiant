"""
Parametrisierte Quanten-Encoder-Schaltkreise für jeden Cantor-Level.

Jeder Level erhält ein eigenes AmplitudeEmbedding gefolgt von einem
variationellen Rot+Ring-CNOT Ansatz und abschliessender Z-Messung.

Ansatz pro Schicht
------------------
    für i = 0 … n_qubits-1:
        Rot(θᵢ, φᵢ, ωᵢ)   ← 3 Euler-Winkel, trainierbar
    Ring-CNOT: 0→1→2→…→(n-1)→0  ← Verschränkung

Parameter pro Encoder
---------------------
    Viertel  (9 Qubits, 2 Schichten) : 2 × 9 × 3 =  54
    Hälfte  (10 Qubits, 2 Schichten) : 2 × 10 × 3 = 60
    Voll    (11 Qubits, 3 Schichten) : 3 × 11 × 3 = 99

Bloch-Sphäre
------------
    `encoder.get_bloch_vectors(x)` liefert (⟨X⟩, ⟨Y⟩, ⟨Z⟩) des
    Root-Qubits (letztes Qubit) — der echte Quantenzustand auf der Sphäre.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml


class QuantumEncoderLevel(nn.Module):
    """
    Trainierbarer Quanten-Encoder für einen Cantor-Level.

    Wendet AmplitudeEmbedding → variationelle Rot+CNOT-Schichten → Z-Messung an.

    Args:
        n_qubits:  Anzahl Qubits (9, 10 oder 11 für Viertel/Hälfte/Voll)
        n_layers:  Anzahl variationeller Schichten
        use_gpu:   Falls True: lightning.gpu, sonst lightning.qubit

    Input  (forward) : (batch, 2^n_qubits) — L2-normalisierter Amplitudenvektor
    Output (forward) : (batch, n_qubits)   — Pauli-Z Erwartungswerte in [−1, +1]
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        use_gpu: bool = False,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

        dev_name = "lightning.gpu" if use_gpu else "lightning.qubit"

        # ── Haupt-QNode: ⟨Z⟩ auf allen Qubits ───────────────────────────────
        _dev_z = qml.device(dev_name, wires=n_qubits)

        @qml.qnode(_dev_z, interface="torch", diff_method="adjoint")
        def _circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            """
            Args:
                inputs:  (2^n_qubits,) L2-normalisierter Amplitudenvektor
                weights: (n_layers, n_qubits, 3) Euler-Winkel

            Returns:
                Liste von n_qubits Pauli-Z Erwartungswerten in [−1, +1]
            """
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=False)
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=i,
                    )
                # Ring-Verschränkung: 0→1→…→(n-1)→0
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # ── Bloch-QNode: ⟨X⟩, ⟨Y⟩, ⟨Z⟩ des Root-Qubits ────────────────────
        _dev_bloch = qml.device(dev_name, wires=n_qubits)
        root = n_qubits - 1  # letztes Qubit = Root

        @qml.qnode(_dev_bloch, interface="torch", diff_method="adjoint")
        def _circuit_bloch(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            """Gibt (⟨X⟩, ⟨Y⟩, ⟨Z⟩) des Root-Qubits zurück."""
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=False)
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=i,
                    )
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [
                qml.expval(qml.PauliX(root)),
                qml.expval(qml.PauliY(root)),
                qml.expval(qml.PauliZ(root)),
            ]

        # Gespeichert als Attribut für get_bloch_vectors
        self._circuit_bloch = _circuit_bloch

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2^n_qubits) — L2-normalisiertes Spektralsegment

        Returns:
            (batch, n_qubits) Pauli-Z Erwartungswerte in [−1, +1]
        """
        return torch.stack(
            [self.qlayer(x[i]) for i in range(x.shape[0])]
        )

    # ── Bloch-Sphäre ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_bloch_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Berechnet den Bloch-Vektor (rₓ, r_y, r_z) des Root-Qubits für jedes
        Sample.

        Der Bloch-Vektor ist:
            rₓ = ⟨X_{n-1}⟩,  r_y = ⟨Y_{n-1}⟩,  r_z = ⟨Z_{n-1}⟩

        Für ein reines Einzelqubit-System gilt |r| = 1 (Oberfläche der Sphäre).
        Da das Root-Qubit mit anderen verschränkt ist, gilt typischerweise |r| < 1
        (Inneres der Sphäre = gemischter Zustand des reduzierten Systems).

        Args:
            x: (batch, 2^n_qubits) — L2-normalisierte Amplituden

        Returns:
            (batch, 3) Bloch-Koordinaten
        """
        # Gewichte aus TorchLayer extrahieren
        weights = dict(self.qlayer.named_parameters())["weights"]

        return torch.stack([
            torch.stack(
                self._circuit_bloch(x[i], weights)
            )
            for i in range(x.shape[0])
        ])

    # ── Info ───────────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        """Anzahl trainierbarer Quantenparameter."""
        return self.n_layers * self.n_qubits * 3

    def __repr__(self) -> str:
        return (
            f"QuantumEncoderLevel("
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"dim={self.dim}, "
            f"params={self.n_params()})"
        )


if __name__ == "__main__":
    print("QuantumEncoderLevel — Smoke-Tests\n")

    for n_q, n_l, name in [(9, 2, "Viertel"), (10, 2, "Hälfte"), (11, 3, "Voll")]:
        enc = QuantumEncoderLevel(n_qubits=n_q, n_layers=n_l)
        x = torch.randn(2, enc.dim)
        # L2-normalisieren
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

        out = enc(x)
        bv  = enc.get_bloch_vectors(x)
        print(f"  {name:8s} ({n_q} Qubits, {n_l} Schichten)")
        print(f"    Forward:      {tuple(x.shape)} → {tuple(out.shape)}")
        print(f"    Bloch-Vekt.:  {tuple(x.shape)} → {tuple(bv.shape)}")
        print(f"    |r| (norms):  {bv.norm(dim=1).tolist()}")
        print(f"    Param-Anzahl: {enc.n_params()}\n")
