"""
QCNN + TTN Amplitude-Embedding model for IR spectroscopy functional group classification.

Architecture overview
---------------------
1. Amplitude embedding
   IR spectrum (1800 pts, SNV-normalised) is zero-padded to 2^11 = 2048, L2-normalised
   and embedded as quantum state amplitudes across 11 qubits:
       |ψ⟩ = Σᵢ xᵢ |i⟩      (xᵢ = normalised spectral intensity at bin i)

2. QCNN convolutional layers  (arXiv:1810.03787 – Cong et al.)
   Two passes of 2-qubit parametrised gates on neighbouring pairs:
     Pass 1 (even stride): (0,1),(2,3),(4,5),(6,7),(8,9)
     Pass 2 (odd  stride): (1,2),(3,4),(5,6),(7,8),(9,10)

3. TTN hierarchical merge     (arXiv:2104.02249 – Wall & D'Aguanno)
   Binary-tree contraction, qubit count: 11 → 6 → 3 → 2 → 1 (root = qubit 10)
     Level 1: (0,1),(2,3),(4,5),(6,7),(8,9)
     Level 2: (1,3),(5,7),(9,10)
     Level 3: (3,7)
     Level 4: (7,10)

4. Measurement
   ⟨Z⟩ on all 11 qubits  →  feature vector of shape (11,) in [−1, +1]

5. Classical head
   Linear(11 → 64) → GELU → Dropout(0.2) → Linear(64 → 37)
   Outputs raw logits for BCEWithLogitsLoss.

Cantor hierarchy mode (use_cantor=True)
---------------------------------------
Before amplitude embedding the padded spectrum is passed through a 1-D Haar wavelet
transform.  This reorders coefficients from coarsest (index 0 = DC component) to
finest (indices 1024–2047 = sample-level differences), aligning perfectly with the
TTN bottom-up hierarchy described in arXiv:2104.02249.

Bloch sphere extraction
-----------------------
After training, call ``model.get_bloch_vectors(x)`` to obtain the Bloch vector
(⟨X₁₀⟩, ⟨Y₁₀⟩, ⟨Z₁₀⟩) of the root qubit (qubit 10) for every input spectrum.
These 3-D coordinates lie within the unit ball — chemically similar spectra cluster
together on the sphere because the trained QCNN has shaped the latent space accordingly.

Gradient method: adjoint differentiation (most efficient for lightning.qubit).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

# ── Constants ──────────────────────────────────────────────────────────────────
N_QUBITS:    int = 11
AMP_DIM:     int = 2 ** N_QUBITS      # 2048  (≥ 1800 spectral points)
INPUT_DIM:   int = 1800
N_CLASSES:   int = 37
N_READOUT:   int = N_QUBITS - 1       # qubit 10 — root of the TTN
USE_GPU:     bool = False

# ── Wire pairs for QCNN convolutional passes ──────────────────────────────────
_CONV_EVEN: list[tuple[int, int]] = [(0,1),(2,3),(4,5),(6,7),(8,9)]      # 5 pairs
_CONV_ODD:  list[tuple[int, int]] = [(1,2),(3,4),(5,6),(7,8),(9,10)]     # 5 pairs

# ── Wire pairs for TTN hierarchical merge ─────────────────────────────────────
# Survivors after each level → successively fewer active qubits
_TTN_L1: list[tuple[int, int]] = [(0,1),(2,3),(4,5),(6,7),(8,9)]   # 5 pairs → {1,3,5,7,9,10}
_TTN_L2: list[tuple[int, int]] = [(1,3),(5,7),(9,10)]              # 3 pairs → {3,7,10}
_TTN_L3: list[tuple[int, int]] = [(3,7)]                           # 1 pair  → {7,10}
_TTN_L4: list[tuple[int, int]] = [(7,10)]                          # 1 pair  → {10} (root)

# Parameter counts per stage:
#   QCNN conv even:   5 × 8 =  40
#   QCNN conv odd:    5 × 8 =  40
#   TTN level 1:      5 × 8 =  40
#   TTN level 2:      3 × 8 =  24
#   TTN level 3:      1 × 8 =   8
#   TTN level 4:      1 × 8 =   8
#   Total quantum:              160

# ── PennyLane device ──────────────────────────────────────────────────────────
_dev = qml.device(
    "lightning.gpu" if USE_GPU else "lightning.qubit",
    wires=N_QUBITS,
)


# ── Gate primitive ────────────────────────────────────────────────────────────

def _two_qubit_block(wi: int, wj: int, p: torch.Tensor) -> None:
    """
    Parametrised 2-qubit SU(4)-approximation gate.

    Pattern (8 parameters):
        RZ(p0) RY(p1) on wi
        CNOT(wi → wj)
        RZ(p2) RY(p3) on wj
        CNOT(wj → wi)
        RZ(p4) RY(p5) on wi
        RZ(p6) RY(p7) on wj

    Inspired by the hardware-efficient ansatz from arXiv:1704.05018.
    """
    qml.RZ(p[0], wires=wi)
    qml.RY(p[1], wires=wi)
    qml.CNOT(wires=[wi, wj])
    qml.RZ(p[2], wires=wj)
    qml.RY(p[3], wires=wj)
    qml.CNOT(wires=[wj, wi])
    qml.RZ(p[4], wires=wi)
    qml.RY(p[5], wires=wi)
    qml.RZ(p[6], wires=wj)
    qml.RY(p[7], wires=wj)


def _apply_pairs(
    pairs: list[tuple[int, int]],
    weights: torch.Tensor,
) -> None:
    """Apply _two_qubit_block to each (wi, wj) pair using weights[k]."""
    for k, (wi, wj) in enumerate(pairs):
        _two_qubit_block(wi, wj, weights[k])


# ── Main quantum circuit  →  ⟨Z⟩ on all 11 qubits ────────────────────────────

@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def _circuit_z(
    inputs:    torch.Tensor,   # (2048,)  L2-normalised amplitude vector
    conv_w1:   torch.Tensor,   # (5, 8)   QCNN even-stride convolutional
    conv_w2:   torch.Tensor,   # (5, 8)   QCNN odd-stride convolutional
    ttn_l1_w:  torch.Tensor,   # (5, 8)   TTN level 1
    ttn_l2_w:  torch.Tensor,   # (3, 8)   TTN level 2
    ttn_l3_w:  torch.Tensor,   # (1, 8)   TTN level 3
    ttn_l4_w:  torch.Tensor,   # (1, 8)   TTN level 4
) -> list:
    """
    Returns ⟨Z⟩ expectation on every qubit → 11 values in [−1, +1].
    Used during training as the quantum feature vector.
    """
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=False)
    _apply_pairs(_CONV_EVEN, conv_w1)
    _apply_pairs(_CONV_ODD,  conv_w2)
    _apply_pairs(_TTN_L1, ttn_l1_w)
    _apply_pairs(_TTN_L2, ttn_l2_w)
    _apply_pairs(_TTN_L3, ttn_l3_w)
    _apply_pairs(_TTN_L4, ttn_l4_w)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ── Bloch-sphere readout circuit  →  ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on root qubit ──────────

@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def _circuit_bloch(
    inputs:    torch.Tensor,
    conv_w1:   torch.Tensor,
    conv_w2:   torch.Tensor,
    ttn_l1_w:  torch.Tensor,
    ttn_l2_w:  torch.Tensor,
    ttn_l3_w:  torch.Tensor,
    ttn_l4_w:  torch.Tensor,
) -> list:
    """
    Returns (⟨X₁₀⟩, ⟨Y₁₀⟩, ⟨Z₁₀⟩) for the root qubit.
    These are the Cartesian Bloch-sphere coordinates.
    All three expectations are computed from a single circuit execution.
    """
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=False)
    _apply_pairs(_CONV_EVEN, conv_w1)
    _apply_pairs(_CONV_ODD,  conv_w2)
    _apply_pairs(_TTN_L1, ttn_l1_w)
    _apply_pairs(_TTN_L2, ttn_l2_w)
    _apply_pairs(_TTN_L3, ttn_l3_w)
    _apply_pairs(_TTN_L4, ttn_l4_w)
    return [
        qml.expval(qml.PauliX(N_READOUT)),
        qml.expval(qml.PauliY(N_READOUT)),
        qml.expval(qml.PauliZ(N_READOUT)),
    ]


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _pad_and_normalise(x: torch.Tensor) -> torch.Tensor:
    """
    Pad a batch of spectra from 1800 to 2048 with zeros and L2-normalise.

    Args:
        x: (batch, 1800) float tensor — SNV-normalised spectra

    Returns:
        (batch, 2048) unit-norm tensor ready for AmplitudeEmbedding
    """
    batch = x.shape[0]
    out = torch.zeros(batch, AMP_DIM, device=x.device, dtype=x.dtype)
    out[:, :INPUT_DIM] = x
    norms = out.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return out / norms


def _haar_pad_and_normalise(x: torch.Tensor) -> torch.Tensor:
    """
    Haar wavelet transform → Cantor hierarchy encoding.

    Steps:
      1. Pad spectrum to 2048 with zeros.
      2. Apply lifting-scheme 1-D Haar wavelet transform (log₂(2048) = 11 levels).
      3. L2-normalise.

    After the transform the coefficient ordering is:
      index 0           : DC component (global mean)
      index 1           : coarsest detail (low-half vs high-half mean difference)
      indices 2–3       : next level details  (quarter-level)
      …
      indices 1024–2047 : finest details (per-sample differences)

    This ordering aligns with the TTN bottom-up hierarchy:
    the TTN leaf qubits (level 1 pairs) process the finest spectral details
    while the root qubit (qubit 10) aggregates the coarsest information —
    exactly the Cantor hierarchy described in the model design.

    Args:
        x: (batch, 1800) float tensor — SNV-normalised spectra

    Returns:
        (batch, 2048) L2-normalised Haar-coefficient tensor
    """
    batch = x.shape[0]
    result = torch.zeros(batch, AMP_DIM, device=x.device, dtype=x.dtype)
    result[:, :INPUT_DIM] = x

    # Lifting-scheme Haar transform (in-place on result)
    length = AMP_DIM
    while length > 1:
        half = length // 2
        evens = result[:, 0:length:2].clone()
        odds  = result[:, 1:length:2].clone()
        result[:, :half]     = (evens + odds) * 0.5
        result[:, half:length] = (evens - odds) * 0.5
        length = half

    norms = result.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return result / norms


# ── Model ─────────────────────────────────────────────────────────────────────

class QCNNIRClassifier(nn.Module):
    """
    QCNN + TTN hybrid quantum-classical classifier for IR functional-group prediction.

    Input  : (batch, 1800) float32 — SNV-normalised IR spectrum
    Output : (batch, 37)   float32 — raw logits for BCEWithLogitsLoss

    Args:
        use_cantor: If True, apply Haar wavelet transform before amplitude
                    embedding, which creates the Cantor-hierarchy frequency ordering.
                    Defaults to False (standard zero-padded amplitude embedding).
    """

    def __init__(self, use_cantor: bool = False) -> None:
        super().__init__()
        self.use_cantor = use_cantor

        # ── Quantum layer ──────────────────────────────────────────────────────
        weight_shapes = {
            "conv_w1":  (len(_CONV_EVEN), 8),   # (5, 8)
            "conv_w2":  (len(_CONV_ODD),  8),   # (5, 8)
            "ttn_l1_w": (len(_TTN_L1),    8),   # (5, 8)
            "ttn_l2_w": (len(_TTN_L2),    8),   # (3, 8)
            "ttn_l3_w": (len(_TTN_L3),    8),   # (1, 8)
            "ttn_l4_w": (len(_TTN_L4),    8),   # (1, 8)
        }
        self.qlayer = qml.qnn.TorchLayer(_circuit_z, weight_shapes)

        # ── Classical head ─────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_CLASSES),
        )

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Pad + (optionally Haar-transform) + L2-normalise → (batch, 2048)."""
        if self.use_cantor:
            return _haar_pad_and_normalise(x)
        return _pad_and_normalise(x)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1800) SNV-normalised spectra

        Returns:
            (batch, 37) raw logits
        """
        amplitudes = self._prep(x)                             # (batch, 2048)

        # Process each sample individually through the quantum circuit.
        # TorchLayer handles a single sample (1-D input); we stack the results.
        q_out = torch.stack(
            [self.qlayer(amplitudes[i]) for i in range(amplitudes.shape[0])]
        )                                                       # (batch, 11)

        return self.head(q_out)                                 # (batch, 37)

    # ── Bloch sphere extraction ────────────────────────────────────────────────

    @torch.no_grad()
    def get_bloch_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Bloch vector (rₓ, r_y, r_z) of the root qubit (qubit 10)
        for each input spectrum.

        The Bloch vector is:
            rₓ = ⟨X₁₀⟩,   r_y = ⟨Y₁₀⟩,   r_z = ⟨Z₁₀⟩
        and satisfies |r| ≤ 1 (equality for pure single-qubit states;
        the root qubit is entangled with the others, so |r| < 1 in general).

        Args:
            x: (batch, 1800) SNV-normalised spectra

        Returns:
            (batch, 3) Bloch coordinates for visualisation
        """
        amplitudes = self._prep(x)

        # Extract the shared weights from TorchLayer to pass to _circuit_bloch.
        # TorchLayer registers each weight as nn.Parameter under its key name.
        w = {name: param for name, param in self.qlayer.named_parameters()}

        bloch_vecs = torch.stack([
            torch.stack(
                _circuit_bloch(
                    amplitudes[i],
                    conv_w1=w["conv_w1"],
                    conv_w2=w["conv_w2"],
                    ttn_l1_w=w["ttn_l1_w"],
                    ttn_l2_w=w["ttn_l2_w"],
                    ttn_l3_w=w["ttn_l3_w"],
                    ttn_l4_w=w["ttn_l4_w"],
                )
            )
            for i in range(amplitudes.shape[0])
        ])                                                      # (batch, 3)

        return bloch_vecs


# ── Parameter summary ──────────────────────────────────────────────────────────

def param_summary(model: QCNNIRClassifier) -> None:
    """Print quantum vs. classical parameter counts."""
    total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_total = sum(p.numel() for p in model.qlayer.parameters() if p.requires_grad)
    c_total = total - q_total

    print("QCNNIRClassifier  —  parameter summary")
    print(f"  Quantum  : {q_total:>8,}  "
          f"(QCNN conv: {2*5*8}, TTN: {(5+3+1+1)*8})")
    print(f"  Classical: {c_total:>8,}  (head: 11→64→37)")
    print(f"  Total    : {total:>8,}")
    print(f"  N_QUBITS : {N_QUBITS}   AMP_DIM : {AMP_DIM}")
    print(f"  use_cantor: {model.use_cantor}")


if __name__ == "__main__":
    model = QCNNIRClassifier(use_cantor=False)
    param_summary(model)

    # Quick shape check
    dummy = torch.randn(2, INPUT_DIM)
    out   = model(dummy)
    print(f"\nForward check  input: {tuple(dummy.shape)}  →  output: {tuple(out.shape)}")

    bv = model.get_bloch_vectors(dummy)
    print(f"Bloch vectors  shape: {tuple(bv.shape)}   "
          f"norms: {bv.norm(dim=1).tolist()}")
