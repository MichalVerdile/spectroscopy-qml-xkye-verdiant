"""
Quantum Kernel for IR spectroscopy functional group classification.

Based on: Havlíček et al. (2019) "Supervised learning with quantum-enhanced
feature spaces" — Nature 567, 209–212.

Core idea:
    Instead of training a parametrised quantum circuit, we use a FIXED quantum
    feature map φ: ℝᴺ → Hilbert space and define a kernel

        K(x, x') = |⟨φ(x)|φ(x')⟩|²

    i.e. the squared overlap (fidelity) of two quantum states.  A classical SVM
    is then trained on this kernel matrix — the quantum circuit is never
    differentiated, which avoids barren plateaus entirely.

ZZ Feature Map (2 repetitions):
    For each repetition:
        H on every qubit
        RZ(xᵢ) on qubit i
        For neighbouring pairs (i, i+1):
            CNOT  →  RZ((π−xᵢ)(π−xᵢ₊₁))  →  CNOT
    The cross-term interactions encode correlations between spectral features.

Preprocessing pipeline:
    SNV-normalised spectrum (1800,)
        → PCA(n_components=N_QUBITS)
        → MinMaxScaler  → [−π, π]  (optimal range for RZ gates)

Multi-label output:
    OneVsRest SVM (one binary classifier per functional group class).

References:
    - Havlíček et al., Nature 567 (2019)
    - Schuld & Killoran, PRL 122 (2019)
    - PennyLane docs: qml.kernels
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS:  int   = 8      # PCA components = number of qubits
N_REPS:    int   = 2      # feature-map repetitions (depth)
N_CLASSES: int   = 37
INPUT_DIM: int   = 1800
USE_GPU:   bool  = False

# ── PennyLane device ──────────────────────────────────────────────────────────
_dev = qml.device("lightning.gpu" if USE_GPU else "lightning.qubit",
                  wires=N_QUBITS)


# ── ZZ Feature Map ────────────────────────────────────────────────────────────

def _zz_feature_map(x: np.ndarray) -> None:
    """
    Apply the ZZ Feature Map circuit to the current quantum state.

    Each repetition:
      1. Hadamard on all qubits
      2. RZ(xᵢ) on qubit i
      3. For each neighbouring pair (i, i+1):
             CNOT → RZ((π−xᵢ)(π−xᵢ₊₁)) → CNOT

    Args:
        x: Feature vector of shape (N_QUBITS,), values in [−π, π].
    """
    for _ in range(N_REPS):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
        for i in range(N_QUBITS):
            qml.RZ(x[i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ((np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
            qml.CNOT(wires=[i, i + 1])


@qml.qnode(_dev, interface="autograd")
def _kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute the fidelity kernel K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|².

    Implements the "swap test" via:
        |0⟩  →  U(x1)†  U(x2)  →  measure P(|0…0⟩)

    The probability of the all-zeros outcome equals the fidelity.

    Args:
        x1, x2: Feature vectors of shape (N_QUBITS,).

    Returns:
        Scalar fidelity in [0, 1].
    """
    _zz_feature_map(x2)
    qml.adjoint(_zz_feature_map)(x1)
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """K(x1, x2) — returns the |0…0⟩ probability (fidelity)."""
    return float(_kernel_circuit(x1, x2)[0])


def build_kernel_matrix(X1: np.ndarray, X2: np.ndarray,
                        verbose: bool = True) -> np.ndarray:
    """
    Compute the full N×M kernel matrix K[i,j] = K(X1[i], X2[j]).

    Uses PennyLane's ``qml.kernels.kernel_matrix`` for the square train
    matrix (exploits symmetry, computes only upper triangle) and falls back
    to the vectorised PennyLane batch for the rectangular test matrix.

    Args:
        X1: (N, N_QUBITS) query points.
        X2: (M, N_QUBITS) reference points. If X1 is X2 (same object or equal
            arrays) the symmetric shortcut is used.
        verbose: print row-level progress.

    Returns:
        (N, M) kernel matrix, values in [0, 1].
    """
    symmetric = X1 is X2 or (X1.shape == X2.shape and np.array_equal(X1, X2))

    if symmetric:
        # qml.kernels.kernel_matrix only computes upper triangle → 2× faster
        if verbose:
            print(f"  symmetric matrix ({len(X1)}×{len(X1)}) — upper-triangle only")
        K = qml.kernels.kernel_matrix(X1, X2, quantum_kernel)
    else:
        N, M = len(X1), len(X2)
        K = np.zeros((N, M))
        for i in range(N):
            if verbose and (i % max(1, N // 10) == 0):
                print(f"  row {i}/{N}  ({100*i/N:.0f}%)")
            for j in range(M):
                K[i, j] = quantum_kernel(X1[i], X2[j])
    return K


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def build_preprocessor(n_components: int = N_QUBITS) -> Pipeline:
    """
    Returns a sklearn Pipeline:
        PCA(n_components) → MinMaxScaler([−π, π])

    Fit on training data, transform train + test.
    """
    return Pipeline([
        ("pca",   PCA(n_components=n_components, random_state=42)),
        ("scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
    ])
