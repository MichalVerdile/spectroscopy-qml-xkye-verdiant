"""IR specialist quantum model for the TTN-10.2 weakest classes.

This variant is intentionally restricted to the 10 functional groups with the
lowest ``F1 TTN`` values in the provided TTN-10.2 error-analysis table:

    10  Azo compound
    19  Hydrazone
    21  Imine
    34  Thial
    1   Acyl halide
    0   Acid anhydride
    13  Enamine
    33  Sulfoxide
    27  Phosphine
    35  Thioamide

The model therefore defaults to a 10-logit specialist output instead of the
full 37-label IR multi-label target space. Use ``select_specialist_labels`` to
project full IR labels onto this specialist subset before training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
import math

from spectroscopy_qml.ir.tree_tensor_network.data_loader import FUNCTIONAL_GROUPS


TTN10_2_HARD_CLASS_INDICES = (10, 19, 21, 34, 1, 0, 13, 33, 27, 35)
ALL_FUNCTIONAL_GROUP_NAMES = tuple(FUNCTIONAL_GROUPS.keys())
TTN10_2_HARD_CLASS_NAMES = tuple(
    ALL_FUNCTIONAL_GROUP_NAMES[index] for index in TTN10_2_HARD_CLASS_INDICES
)
N_SPECIALIST_CLASSES = len(TTN10_2_HARD_CLASS_INDICES)


def select_specialist_labels(labels: torch.Tensor) -> torch.Tensor:
    """Project full IR label vectors onto the TTN-10.2 hard-class subset."""
    if labels.ndim < 2:
        raise ValueError("labels must have shape [batch, num_classes].")
    if labels.shape[-1] <= max(TTN10_2_HARD_CLASS_INDICES):
        raise ValueError(
            "labels do not contain the full IR functional-group axis needed for "
            f"indices {TTN10_2_HARD_CLASS_INDICES}."
        )

    index = torch.tensor(
        TTN10_2_HARD_CLASS_INDICES,
        device=labels.device,
        dtype=torch.long,
    )
    return torch.index_select(labels, dim=-1, index=index)


# ---------------------------------------------------------------------------
# Classical Feature Extractor — deeper, more aggressive compression
# ---------------------------------------------------------------------------

class SpectralFeatureExtractor(nn.Module):
    """
    Heavy 1D CNN that compresses ~4448 spectral bins down to a small
    number of features (typically 4).

    Architecture:
      Block 1: Conv(1→16,  k=15, s=4) → BN → ReLU → MaxPool(2)
      Block 2: Conv(16→32, k=9,  s=2) → BN → ReLU → MaxPool(2)
      Block 3: Conv(32→64, k=5,  s=2) → BN → ReLU → MaxPool(2)
      Head   : AdaptiveAvgPool(4) → Flatten
                → Linear(256→32) → ReLU
                → Linear(32→16)  → ReLU
                → Linear(16→out_features)

    Shape trace (L=4448, out_features=4):
      [B, 1, 4448] → [B, 16, 556] → [B, 32, 139] → [B, 64, 35]
                   → [B, 64, 4]   → [B, 256]     → [B, 4]
    """

    def __init__(self, out_features: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1 — coarse local motifs (absorption / emission bumps)
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2 — mid-level combinations
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3 — high-level abstract features
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.AdaptiveAvgPool1d(4),  # → [B, 64, 4]
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_features),
        )

    def forward(self, x):
        """x: [B, 1, L] → [B, out_features]"""
        B = x.size(0)
        h = self.backbone(x)  # [B, 64, 4]
        h = h.reshape(B, -1)  # [B, 256]
        return self.proj(h)  # [B, out_features]


# ---------------------------------------------------------------------------
# Angle Encoding Classifier
# ---------------------------------------------------------------------------

class AngleEncodingClassifier(nn.Module):
    """
    Hybrid quantum-classical model with angle (rotation) encoding.

    Pipeline:
      flux → heavy CNN extractor → small feature vector (dim = n_qubits)
           → bounded to [-π, π] via tanh
           → data-reuploading VQC (RY encoding + Rot gates + CNOT ring)
           → PauliZ expectation values
           → concatenate scalars → MLP head → logits

    Defaults (n_qubits=4, n_layers=6) match the smaller feature dimension
    from the new extractor. Fewer qubits is compensated by more
    re-uploading layers, which increases Fourier frequency content.
    """

    def __init__(
            self,
            num_classes: int = N_SPECIALIST_CLASSES,
            n_qubits: int = 4,
            n_layers: int = 6,
            n_scalars: int = 6,
            dropout: float = 0.2,
            specialist_indices: tuple[int, ...] = TTN10_2_HARD_CLASS_INDICES,
    ):
        super().__init__()
        self.specialist_indices = tuple(specialist_indices)
        if num_classes != len(self.specialist_indices):
            raise ValueError(
                "num_classes must match the number of specialist_indices. "
                f"Got num_classes={num_classes} and "
                f"{len(self.specialist_indices)} specialist indices."
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.specialist_class_names = tuple(
            ALL_FUNCTIONAL_GROUP_NAMES[index] for index in self.specialist_indices
        )

        # Heavy classical feature reduction: flux → n_qubits values
        self.extractor = SpectralFeatureExtractor(out_features=n_qubits)

        # Quantum device + circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="backprop",
        )

        # Trainable quantum parameters: (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Classical post-processing head
        head_in = n_qubits + n_scalars
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def _circuit(self, features, weights):
        """
        Data-reuploading VQC with angle encoding.

        Supports PennyLane parameter broadcasting:
          features : (B, n_qubits) — batched input
          weights  : (n_layers, n_qubits, 3) — shared across batch

        features[:, q] is shape (B,) → PennyLane runs the whole batch in
        one vectorised pass (no Python per-sample loop).
        """
        for layer in range(self.n_layers):
            # Encode data as RY rotations — re-uploaded each layer
            for q in range(self.n_qubits):
                qml.RY(features[:, q], wires=q)  # (B,) — broadcasted

            # Trainable rotations — scalars, shared across batch
            for q in range(self.n_qubits):
                qml.Rot(
                    weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    wires=q,
                )

            # Entanglement — circular CNOT chain
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def forward(self, flux, scalars=None):
        """
        Args:
            flux    : [B, 1, L]
            scalars : [B, n_scalars] or None
        Returns:
            logits  : [B, num_classes]
        """
        B = flux.size(0)

        # Keep track of where our data originally lived (likely 'mps')
        original_device = flux.device

        # Heavy classical extraction → bound to [-π, π]
        feat = self.extractor(flux)  # [B, n_qubits]
        feat = torch.tanh(feat) * math.pi  # bound to [-π, π]

        # --- THE FIX: MOVE DATA TO CPU FOR PENNYLANE ---
        feat_cpu = feat.cpu()
        weights_cpu = self.q_weights.cpu()

        # Run quantum circuit on CPU
        q_list = self.qnode(feat_cpu, weights_cpu)
        q_out = torch.stack(q_list, dim=1).float()  # [B, n_qubits]

        # --- MOVE BACK TO ORIGINAL DEVICE ---
        q_out = q_out.to(original_device)

        # Concatenate scalar features
        if scalars is not None and scalars.numel() > 0:
            q_out = torch.cat([q_out, scalars], dim=1)
        else:
            pad = torch.zeros(
                B, self.head[0].in_features - self.n_qubits,
                device=original_device,
            )
            q_out = torch.cat([q_out, pad], dim=1)

        return self.head(q_out)


# ---------------------------------------------------------------------------
# Factory (kept for API compatibility with existing training scripts)
# ---------------------------------------------------------------------------

def get_quantum_model(encoding: str = "angle", **kwargs) -> nn.Module:
    """
    Factory — angle encoding only in this version, per professor's feedback.
    Amplitude encoding has been removed.
    """
    if encoding == "angle":
        return AngleEncodingClassifier(**kwargs)
    else:
        raise ValueError(
            f"Unknown encoding '{encoding}'. "
            "Only 'angle' is supported in this version."
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("ANGLE ENCODING SPECIALIST — TTN 10.2 hard classes")
    print("=" * 64)
    print("Specialist indices:", TTN10_2_HARD_CLASS_INDICES)
    print("Specialist names:  ", ", ".join(TTN10_2_HARD_CLASS_NAMES))

    m = AngleEncodingClassifier(
        num_classes=N_SPECIALIST_CLASSES, n_qubits=4, n_layers=6, n_scalars=6,
    )
    x = torch.randn(2, 1, 4448)
    s = torch.randn(2, 6)
    out = m(x, s)

    p_total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    p_ext = sum(p.numel() for p in m.extractor.parameters() if p.requires_grad)
    p_q = m.q_weights.numel()
    p_head = sum(p.numel() for p in m.head.parameters() if p.requires_grad)

    print(f"  flux {tuple(x.shape)}  scalars {tuple(s.shape)}  → logits {tuple(out.shape)}")
    print(f"  Trainable params total:    {p_total:,}")
    print(f"    extractor (classical):   {p_ext:,}")
    print(f"    quantum weights:         {p_q:,}")
    print(f"    head (classical):        {p_head:,}")

    full_labels = torch.randint(0, 2, (2, len(ALL_FUNCTIONAL_GROUP_NAMES))).float()
    specialist_labels = select_specialist_labels(full_labels)
    print(f"  full labels shape:         {tuple(full_labels.shape)}")
    print(f"  specialist labels shape:   {tuple(specialist_labels.shape)}")
