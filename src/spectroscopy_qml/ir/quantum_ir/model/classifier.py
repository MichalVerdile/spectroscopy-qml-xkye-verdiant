"""
Cantor-Quanten IR-Klassifikator — vollständige Pipeline.

Architektur
-----------

    IR-Spektrum (1800,)
          │
          ▼
    ┌──────────────────────────────────────────────────────────┐
    │  CantorSplit                                             │
    │  ┌──────────────────┐  ┌───────────────┐  ┌──────────┐ │
    │  │ Viertel 0–3      │  │ Hälfte 0–1    │  │ Voll     │ │
    │  │ (je 450 Punkte)  │  │ (je 900 Pkt.) │  │ (1800)   │ │
    │  └──────────────────┘  └───────────────┘  └──────────┘ │
    └──────────────────────────────────────────────────────────┘
          │ pad+normalisieren       │ pad+normalisieren   │ pad+normalisieren
          ▼                        ▼                      ▼
    4 × (batch, 512)         2 × (batch,1024)       (batch, 2048)
          │                        │                      │
          ▼                        ▼                      ▼
    ┌──────────────┐  ┌──────────────────────┐  ┌──────────────┐
    │ q_quarter    │  │ q_half0  / q_half1   │  │ q_full       │
    │ (9 Qubits,   │  │ (10 Qubits, 2 Layer) │  │ (11 Qubits,  │
    │  2 Layer,    │  │ 60 Params je Hälfte  │  │  3 Layer,    │
    │  54 Params,  │  └──────────────────────┘  │  99 Params)  │
    │  geteilt)    │                             └──────────────┘
    └──────────────┘
          │ 4×(batch,9)            │ 2×(batch,10)          │ (batch,11)
          ▼                        ▼                        ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Klassischer TTN-Merger  (trainierbar)                       │
    │                                                              │
    │  m01 = Linear(18→9)+GELU  [Q0‖Q1]                           │
    │  m23 = Linear(18→9)+GELU  [Q2‖Q3]                           │
    │  m_h0 = Linear(19→10)+GELU  [H0‖m01]  ← Hälfte + Viertel   │
    │  m_h1 = Linear(19→10)+GELU  [H1‖m23]                        │
    │  g = Linear(31→16)+GELU   [F‖m_h0‖m_h1]  ← Global-Merge    │
    └──────────────────────────────────────────────────────────────┘
          │ (batch, 16)
          ▼
    ┌──────────────────┐
    │  Klassischer Kopf│
    │  Linear(16→37)   │
    └──────────────────┘
          │
          ▼
    37 Logits (BCEWithLogitsLoss)

Trainierbare Parameter
----------------------
  Quantenparameter:
    q_quarter  (geteilt)  :   54   (2 × 9 × 3)
    q_half0               :   60   (2 × 10 × 3)
    q_half1               :   60   (2 × 10 × 3)
    q_full                :   99   (3 × 11 × 3)
    ──────────────────────────────────────────
    Quantum gesamt        :  273

  Klassische Parameter:
    merge_q01             :  171   (18×9 + 9)
    merge_q23             :  171   (18×9 + 9)
    merge_h0              :  200   (19×10 + 10)
    merge_h1              :  200   (19×10 + 10)
    merge_global          :  512   (31×16 + 16)
    head                  :  629   (16×37 + 37)
    ──────────────────────────────────────────
    Klassisch gesamt      : 1'883

  Total                   : 2'156
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from spectroscopy_qml.ir.quantum_ir.model.encoding import CantorSplit
from spectroscopy_qml.ir.quantum_ir.model.ttn_circuit import QuantumEncoderLevel

# ── Konstanten ────────────────────────────────────────────────────────────────
N_CLASSES: int = 37


class CantorQuantumClassifier(nn.Module):
    """
    Cantor-Quanten IR-Klassifikator für 37 funktionelle Gruppen.

    Input  : (batch, 1800) float32 — SNV-normalisiertes IR-Spektrum
    Output : (batch, 37)   float32 — rohe Logits für BCEWithLogitsLoss

    Args:
        use_gpu:          Falls True: lightning.gpu für alle Encoder (Standard: False)
        shared_quarters:  Falls True: alle 4 Viertel teilen dieselben Quantengewichte
                          (spart Params, erzwingt Transversalinvarianz).
                          Falls False: je ein eigener 9-Qubit-Encoder pro Viertel.
    """

    def __init__(
        self,
        use_gpu:         bool = False,
        shared_quarters: bool = True,
    ) -> None:
        super().__init__()
        self.shared_quarters = shared_quarters

        # ── 1. Cantor-Split ───────────────────────────────────────────────────
        self.splitter = CantorSplit()

        # ── 2. Viertel-Encoder (Ebene 3, 9 Qubits) ───────────────────────────
        if shared_quarters:
            # Geteilte Gewichte: ein Encoder für alle 4 Viertel
            self.q_quarter = QuantumEncoderLevel(n_qubits=9, n_layers=2, use_gpu=use_gpu)
        else:
            # Separate Gewichte: spektralregionspezifische Encoder
            self.q_quarter0 = QuantumEncoderLevel(n_qubits=9, n_layers=2, use_gpu=use_gpu)
            self.q_quarter1 = QuantumEncoderLevel(n_qubits=9, n_layers=2, use_gpu=use_gpu)
            self.q_quarter2 = QuantumEncoderLevel(n_qubits=9, n_layers=2, use_gpu=use_gpu)
            self.q_quarter3 = QuantumEncoderLevel(n_qubits=9, n_layers=2, use_gpu=use_gpu)

        # ── 3. Hälften-Encoder (Ebene 2, 10 Qubits) ──────────────────────────
        # Separate Encoder: 0–900 cm⁻¹ und 900–1800 cm⁻¹ sind chemisch sehr verschieden
        self.q_half0 = QuantumEncoderLevel(n_qubits=10, n_layers=2, use_gpu=use_gpu)
        self.q_half1 = QuantumEncoderLevel(n_qubits=10, n_layers=2, use_gpu=use_gpu)

        # ── 4. Volles-Spektrum-Encoder (Ebene 1, 11 Qubits) ──────────────────
        self.q_full = QuantumEncoderLevel(n_qubits=11, n_layers=3, use_gpu=use_gpu)

        # ── 5. Klassischer TTN-Merger ─────────────────────────────────────────
        # Viertel-Paare zusammenführen (9+9=18 → 9)
        self.merge_q01 = nn.Sequential(nn.Linear(18,  9), nn.GELU())
        self.merge_q23 = nn.Sequential(nn.Linear(18,  9), nn.GELU())

        # Hälfte + zusammengeführte Viertel (10+9=19 → 10)
        self.merge_h0  = nn.Sequential(nn.Linear(19, 10), nn.GELU())
        self.merge_h1  = nn.Sequential(nn.Linear(19, 10), nn.GELU())

        # Globale Zusammenführung: voll + beide Hälften-Merges (11+10+10=31 → 16)
        self.merge_global = nn.Sequential(nn.Linear(31, 16), nn.GELU())

        # ── 6. Klassifikationskopf ────────────────────────────────────────────
        self.head = nn.Linear(16, N_CLASSES)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1800) SNV-normalisiertes IR-Spektrum

        Returns:
            (batch, 37) rohe Logits
        """
        segs = self.splitter(x)

        # Ebene 3 — Viertel (geteilter oder separater Encoder)
        if self.shared_quarters:
            q0 = self.q_quarter(segs["quarter0"])
            q1 = self.q_quarter(segs["quarter1"])
            q2 = self.q_quarter(segs["quarter2"])
            q3 = self.q_quarter(segs["quarter3"])
        else:
            q0 = self.q_quarter0(segs["quarter0"])
            q1 = self.q_quarter1(segs["quarter1"])
            q2 = self.q_quarter2(segs["quarter2"])
            q3 = self.q_quarter3(segs["quarter3"])

        # Ebene 2 — Hälften
        h0 = self.q_half0(segs["half0"])
        h1 = self.q_half1(segs["half1"])

        # Ebene 1 — Volles Spektrum
        f  = self.q_full(segs["full"])

        # Klassischer TTN-Merger
        m01 = self.merge_q01(torch.cat([q0, q1], dim=-1))   # (batch, 9)
        m23 = self.merge_q23(torch.cat([q2, q3], dim=-1))   # (batch, 9)
        m_h0 = self.merge_h0(torch.cat([h0, m01], dim=-1))  # (batch, 10)
        m_h1 = self.merge_h1(torch.cat([h1, m23], dim=-1))  # (batch, 10)
        g = self.merge_global(
            torch.cat([f, m_h0, m_h1], dim=-1)              # (batch, 31)
        )                                                    # (batch, 16)

        return self.head(g)                                  # (batch, 37)

    # ── Bloch-Sphäre ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_bloch_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bloch-Vektor (⟨X⟩, ⟨Y⟩, ⟨Z⟩) des Root-Qubits des vollen Spektrum-Encoders.

        Der vollständige Spektrum-Encoder (11 Qubits) verarbeitet das globale
        Spektralmuster. Sein Root-Qubit (Qubit 10) fasst den gesamten Quantenzustand
        zusammen — chemisch ähnliche Spektren clustern auf der Bloch-Sphäre.

        Args:
            x: (batch, 1800)

        Returns:
            (batch, 3) — Bloch-Koordinaten in [−1, +1]³
        """
        segs = self.splitter(x)
        return self.q_full.get_bloch_vectors(segs["full"])

    @torch.no_grad()
    def get_all_bloch_vectors(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Bloch-Vektoren aller Encoder-Root-Qubits.

        Nützlich, um zu vergleichen, wie verschiedene Frequenzbereiche
        die Quantenzustände formen.

        Returns:
            dict mit Schlüsseln 'full', 'half0', 'half1',
            'quarter' (oder 'quarter0'..'quarter3' je nach shared_quarters)
        """
        segs = self.splitter(x)
        result: dict[str, torch.Tensor] = {
            "full":  self.q_full.get_bloch_vectors(segs["full"]),
            "half0": self.q_half0.get_bloch_vectors(segs["half0"]),
            "half1": self.q_half1.get_bloch_vectors(segs["half1"]),
        }
        if self.shared_quarters:
            result["quarter"] = self.q_quarter.get_bloch_vectors(segs["quarter0"])
        else:
            for k in range(4):
                enc = getattr(self, f"q_quarter{k}")
                result[f"quarter{k}"] = enc.get_bloch_vectors(segs[f"quarter{k}"])
        return result

    # ── Parameter-Übersicht ────────────────────────────────────────────────────

    def param_summary(self) -> None:
        """Gibt eine Übersicht aller trainierbaren Parameter aus."""
        total   = sum(p.numel() for p in self.parameters() if p.requires_grad)

        def q_count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        if self.shared_quarters:
            q_quarter_params = q_count(self.q_quarter)
        else:
            q_quarter_params = sum(
                q_count(getattr(self, f"q_quarter{k}")) for k in range(4)
            )

        q_params = (
            q_quarter_params
            + q_count(self.q_half0)
            + q_count(self.q_half1)
            + q_count(self.q_full)
        )
        c_params = total - q_params

        shared_note = "(geteilt)" if self.shared_quarters else "(separat)"
        print("CantorQuantumClassifier — Parameter-Übersicht")
        print(f"  Quantum gesamt   : {q_params:>8,}")
        print(f"    q_quarter {shared_note}: {q_quarter_params:>6,}")
        print(f"    q_half0          : {q_count(self.q_half0):>6,}")
        print(f"    q_half1          : {q_count(self.q_half1):>6,}")
        print(f"    q_full           : {q_count(self.q_full):>6,}")
        print(f"  Klassisch gesamt : {c_params:>8,}")
        print(f"    Merger           : {q_count(self.merge_q01)+q_count(self.merge_q23)+q_count(self.merge_h0)+q_count(self.merge_h1)+q_count(self.merge_global):>6,}")
        print(f"    Head             : {q_count(self.head):>6,}")
        print(f"  {'─'*32}")
        print(f"  Total trainierbar: {total:>8,}")


if __name__ == "__main__":
    model = CantorQuantumClassifier(shared_quarters=True)
    model.param_summary()

    print()
    x = torch.randn(2, 1800)
    out = model(x)
    print(f"\nForward:  (2, 1800) → {tuple(out.shape)}")

    bv = model.get_bloch_vectors(x)
    print(f"Bloch:    (2, 1800) → {tuple(bv.shape)}  |r|={bv.norm(dim=1).tolist()}")

    all_bv = model.get_all_bloch_vectors(x)
    print("\nAlle Encoder Bloch-Vektoren:")
    for name, vec in all_bv.items():
        print(f"  {name:<12}  shape={tuple(vec.shape)}  |r|={vec.norm(dim=1).tolist()}")
