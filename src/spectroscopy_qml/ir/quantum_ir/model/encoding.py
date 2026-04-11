"""
Cantor-Split Encoding für IR-Spektren.

Teilt ein 1800-Punkte-IR-Spektrum hierarchisch in drei Cantor-Ebenen:

  Ebene 1  Volles Spektrum  (1800 Punkte)  → auf 2048 (2^11) gepaddet
  Ebene 2  Zwei Hälften     (900 Punkte)   → auf 1024 (2^10) gepaddet
  Ebene 3  Vier Viertel     (450 Punkte)   → auf  512 (2^9)  gepaddet

Jedes Segment wird L2-normalisiert und ist direkt als Eingangsvektor
für qml.AmplitudeEmbedding nutzbar.

Spektrale Bedeutung der Splits
--------------------------------
  Hälfte 0 (   0– 900 cm⁻¹) : C–C, C–H Streckschwingungen + Fingerprint
  Hälfte 1 ( 900–1800 cm⁻¹) : C=O, O–H, N–H Gruppen

  Viertel 0 (   0– 450 cm⁻¹) : Fingerprint Region
  Viertel 1 ( 450– 900 cm⁻¹) : C–H, C–C Biegeschwingungen
  Viertel 2 ( 900–1350 cm⁻¹) : C–O, C–N Streckschwingungen
  Viertel 3 (1350–1800 cm⁻¹) : C=O, N–H, O–H Streckschwingungen
"""

from __future__ import annotations

import torch

# ── Spektrale Konstanten ──────────────────────────────────────────────────────
INPUT_DIM    = 1800   # Original-Spektrumlänge

# Zieldimensionen für Amplituden-Einbettung (Potenzen von 2)
DIM_FULL     = 2048   # 2^11 — volles Spektrum
DIM_HALF     = 1024   # 2^10 — jede Hälfte
DIM_QUARTER  =  512   # 2^9  — jedes Viertel

# Segmentgrenzen (Indizes in 1800-Punkte-Spektrum)
HALF_LEN     = INPUT_DIM // 2   # 900
QUARTER_LEN  = INPUT_DIM // 4   # 450


def pad_and_normalise(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Füllt ein Batch von Spektralsegmenten mit Nullen auf target_dim auf
    und L2-normalisiert jede Zeile.

    Args:
        x:          (batch, segment_len) float32 — Spektralsegment
        target_dim: Ganzzahl als Potenz von 2 (Ziellänge für AmplitudeEmbedding)

    Returns:
        (batch, target_dim) L2-normalisierter Tensor
    """
    batch, seg_len = x.shape
    if seg_len > target_dim:
        raise ValueError(
            f"Segmentlänge {seg_len} überschreitet target_dim {target_dim}"
        )
    out = torch.zeros(batch, target_dim, device=x.device, dtype=x.dtype)
    out[:, :seg_len] = x
    norms = out.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return out / norms


class CantorSplit:
    """
    Teilt einen Batch von IR-Spektren in alle Cantor-Hierarchie-Segmente auf.

    Alle Ausgaben sind L2-normalisiert und direkt für qml.AmplitudeEmbedding
    einsetzbar.

    Beispiel::

        splitter = CantorSplit()
        segs = splitter(x)          # x: (batch, 1800)

        segs["full"]      → (batch, 2048)  # 2^11 — Ebene 1
        segs["half0"]     → (batch, 1024)  # 2^10 — Ebene 2, links
        segs["half1"]     → (batch, 1024)  # 2^10 — Ebene 2, rechts
        segs["quarter0"]  → (batch,  512)  # 2^9  — Ebene 3, Viertel 0
        segs["quarter1"]  → (batch,  512)  # 2^9  — Ebene 3, Viertel 1
        segs["quarter2"]  → (batch,  512)  # 2^9  — Ebene 3, Viertel 2
        segs["quarter3"]  → (batch,  512)  # 2^9  — Ebene 3, Viertel 3
    """

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, 1800) SNV-normalisiertes IR-Spektrum

        Returns:
            dict mit Schlüsseln 'full', 'half0', 'half1',
            'quarter0' .. 'quarter3' — alle L2-normalisiert
        """
        q = QUARTER_LEN
        return {
            # Ebene 1 — volles Spektrum
            "full":     pad_and_normalise(x, DIM_FULL),

            # Ebene 2 — zwei Hälften
            "half0":    pad_and_normalise(x[:, :HALF_LEN],   DIM_HALF),
            "half1":    pad_and_normalise(x[:, HALF_LEN:],   DIM_HALF),

            # Ebene 3 — vier Viertel
            "quarter0": pad_and_normalise(x[:, 0*q : 1*q],  DIM_QUARTER),
            "quarter1": pad_and_normalise(x[:, 1*q : 2*q],  DIM_QUARTER),
            "quarter2": pad_and_normalise(x[:, 2*q : 3*q],  DIM_QUARTER),
            "quarter3": pad_and_normalise(x[:, 3*q : 4*q],  DIM_QUARTER),
        }

    # ── Hilfsfunktionen ────────────────────────────────────────────────────────

    @staticmethod
    def segment_names() -> list[str]:
        """Gibt alle Segmentnamen in Hierarchie-Reihenfolge zurück."""
        return [
            "full",
            "half0", "half1",
            "quarter0", "quarter1", "quarter2", "quarter3",
        ]

    @staticmethod
    def spectral_ranges() -> dict[str, str]:
        """Gibt die spektrale Bedeutung jedes Segments zurück."""
        return {
            "full":     "0–1800 cm⁻¹  (volles Spektrum)",
            "half0":    "0–900 cm⁻¹   (C–C, C–H Streckschwingungen + Fingerprint)",
            "half1":    "900–1800 cm⁻¹ (C=O, O–H, N–H Gruppen)",
            "quarter0": "0–450 cm⁻¹   (Fingerprint Region)",
            "quarter1": "450–900 cm⁻¹  (C–H, C–C Biegeschwingungen)",
            "quarter2": "900–1350 cm⁻¹ (C–O, C–N Streckschwingungen)",
            "quarter3": "1350–1800 cm⁻¹ (C=O, N–H, O–H Streckschwingungen)",
        }


if __name__ == "__main__":
    splitter = CantorSplit()
    x = torch.randn(4, INPUT_DIM)
    segs = splitter(x)

    print("CantorSplit — Segmentübersicht")
    print(f"{'Segment':<12}  {'Form':<16}  {'Norm (erwartet ≈1.0)'}")
    print("─" * 55)
    for name, tensor in segs.items():
        norms = tensor.norm(dim=1)
        print(f"{name:<12}  {str(tuple(tensor.shape)):<16}  "
              f"min={norms.min():.4f}  max={norms.max():.4f}")
