"""Diagnostic IR spectral windows for the 10 hard functional groups.

Wavenumber mapping assumes 1800-point spectra spanning 400–4000 cm⁻¹:
    index = round((wavenumber - 400) / 2)

Each entry maps a global class index to one or more (wn_low, wn_high) ranges
in cm⁻¹. Multiple windows are concatenated before projection into qubit angles.

Sources:
    Silverstein, Webster & Kiemle, "Spectrometric Identification of Organic
    Compounds", 8th ed.; NIST WebBook IR spectra references.
"""

from __future__ import annotations

SPEC_LEN = 1800
WN_MIN = 400
WN_MAX = 4000


def wn_to_idx(wn: float) -> int:
    """Convert a wavenumber (cm⁻¹) to a 1800-point spectrum array index."""
    step = (WN_MAX - WN_MIN) / (SPEC_LEN - 1)
    return int(round((wn - WN_MIN) / step))


# class_index -> list of (wn_low, wn_high) in cm⁻¹
SPECTRAL_WINDOWS: dict[int, list[tuple[int, int]]] = {
    0:  [(1740, 1870), (1000, 1300)],  # Acid anhydride: dual C=O + C-O-C
    1:  [(1770, 1830)],                # Acyl halide: C=O stretch
    10: [(1400, 1620)],                # Azo compound: N=N stretch
    13: [(1560, 1680), (1000, 1200)],  # Enamine: C=C-N + C-N stretch
    19: [(1580, 1700), (3100, 3400)],  # Hydrazone: C=N + N-H stretch
    21: [(1600, 1700)],                # Imine: C=N stretch
    27: [(2280, 2440), (700, 800)],    # Phosphine: P-H + P-C stretch
    28: [(580,  720)],                 # Sulfide: C-S stretch
    33: [(980,  1090)],                # Sulfoxide: S=O stretch
    34: [(2480, 2630), (1050, 1200)],  # Thial: S-H + C=S stretch
}


def get_window_indices(class_idx: int) -> list[tuple[int, int]]:
    """Return list of (idx_low, idx_high) array index pairs for a class."""
    windows = SPECTRAL_WINDOWS[class_idx]
    return [(wn_to_idx(lo), wn_to_idx(hi)) for lo, hi in windows]


def total_window_size(class_idx: int) -> int:
    """Total number of spectral points across all windows for a class."""
    return sum(hi - lo for lo, hi in get_window_indices(class_idx))
