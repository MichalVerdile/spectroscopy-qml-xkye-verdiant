"""Diagnostic IR spectral windows for the 10 specialist classes in Exp 10.2.2.

Wavenumber mapping: 1800-point spectrum spans 400–4000 cm⁻¹ (2 cm⁻¹/point).
    index = round((wavenumber - 400) / 2)
"""

from __future__ import annotations

SPEC_LEN = 1800
WN_MIN, WN_MAX = 400, 4000


def wn_to_idx(wn: float) -> int:
    step = (WN_MAX - WN_MIN) / (SPEC_LEN - 1)
    return int(round((wn - WN_MIN) / step))


# class_index -> list of (wn_low, wn_high) in cm⁻¹
SPECTRAL_WINDOWS: dict[int, list[tuple[int, int]]] = {
    1:  [(1770, 1830)],                 # Acyl halide:  C=O stretch
    13: [(1560, 1680), (1000, 1200)],   # Enamine:      C=C-N + C-N stretch
    14: [(1580, 1660), (3100, 3600)],   # Enol:         C=C + O-H stretch
    18: [(3100, 3400), (990, 1110)],    # Hydrazine:    N-H + N-N stretch
    19: [(1580, 1700), (3100, 3400)],   # Hydrazone:    C=N + N-H stretch
    21: [(1600, 1700)],                 # Imine:        C=N stretch
    24: [(1700, 1760)],                 # Ketone:       C=O stretch
    28: [(580,  720)],                  # Sulfide:      C-S stretch
    33: [(980,  1090)],                 # Sulfoxide:    S=O stretch
    35: [(1050, 1200), (3100, 3400)],   # Thioamide:    C=S + N-H stretch
}


def get_window_indices(class_idx: int) -> list[tuple[int, int]]:
    windows = SPECTRAL_WINDOWS[class_idx]
    return [(wn_to_idx(lo), wn_to_idx(hi)) for lo, hi in windows]


def total_window_size(class_idx: int) -> int:
    return sum(hi - lo for lo, hi in get_window_indices(class_idx))
