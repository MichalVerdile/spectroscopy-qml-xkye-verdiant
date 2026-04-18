"""
Noise comparison: Exp 10.1 (raw finite-diff) vs Exp 10.2 (Lorentzian-smoothed).

Generates side-by-side plots showing whether Lorentzian smoothing actually
reduces the derivative noise that plagued Exp 10.1.

Usage:
    python scripts/visualize_lorentz_noise_comparison.py
    python scripts/visualize_lorentz_noise_comparison.py --n-samples 100 --n-display 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.mps_encoder.data_loader import interpolate_spectrum

WAVENUMBER_MIN = 400.0
WAVENUMBER_MAX = 4000.0
SPECTRUM_LENGTH = 1800


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_wavenumber_axis(n: int = SPECTRUM_LENGTH) -> np.ndarray:
    return np.linspace(WAVENUMBER_MIN, WAVENUMBER_MAX, n)


def load_raw_spectra(data_dir: Path, n_samples: int, seed: int) -> np.ndarray:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    rng = np.random.default_rng(seed)
    n_files = len(parquet_files)
    base_quota = max(1, n_samples // n_files)
    file_order = rng.permutation(n_files)
    all_spectra: list[np.ndarray] = []
    for fi in file_order:
        if len(all_spectra) >= n_samples:
            break
        df = pd.read_parquet(parquet_files[fi], columns=["ir_spectra"])
        df = df[df["ir_spectra"].notna()].reset_index(drop=True)
        quota = min(base_quota, n_samples - len(all_spectra), len(df))
        for ci in rng.choice(len(df), size=quota, replace=False):
            all_spectra.append(interpolate_spectrum(np.asarray(df.iloc[ci]["ir_spectra"]), SPECTRUM_LENGTH))
    spectra = np.stack(all_spectra)
    print(f"Loaded {len(spectra)} spectra")
    return spectra


def maxabs_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scale = np.abs(arr).max(axis=-1, keepdims=True).clip(min=eps)
    return arr / scale


# ── Feature maps ──────────────────────────────────────────────────────────────

def first_derivative(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x)
    d[..., 0] = x[..., 1] - x[..., 0]
    d[..., -1] = x[..., -1] - x[..., -2]
    d[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2])
    return d


def second_derivative(x: np.ndarray) -> np.ndarray:
    d2 = np.zeros_like(x)
    d2[..., 1:-1] = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
    d2[..., 0] = d2[..., 1]
    d2[..., -1] = d2[..., -2]
    return d2


def lorentz_kernel(gamma: float, half_width: int) -> np.ndarray:
    positions = np.arange(-half_width, half_width + 1, dtype=np.float32)
    k = 1.0 / (1.0 + (positions / gamma) ** 2)
    return k / k.sum()


def lorentz_smooth(spectra: np.ndarray, gamma: float = 3.0, half_width: int = 15) -> np.ndarray:
    kernel = lorentz_kernel(gamma, half_width)
    return np.stack([convolve1d(s, kernel, mode="reflect") for s in spectra])


def voigt_kernel(gamma_l: float, gamma_g: float, eta: float, half_width: int) -> np.ndarray:
    positions = np.arange(-half_width, half_width + 1, dtype=np.float32)
    lorentz = 1.0 / (1.0 + (positions / gamma_l) ** 2)
    lorentz /= lorentz.sum()
    gauss = np.exp(-0.5 * (positions / gamma_g) ** 2)
    gauss /= gauss.sum()
    k = eta * lorentz + (1.0 - eta) * gauss
    return k / k.sum()


def voigt_smooth(spectra: np.ndarray, gamma_l: float = 3.0, gamma_g: float = 2.0,
                 eta: float = 0.5, half_width: int = 20) -> np.ndarray:
    kernel = voigt_kernel(gamma_l, gamma_g, eta, half_width)
    return np.stack([convolve1d(s, kernel, mode="reflect") for s in spectra])


def channels_exp10_1(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw / raw first-diff / raw second-diff (Exp 10.1)."""
    raw = maxabs_normalize(spectra)
    d1  = maxabs_normalize(first_derivative(spectra))
    d2  = maxabs_normalize(second_derivative(spectra))
    return raw, d1, d2


def channels_exp10_2(spectra: np.ndarray, gamma: float = 3.0, half_width: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw / Lorentz-smoothed first-diff / Lorentz-smoothed second-diff (Exp 10.2)."""
    raw      = maxabs_normalize(spectra)
    smoothed = lorentz_smooth(spectra, gamma, half_width)
    d1       = maxabs_normalize(first_derivative(smoothed))
    d2       = maxabs_normalize(second_derivative(smoothed))
    return raw, d1, d2


def channels_exp10_3(spectra: np.ndarray, gamma_l: float = 3.0, gamma_g: float = 2.0,
                     eta: float = 0.5, half_width: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw / Voigt-smoothed first-diff / Voigt-smoothed second-diff (Exp 10.3)."""
    raw      = maxabs_normalize(spectra)
    smoothed = voigt_smooth(spectra, gamma_l, gamma_g, eta, half_width)
    d1       = maxabs_normalize(first_derivative(smoothed))
    d2       = maxabs_normalize(second_derivative(smoothed))
    return raw, d1, d2


def sg_noise_std(data: np.ndarray) -> np.ndarray:
    """Per-sample SG residual std — proxy for high-frequency noise."""
    sg = np.stack([savgol_filter(s, window_length=15, polyorder=3) for s in data])
    return (data - sg).std(axis=-1)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_overlay_comparison(
    spectra: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
    n_display: int,
    gamma: float,
    half_width: int,
) -> None:
    """6-panel overlay: each channel for Exp 10.1 (left) and Exp 10.2 (right)."""
    n = min(n_display, len(spectra))
    raw_101, d1_101, d2_101 = channels_exp10_1(spectra[:n])
    raw_102, d1_102, d2_102 = channels_exp10_2(spectra[:n], gamma, half_width)

    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False)
    fig.suptitle(
        f"Exp 10.1 (raw finite-diff)  vs  Exp 10.2 (Lorentzian-smoothed, γ={gamma})\n"
        "Max-abs normalised channels — what the TTN actually receives",
        fontsize=13, y=0.995,
    )

    pairs = [
        (raw_101, raw_102, "Ch 0: Raw (same for both)"),
        (d1_101,  d1_102,  "Ch 1: 1st Derivative"),
        (d2_101,  d2_102,  "Ch 2: 2nd Derivative"),
    ]
    titles_left  = ["Exp 10.1 — raw", "Exp 10.1 — raw d1", "Exp 10.1 — raw d2"]
    titles_right = ["Exp 10.2 — raw", f"Exp 10.2 — Lorentz d1 (γ={gamma})", f"Exp 10.2 — Lorentz d2 (γ={gamma})"]

    for row, (left, right, _) in enumerate(pairs):
        for col, (data, title) in enumerate([(left, titles_left[row]), (right, titles_right[row])]):
            ax = axes[row, col]
            for s, c in zip(data, cmap):
                ax.plot(wn, s, color=c, alpha=0.55, linewidth=0.7)
            ax.set_title(title, fontsize=10)
            ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
            ax.set_ylim(-1.05, 1.05)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.2)
            if row == 2:
                ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
            if col == 0:
                ax.set_ylabel("Normalised value", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_noise_reduction(
    spectra: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
    gamma: float,
    half_width: int,
    gamma_l: float = 3.0,
    gamma_g: float = 2.0,
    voigt_eta: float = 0.5,
    voigt_half_width: int = 20,
) -> None:
    """Quantitative noise comparison: Exp 10.1 / 10.2 / 10.3 side-by-side."""
    raw_101, d1_101, d2_101 = channels_exp10_1(spectra)
    raw_102, d1_102, d2_102 = channels_exp10_2(spectra, gamma, half_width)
    raw_103, d1_103, d2_103 = channels_exp10_3(spectra, gamma_l, gamma_g, voigt_eta, voigt_half_width)

    noise_101 = [sg_noise_std(raw_101), sg_noise_std(d1_101), sg_noise_std(d2_101)]
    noise_102 = [sg_noise_std(raw_102), sg_noise_std(d1_102), sg_noise_std(d2_102)]
    noise_103 = [sg_noise_std(raw_103), sg_noise_std(d1_103), sg_noise_std(d2_103)]

    ch_labels = ["Raw", "1st Deriv.", "2nd Deriv."]
    colors = {
        "101": ["steelblue",      "darkorange",  "seagreen"],
        "102": ["cornflowerblue", "gold",         "mediumseagreen"],
        "103": ["mediumpurple",   "tomato",       "peru"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Noise (SG residual std) — Exp 10.1 vs 10.2 (Lorentz γ={gamma}) vs 10.3 (Voigt η={voigt_eta})\n"
        "Lower = less high-frequency noise reaching the TTN",
        fontsize=12,
    )

    for col, (label, n101, n102, n103) in enumerate(zip(ch_labels, noise_101, noise_102, noise_103)):
        ax = axes[col]
        vp = ax.violinplot([n101, n102, n103], positions=[1, 2, 3], showmedians=True)
        for body, c in zip(vp["bodies"], [colors["101"][col], colors["102"][col], colors["103"][col]]):
            body.set_facecolor(c)
            body.set_alpha(0.7)

        med101 = float(np.median(n101))
        med102 = float(np.median(n102))
        med103 = float(np.median(n103))
        red102 = (1.0 - med102 / (med101 + 1e-12)) * 100
        red103 = (1.0 - med103 / (med101 + 1e-12)) * 100

        ax.text(1, med101, f" {med101:.4f}", va="bottom", fontsize=8)
        ax.text(2, med102, f" {med102:.4f}\n({red102:+.1f}%)", va="bottom", fontsize=8)
        ax.text(3, med103, f" {med103:.4f}\n({red103:+.1f}%)", va="bottom", fontsize=8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["10.1\n(raw FD)", f"10.2\n(Lorentz)", "10.3\n(Voigt)"], fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("SG residual std (norm. units)", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pointwise_noise_comparison(
    spectra: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
    gamma: float,
    half_width: int,
    gamma_l: float = 3.0,
    gamma_g: float = 2.0,
    voigt_eta: float = 0.5,
    voigt_half_width: int = 20,
) -> None:
    """Pointwise noise std along wavenumber — Exp 10.1 / 10.2 / 10.3."""
    raw_101, d1_101, d2_101 = channels_exp10_1(spectra)
    _,       d1_102, d2_102 = channels_exp10_2(spectra, gamma, half_width)
    _,       d1_103, d2_103 = channels_exp10_3(spectra, gamma_l, gamma_g, voigt_eta, voigt_half_width)

    def ptwise_noise(data: np.ndarray) -> np.ndarray:
        sg = np.stack([savgol_filter(s, window_length=15, polyorder=3) for s in data])
        return (data - sg).std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Pointwise Noise Std — Exp 10.1 vs 10.2 (Lorentz γ={gamma}) vs 10.3 (Voigt η={voigt_eta})\n"
        "Blau = Rausch-Reduktion vs 10.1  |  Rot = Rausch-Zunahme vs 10.1",
        fontsize=12,
    )

    for ax, d101, d102, d103, ch_name in [
        (axes[0], d1_101, d1_102, d1_103, "1st Derivative"),
        (axes[1], d2_101, d2_102, d2_103, "2nd Derivative"),
    ]:
        n101 = ptwise_noise(d101)
        n102 = ptwise_noise(d102)
        n103 = ptwise_noise(d103)

        ax.plot(wn, n101, color="darkorange",    linewidth=1.0, label="Exp 10.1 (raw FD)",        zorder=3)
        ax.plot(wn, n102, color="steelblue",     linewidth=0.9, label=f"Exp 10.2 (Lorentz γ={gamma})", zorder=4)
        ax.plot(wn, n103, color="mediumpurple",  linewidth=0.9, label=f"Exp 10.3 (Voigt η={voigt_eta})", zorder=5)
        ax.fill_between(wn, n102, n101, where=(n101 >= n102), alpha=0.15, color="steelblue")
        ax.fill_between(wn, n103, n101, where=(n101 >= n103), alpha=0.15, color="mediumpurple")
        ax.set_title(f"{ch_name} — pointwise noise std", fontsize=11)
        ax.set_ylabel("Std dev (norm. units)", fontsize=9)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_single_spectrum_zoom(
    spectra: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
    gamma: float,
    half_width: int,
    sample_idx: int = 0,
) -> None:
    """Zoom into one spectrum to clearly see smoothing effect on the derivatives."""
    s = spectra[sample_idx:sample_idx + 1]
    raw_101, d1_101, d2_101 = channels_exp10_1(s)
    _,       d1_102, d2_102 = channels_exp10_2(s, gamma, half_width)

    # Zoom to fingerprint region 600-1800 cm⁻¹
    mask = (wn >= 600) & (wn <= 1800)
    wn_z = wn[mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Single Spectrum Zoom (fingerprint 600–1800 cm⁻¹) — sample #{sample_idx}\n"
        "Exp 10.1 (raw) vs Exp 10.2 (Lorentz γ={gamma})",
        fontsize=13,
    )

    for row, (d101, d102, name, c101, c102) in enumerate([
        (d1_101[0], d1_102[0], "1st Derivative", "darkorange", "steelblue"),
        (d2_101[0], d2_102[0], "2nd Derivative", "seagreen",   "mediumorchid"),
    ]):
        axes[row, 0].plot(wn_z, d101[mask], color=c101, linewidth=0.9, label="Exp 10.1 (raw)")
        axes[row, 0].set_title(f"{name} — Exp 10.1 (raw finite-diff)", fontsize=10)
        axes[row, 0].axhline(0, color="k", linewidth=0.4, linestyle="--")
        axes[row, 0].invert_xaxis()
        axes[row, 0].grid(True, alpha=0.2)
        axes[row, 0].set_ylabel("Normalised value", fontsize=9)

        axes[row, 1].plot(wn_z, d102[mask], color=c102, linewidth=0.9, label=f"Exp 10.2 (Lorentz γ={gamma})")
        axes[row, 1].set_title(f"{name} — Exp 10.2 (Lorentzian-smoothed)", fontsize=10)
        axes[row, 1].axhline(0, color="k", linewidth=0.4, linestyle="--")
        axes[row, 1].invert_xaxis()
        axes[row, 1].grid(True, alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Noise comparison: Exp 10.1 vs Exp 10.2")
    parser.add_argument("--data-dir",       type=Path,  default=Path("data/raw"))
    parser.add_argument("--n-samples",      type=int,   default=50)
    parser.add_argument("--n-display",      type=int,   default=10)
    parser.add_argument("--out-dir",        type=Path,  default=Path("reports/lorentz_noise_comparison"))
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--lorentz-gamma",       type=float, default=3.0)
    parser.add_argument("--kernel-half-width",   type=int,   default=15)
    parser.add_argument("--voigt-gamma-l",        type=float, default=3.0)
    parser.add_argument("--voigt-gamma-g",        type=float, default=2.0)
    parser.add_argument("--voigt-eta",            type=float, default=0.5)
    parser.add_argument("--voigt-kernel-half-width", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wn = make_wavenumber_axis()

    print(f"Loading spectra (n={args.n_samples})...")
    spectra = load_raw_spectra(args.data_dir, args.n_samples, args.seed)

    print("\nGenerating plots...")

    plot_overlay_comparison(
        spectra, wn,
        out_path=args.out_dir / "01_overlay_exp101_vs_exp102.png",
        n_display=args.n_display,
        gamma=args.lorentz_gamma,
        half_width=args.kernel_half_width,
    )
    plot_noise_reduction(
        spectra, wn,
        out_path=args.out_dir / "02_noise_reduction_violin.png",
        gamma=args.lorentz_gamma,
        half_width=args.kernel_half_width,
        gamma_l=args.voigt_gamma_l,
        gamma_g=args.voigt_gamma_g,
        voigt_eta=args.voigt_eta,
        voigt_half_width=args.voigt_kernel_half_width,
    )
    plot_pointwise_noise_comparison(
        spectra, wn,
        out_path=args.out_dir / "03_pointwise_noise_comparison.png",
        gamma=args.lorentz_gamma,
        half_width=args.kernel_half_width,
        gamma_l=args.voigt_gamma_l,
        gamma_g=args.voigt_gamma_g,
        voigt_eta=args.voigt_eta,
        voigt_half_width=args.voigt_kernel_half_width,
    )
    plot_single_spectrum_zoom(
        spectra, wn,
        out_path=args.out_dir / "04_single_spectrum_zoom.png",
        gamma=args.lorentz_gamma,
        half_width=args.kernel_half_width,
        sample_idx=0,
    )

    print(f"\nAlle Plots gespeichert in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
