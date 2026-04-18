"""
Noise comparison: analytical derivative kernels (Lorentzian / Voigt / SG)
at different gamma values vs raw finite-difference (Exp 10.1).

Uses the exact same conv1d kernels as the models (Exp 10.2 / 10.3 / 10.4),
so the plots reflect what the TTN actually receives.

The report now includes two complementary views:
1. Noise proxy: SG residual std (lower is better)
2. Signal preservation: correlation to a lightly smoothed raw-FD reference
   (higher is better)

Usage:
    python scripts/visualize_analytical_noise_comparison.py
    python scripts/visualize_analytical_noise_comparison.py --n-samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import savgol_coeffs, savgol_filter

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.mps_encoder.data_loader import interpolate_spectrum

WAVENUMBER_MIN = 400.0
WAVENUMBER_MAX = 4000.0
SPECTRUM_LENGTH = 1800


# ── Data loading ──────────────────────────────────────────────────────────────

def make_wavenumber_axis() -> np.ndarray:
    return np.linspace(WAVENUMBER_MIN, WAVENUMBER_MAX, SPECTRUM_LENGTH)


def load_raw_spectra(data_dir: Path, n_samples: int, seed: int) -> torch.Tensor:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    rng = np.random.default_rng(seed)
    base_quota = max(1, n_samples // len(parquet_files))
    all_spectra: list[np.ndarray] = []
    for fi in rng.permutation(len(parquet_files)):
        if len(all_spectra) >= n_samples:
            break
        df = pd.read_parquet(parquet_files[fi], columns=["ir_spectra"])
        df = df[df["ir_spectra"].notna()].reset_index(drop=True)
        quota = min(base_quota, n_samples - len(all_spectra), len(df))
        for ci in rng.choice(len(df), size=quota, replace=False):
            all_spectra.append(
                interpolate_spectrum(np.asarray(df.iloc[ci]["ir_spectra"]), SPECTRUM_LENGTH)
            )
    print(f"Loaded {len(all_spectra)} spectra")
    return torch.tensor(np.stack(all_spectra), dtype=torch.float32)


# ── Kernel builders (match model exactly) ────────────────────────────────────

def make_lorentz_kernels(gamma: float, half_width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.arange(-half_width, half_width + 1, dtype=torch.float32)
    u = pos / gamma
    l = 1.0 / (1.0 + u ** 2)
    k0 = (l / l.sum()).view(1, 1, -1)
    k1 = (-2.0 * u / gamma * l ** 2).flip(0).view(1, 1, -1)  # flip for cross-corr → conv
    k2 = (2.0 / gamma ** 2 * (3.0 * u ** 2 - 1.0) / (1.0 + u ** 2) ** 3).view(1, 1, -1)
    return k0, k1, k2


def make_voigt_kernels(
    gamma_l: float, gamma_g: float, eta: float, half_width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.arange(-half_width, half_width + 1, dtype=torch.float32)
    ul, ug = pos / gamma_l, pos / gamma_g
    l = 1.0 / (1.0 + ul ** 2)
    g = torch.exp(-0.5 * ug ** 2)
    l1 = -2.0 * ul / gamma_l * l ** 2
    l2 = 2.0 / gamma_l ** 2 * (3.0 * ul ** 2 - 1.0) / (1.0 + ul ** 2) ** 3
    g1 = -ug / gamma_g * g
    g2 = (ug ** 2 / gamma_g ** 2 - 1.0 / gamma_g ** 2) * g
    l_sum, g_sum = l.sum(), g.sum()
    k0 = eta * (l / l_sum) + (1.0 - eta) * (g / g_sum)
    k0 = (k0 / k0.sum()).view(1, 1, -1)
    k1 = (eta * (l1 / l_sum) + (1.0 - eta) * (g1 / g_sum)).flip(0).view(1, 1, -1)
    k2 = (eta * (l2 / l_sum) + (1.0 - eta) * (g2 / g_sum)).view(1, 1, -1)
    return k0, k1, k2


def make_sg_kernels(window_length: int, polyorder: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    half = window_length // 2
    k0 = torch.from_numpy(savgol_coeffs(window_length, polyorder, deriv=0).astype(np.float32)).view(1, 1, -1)
    k1c = savgol_coeffs(window_length, polyorder, deriv=1).astype(np.float32)
    k1 = torch.from_numpy(k1c[::-1].copy()).view(1, 1, -1)
    k2 = torch.from_numpy(savgol_coeffs(window_length, polyorder, deriv=2).astype(np.float32)).view(1, 1, -1)
    return k0, k1, k2


def apply_kernel(x: torch.Tensor, kernel: torch.Tensor, half_width: int) -> torch.Tensor:
    padded = F.pad(x.unsqueeze(1), (half_width, half_width), mode="reflect")
    return F.conv1d(padded, kernel).squeeze(1)


def maxabs_norm(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return t / scale


def raw_fd_channels(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw finite-difference (Exp 10.1 / SpectralDerivativeFeatureMap)."""
    d1 = torch.zeros_like(x)
    d1[:, 0] = x[:, 1] - x[:, 0]
    d1[:, -1] = x[:, -1] - x[:, -2]
    d1[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    d2 = torch.zeros_like(x)
    d2[:, 1:-1] = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    d2[:, 0] = d2[:, 1]
    d2[:, -1] = d2[:, -2]
    return maxabs_norm(x), maxabs_norm(d1), maxabs_norm(d2)


def analytical_channels(
    x: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, half: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ch0 = maxabs_norm(apply_kernel(x, k0, half))
    ch1 = maxabs_norm(apply_kernel(x, k1, half))
    ch2 = maxabs_norm(apply_kernel(x, k2, half))
    return ch0, ch1, ch2


# ── Noise metric ──────────────────────────────────────────────────────────────

def sg_residual_std(data: torch.Tensor) -> np.ndarray:
    """High-frequency noise proxy: std of (signal − 15-pt SG smooth) per sample."""
    d = data.numpy()
    sg = np.stack([savgol_filter(s, window_length=15, polyorder=3) for s in d])
    return (d - sg).std(axis=-1)


def ptwise_noise_std(data: torch.Tensor) -> np.ndarray:
    d = data.numpy()
    sg = np.stack([savgol_filter(s, window_length=15, polyorder=3) for s in d])
    return (d - sg).std(axis=0)


def smooth_reference(data: torch.Tensor, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Lightly denoise a channel to obtain a structure-preservation reference."""
    d = data.numpy()
    return np.stack([savgol_filter(s, window_length=window_length, polyorder=polyorder) for s in d])


def median_structure_corr(data: torch.Tensor, reference: np.ndarray, eps: float = 1e-8) -> float:
    """Median Pearson correlation to a lightly smoothed raw-FD reference."""
    arr = data.numpy()
    corrs: list[float] = []
    for sample, ref in zip(arr, reference, strict=True):
        sample_centered = sample - sample.mean()
        ref_centered = ref - ref.mean()
        denom = np.sqrt(
            np.sum(sample_centered ** 2) * np.sum(ref_centered ** 2)
        )
        corrs.append(float(np.sum(sample_centered * ref_centered) / max(denom, eps)))
    return float(np.median(np.asarray(corrs)))


# ── Plot helpers ──────────────────────────────────────────────────────────────

PALETTE = {
    "raw_fd":       ("#e07b39", "--"),   # orange dashed
    "lorentz_3":    ("#4878cf", "-"),    # blue solid
    "lorentz_10":   ("#1d3557", "-"),    # dark blue solid
    "voigt_3":      ("#a35bd4", "-"),    # purple solid
    "voigt_10":     ("#5a189a", "-"),    # dark purple solid
    "sg":           ("#2a9d8f", "-"),    # teal solid
}

LABELS = {
    "raw_fd":     "Raw FD (10.1)",
    "lorentz_3":  "Lorentz γ=3 (10.2)",
    "lorentz_10": "Lorentz γ=10 (10.2)",
    "voigt_3":    "Voigt γ_L=3 (10.3)",
    "voigt_10":   "Voigt γ_L=10 (10.3)",
    "sg":         "SG w=11 p=3 (10.4)",
}


# ── Plot 1: violin noise per channel ─────────────────────────────────────────

def plot_violin(channels_dict: dict, out_path: Path) -> None:
    ch_names = ["1st Derivative", "2nd Derivative"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Noise (SG residual std) per channel — analytical kernels vs raw FD\n"
        "Lower = fewer high-frequency artefacts reaching the TTN",
        fontsize=12,
    )
    keys = list(channels_dict.keys())
    colors = [PALETTE[k][0] for k in keys]
    x_labels = [LABELS[k] for k in keys]

    for col, ch_idx in enumerate([1, 2]):
        ax = axes[col]
        data_list = [sg_residual_std(channels_dict[k][ch_idx]) for k in keys]
        vp = ax.violinplot(data_list, positions=range(len(keys)), showmedians=True, widths=0.7)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.65)
        for i, (d, k) in enumerate(zip(data_list, keys)):
            med = float(np.median(d))
            ax.text(i, med, f" {med:.4f}", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(ch_names[col], fontsize=11)
        ax.set_ylabel("SG residual std (norm. units)", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 2: pointwise noise along wavenumber ──────────────────────────────────

def plot_pointwise(channels_dict: dict, wn: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(
        "Pointwise noise std along wavenumber — analytical kernels vs raw FD\n"
        "Lower = fewer high-frequency artefacts",
        fontsize=12,
    )
    titles = ["1st Derivative", "2nd Derivative"]
    for ax, ch_idx, title in zip(axes, [1, 2], titles):
        for k, (ch0, ch1, ch2) in channels_dict.items():
            ch = [ch0, ch1, ch2][ch_idx]
            noise = ptwise_noise_std(ch)
            c, ls = PALETTE[k]
            ax.plot(wn, noise, color=c, linestyle=ls, linewidth=0.9, label=LABELS[k], alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Pointwise noise std", fontsize=9)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 3: single spectrum overlay (zoom fingerprint region) ─────────────────

def plot_single_spectrum(
    spectra: torch.Tensor, channels_dict: dict, wn: np.ndarray, out_path: Path, sample_idx: int = 0
) -> None:
    mask = (wn >= 600) & (wn <= 1800)
    wn_z = wn[mask]
    keys_to_show = ["raw_fd", "lorentz_3", "lorentz_10", "voigt_3", "voigt_10", "sg"]

    fig, axes = plt.subplots(2, len(keys_to_show), figsize=(22, 8), sharex=True)
    fig.suptitle(
        f"Single spectrum #{sample_idx} — fingerprint region 600–1800 cm⁻¹\n"
        "Row 1: 1st Derivative   |   Row 2: 2nd Derivative",
        fontsize=12,
    )
    for col, k in enumerate(keys_to_show):
        ch0, ch1, ch2 = channels_dict[k]
        for row, (ch, ch_name) in enumerate([(ch1, "d1"), (ch2, "d2")]):
            ax = axes[row, col]
            signal = ch[sample_idx].numpy()
            c, ls = PALETTE[k]
            ax.plot(wn_z, signal[mask], color=c, linestyle=ls, linewidth=0.85)
            ax.axhline(0, color="k", linewidth=0.35, linestyle=":")
            ax.set_title(f"{LABELS[k]}\n({ch_name})", fontsize=8.5)
            ax.set_ylim(-1.05, 1.05)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.2)
            if row == 1:
                ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Normalised value", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 4: noise reduction factor bar chart ──────────────────────────────────

def plot_noise_reduction_bars(channels_dict: dict, out_path: Path) -> None:
    ref_noise_d1 = float(np.median(sg_residual_std(channels_dict["raw_fd"][1])))
    ref_noise_d2 = float(np.median(sg_residual_std(channels_dict["raw_fd"][2])))

    keys = [k for k in channels_dict if k != "raw_fd"]
    d1_ratios = []
    d2_ratios = []
    for k in keys:
        med_d1 = float(np.median(sg_residual_std(channels_dict[k][1])))
        med_d2 = float(np.median(sg_residual_std(channels_dict[k][2])))
        d1_ratios.append(med_d1 / ref_noise_d1)
        d2_ratios.append(med_d2 / ref_noise_d2)

    x = np.arange(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width / 2, d1_ratios, width, label="1st Derivative",
                   color=[PALETTE[k][0] for k in keys], alpha=0.75)
    bars2 = ax.bar(x + width / 2, d2_ratios, width, label="2nd Derivative",
                   color=[PALETTE[k][0] for k in keys], alpha=0.45, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="darkorange", linestyle="--", linewidth=1.2, label="Raw FD baseline (=1.0)")
    ax.axhline(0.0, color="k", linewidth=0.4)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.2f}×", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keys], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Noise ratio vs Raw FD (lower = better)", fontsize=10)
    ax.set_title(
        "Noise reduction factor per analytical kernel (median SG residual std)\n"
        "1.0 = same as raw FD  |  <1.0 = less noise  |  >1.0 = more noise",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(d1_ratios), max(d2_ratios)) * 1.25)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 5: signal preservation vs lightly smoothed raw-FD reference ─────────

def plot_signal_preservation_bars(
    channels_dict: dict,
    raw_refs: tuple[np.ndarray, np.ndarray],
    out_path: Path,
) -> None:
    ref_d1, ref_d2 = raw_refs
    keys = [k for k in channels_dict if k != "raw_fd"]
    d1_corrs = [median_structure_corr(channels_dict[k][1], ref_d1) for k in keys]
    d2_corrs = [median_structure_corr(channels_dict[k][2], ref_d2) for k in keys]

    x = np.arange(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(
        x - width / 2,
        d1_corrs,
        width,
        label="1st Derivative",
        color=[PALETTE[k][0] for k in keys],
        alpha=0.75,
    )
    bars2 = ax.bar(
        x + width / 2,
        d2_corrs,
        width,
        label="2nd Derivative",
        color=[PALETTE[k][0] for k in keys],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keys], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Median Pearson r vs smoothed raw FD", fontsize=10)
    ax.set_title(
        "Signal preservation per analytical kernel\n"
        "Reference = lightly smoothed raw-FD derivative channel | higher = closer structure",
        fontsize=11,
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical kernel noise comparison")
    parser.add_argument("--data-dir",    type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir",     type=Path, default=Path("reports/analytical_noise_comparison"))
    parser.add_argument("--n-samples",   type=int,  default=80)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--sample-idx",  type=int,  default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wn = make_wavenumber_axis()

    print(f"Loading {args.n_samples} spectra...")
    spectra = load_raw_spectra(args.data_dir, args.n_samples, args.seed)

    print("Computing channels...")
    with torch.no_grad():
        k0_l3,  k1_l3,  k2_l3  = make_lorentz_kernels(gamma=3.0,  half_width=15)
        k0_l10, k1_l10, k2_l10 = make_lorentz_kernels(gamma=10.0, half_width=50)
        k0_v3,  k1_v3,  k2_v3  = make_voigt_kernels(gamma_l=3.0,  gamma_g=2.0, eta=0.5, half_width=20)
        k0_v10, k1_v10, k2_v10 = make_voigt_kernels(gamma_l=10.0, gamma_g=6.0, eta=0.5, half_width=50)
        k0_sg,  k1_sg,  k2_sg  = make_sg_kernels(window_length=11, polyorder=3)

        channels: dict = {
            "raw_fd":     raw_fd_channels(spectra),
            "lorentz_3":  analytical_channels(spectra, k0_l3,  k1_l3,  k2_l3,  15),
            "lorentz_10": analytical_channels(spectra, k0_l10, k1_l10, k2_l10, 50),
            "voigt_3":    analytical_channels(spectra, k0_v3,  k1_v3,  k2_v3,  20),
            "voigt_10":   analytical_channels(spectra, k0_v10, k1_v10, k2_v10, 50),
            "sg":         analytical_channels(spectra, k0_sg,  k1_sg,  k2_sg,  5),
        }
        raw_ref_d1 = smooth_reference(channels["raw_fd"][1])
        raw_ref_d2 = smooth_reference(channels["raw_fd"][2])

    print("\nGenerating plots...")
    plot_violin(channels, args.out_dir / "01_noise_violin.png")
    plot_pointwise(channels, wn, args.out_dir / "02_pointwise_noise.png")
    plot_single_spectrum(spectra, channels, wn, args.out_dir / "03_single_spectrum_zoom.png", args.sample_idx)
    plot_noise_reduction_bars(channels, args.out_dir / "04_noise_reduction_bars.png")
    plot_signal_preservation_bars(
        channels,
        (raw_ref_d1, raw_ref_d2),
        args.out_dir / "05_signal_preservation_bars.png",
    )

    # Print summary table
    print("\n── Noise + preservation summary ──")
    print(
        f"{'Method':<22} {'d1 noise':>10} {'d2 noise':>10} "
        f"{'d1 ratio':>10} {'d2 ratio':>10} {'d1 corr':>10} {'d2 corr':>10}"
    )
    ref_d1 = float(np.median(sg_residual_std(channels["raw_fd"][1])))
    ref_d2 = float(np.median(sg_residual_std(channels["raw_fd"][2])))
    for k, (ch0, ch1, ch2) in channels.items():
        med_d1 = float(np.median(sg_residual_std(ch1)))
        med_d2 = float(np.median(sg_residual_std(ch2)))
        corr_d1 = median_structure_corr(ch1, raw_ref_d1)
        corr_d2 = median_structure_corr(ch2, raw_ref_d2)
        print(
            f"{LABELS[k]:<22} {med_d1:>10.5f} {med_d2:>10.5f} "
            f"{med_d1/ref_d1:>10.3f}× {med_d2/ref_d2:>10.3f}× "
            f"{corr_d1:>10.3f} {corr_d2:>10.3f}"
        )

    print(f"\nAlle Plots gespeichert in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
