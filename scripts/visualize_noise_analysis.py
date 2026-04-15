"""
Spectral variation and noise analysis for IR spectra.

Plots raw spectra, first derivatives, and second derivatives to characterize
inter-sample variation and instrument noise.

NOTE on terminology
-------------------
"Inter-sample variation" (std / mean across different molecules) reflects
real chemical differences between compounds, NOT instrument noise.
True instrument noise is estimated separately from flat baseline regions
(the 2200-2500 cm⁻¹ dead zone, where organic molecules absorb very little)
and from Savitzky-Golay residuals within each spectrum.

NOTE on wavenumber axis
-----------------------
The parquet files store spectra as 1800-point arrays without an explicit
wavenumber column. The axis 400–4000 cm⁻¹ is derived empirically: the
strongest mean peak falls at index ~1293, which corresponds to ~2988 cm⁻¹
(C-H stretch, expected 2850-3000 cm⁻¹). This is verified at startup and
a warning is printed if the empirical peak deviates by more than 200 cm⁻¹.

NOTE on Experiment 6 channels
------------------------------
Experiment 6 applies (a) a learned positional embedding and (b) per-sample
max-abs normalization before feeding channels to the TTN. Plot 01 shows
raw (unnormalized) spectra; plot 06 shows the normalized channels that the
model actually consumes.

Usage:
    python scripts/visualize_noise_analysis.py
    python scripts/visualize_noise_analysis.py --n-samples 100 --n-display 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.mps_encoder.data_loader import interpolate_spectrum

# Assumed mid-IR range — verified empirically at startup (see verify_wavenumber_axis)
WAVENUMBER_MIN = 400.0
WAVENUMBER_MAX = 4000.0
SPECTRUM_LENGTH = 1800

# Flat baseline region: 2200-2500 cm⁻¹ ("dead zone" in organic IR)
# Used to estimate instrument noise independent of chemical variation
BASELINE_WN_LOW = 2200.0
BASELINE_WN_HIGH = 2500.0


# ── Wavenumber helpers ────────────────────────────────────────────────────────

def make_wavenumber_axis(n: int = SPECTRUM_LENGTH) -> np.ndarray:
    return np.linspace(WAVENUMBER_MIN, WAVENUMBER_MAX, n)


def wn_to_index(wn: float, axis: np.ndarray) -> int:
    return int(np.argmin(np.abs(axis - wn)))


def verify_wavenumber_axis(mean_spectrum: np.ndarray, axis: np.ndarray) -> None:
    """Warn if the empirical C-H stretch peak deviates from the expected ~2991 cm⁻¹."""
    peak_idx = int(np.argmax(mean_spectrum))
    peak_wn = axis[peak_idx]
    expected_wn = 2991.0
    deviation = abs(peak_wn - expected_wn)
    print(f"Wavenumber axis check: strongest mean peak at index {peak_idx} → {peak_wn:.0f} cm⁻¹ "
          f"(expected C-H stretch ~{expected_wn:.0f} cm⁻¹, deviation {deviation:.0f} cm⁻¹)")
    if deviation > 200:
        print("  WARNING: deviation > 200 cm⁻¹ — assumed wavenumber axis may be wrong.")
    else:
        print("  OK: axis appears consistent with standard mid-IR 400-4000 cm⁻¹ range.")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw_spectra(
    data_dir: Path,
    n_samples: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Load a random subset of spectra sampled uniformly across ALL parquet files.

    Samples are drawn proportionally from every file so that the selection
    is not biased toward the first few files in the sorted directory listing.
    """
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    rng = np.random.default_rng(seed)

    # Assign a per-file quota proportional to file size (approximate via equal share)
    n_files = len(parquet_files)
    base_quota = max(1, n_samples // n_files)
    # Shuffle file order to avoid sorted-name bias
    file_order = rng.permutation(n_files)

    all_spectra: list[np.ndarray] = []
    all_smiles: list[str] = []

    for fi in file_order:
        if len(all_spectra) >= n_samples:
            break
        pf = parquet_files[fi]
        df = pd.read_parquet(pf, columns=["ir_spectra", "smiles"])
        df = df[df["ir_spectra"].notna()].reset_index(drop=True)
        quota = min(base_quota, n_samples - len(all_spectra), len(df))
        chosen = rng.choice(len(df), size=quota, replace=False)
        for ci in chosen:
            row = df.iloc[ci]
            spectrum = interpolate_spectrum(np.asarray(row["ir_spectra"]), SPECTRUM_LENGTH)
            all_spectra.append(spectrum)
            all_smiles.append(str(row["smiles"]))

    spectra = np.stack(all_spectra)
    print(f"Loaded {len(spectra)} spectra from {min(len(file_order), n_samples)} files "
          f"(out of {n_files} total)")
    return spectra, all_smiles


# ── Derivative computation (matches Experiment 6 SpectralDerivativeFeatureMap) ──

def first_derivative_exp6(spectra: np.ndarray) -> np.ndarray:
    """Central difference with one-sided boundaries — identical to Exp6 model."""
    d = np.zeros_like(spectra)
    d[..., 0] = spectra[..., 1] - spectra[..., 0]
    d[..., -1] = spectra[..., -1] - spectra[..., -2]
    d[..., 1:-1] = 0.5 * (spectra[..., 2:] - spectra[..., :-2])
    return d


def second_derivative_exp6(spectra: np.ndarray) -> np.ndarray:
    """Second-order central difference with boundary replication — identical to Exp6 model."""
    d2 = np.zeros_like(spectra)
    d2[..., 1:-1] = spectra[..., 2:] - 2.0 * spectra[..., 1:-1] + spectra[..., :-2]
    d2[..., 0] = d2[..., 1]
    d2[..., -1] = d2[..., -2]
    return d2


def maxabs_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-sample max-abs normalization, matching Exp6's _normalize_channel."""
    scale = np.abs(arr).max(axis=-1, keepdims=True).clip(min=eps)
    return arr / scale


# ── Instrument noise estimate ─────────────────────────────────────────────────

def estimate_instrument_noise(spectra: np.ndarray, axis: np.ndarray) -> dict:
    """
    Two complementary estimates of true instrument noise.

    1. Baseline std: std within the flat 2200-2500 cm⁻¹ dead zone per sample.
       This region has minimal chemical absorption, so residual variance ≈ noise.

    2. SG residual std: std of (spectrum - Savitzky-Golay smooth) per sample.
       SG removes broad absorption features; what remains is high-freq noise.
    """
    i_low = wn_to_index(BASELINE_WN_LOW, axis)
    i_high = wn_to_index(BASELINE_WN_HIGH, axis)
    baseline_region = spectra[:, i_low:i_high]
    baseline_std = baseline_region.std(axis=-1)

    # SG filter: window=15 points, poly=3 — smooth enough to capture IR bands
    sg_smooth = np.stack([savgol_filter(s, window_length=15, polyorder=3) for s in spectra])
    sg_residuals = spectra - sg_smooth
    sg_std = sg_residuals.std(axis=-1)

    return {
        "baseline_std": baseline_std,
        "sg_residual_std": sg_std,
        "sg_residuals": sg_residuals,
        "baseline_region": baseline_region,
        "baseline_wn": axis[i_low:i_high],
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_overview(
    spectra: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
    n_display: int = 10,
) -> None:
    """Three-row overlay: raw / 1st derivative / 2nd derivative (unnormalized)."""
    n = min(n_display, len(spectra))
    cmap = plt.cm.viridis(np.linspace(0, 1, n))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        "IR Spectra — Raw · 1st Derivative · 2nd Derivative\n"
        "(unnormalized; inter-sample spread reflects chemical diversity)",
        fontsize=13, y=0.99,
    )

    rows = [
        (spectra[:n], "Raw Absorbance", "Absorbance (a.u.)"),
        (d1[:n],      "1st Derivative (Exp6 stencil)", "d(A)/d(ν)"),
        (d2[:n],      "2nd Derivative (Exp6 stencil)", "d²(A)/d(ν²)"),
    ]

    for ax, (data, title, ylabel) in zip(axes, rows):
        for s, c in zip(data, cmap):
            ax.plot(wn, s, color=c, alpha=0.6, linewidth=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Wavenumber (cm⁻¹) — assumed 400–4000, empirically verified", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_exp6_normalized_channels(
    spectra: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
    n_display: int = 10,
) -> None:
    """Three-row overlay of max-abs normalized channels — what Exp6 actually feeds the TTN.

    Note: positional embedding is learned and not reproduced here (it is small,
    ~1/sqrt(1800) ≈ 0.024 std, so its effect on the visual is negligible).
    """
    n = min(n_display, len(spectra))
    cmap = plt.cm.plasma(np.linspace(0, 1, n))

    raw_norm = maxabs_normalize(spectra[:n])
    d1_norm  = maxabs_normalize(d1[:n])
    d2_norm  = maxabs_normalize(d2[:n])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        "Experiment 6 Input Channels — Per-sample Max-Abs Normalized\n"
        "(matches SpectralDerivativeFeatureMap output fed to the TTN)",
        fontsize=13, y=0.99,
    )

    rows = [
        (raw_norm, "Channel 0: Raw (normalized)",         "A / max|A|"),
        (d1_norm,  "Channel 1: 1st Derivative (normalized)", "d¹ / max|d¹|"),
        (d2_norm,  "Channel 2: 2nd Derivative (normalized)", "d² / max|d²|"),
    ]

    for ax, (data, title, ylabel) in zip(axes, rows):
        for s, c in zip(data, cmap):
            ax.plot(wn, s, color=c, alpha=0.6, linewidth=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.set_ylim(-1.05, 1.05)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_inter_sample_variation(
    spectra: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
) -> None:
    """Mean ± 2σ across samples — shows chemical diversity, NOT instrument noise."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        "Pointwise Mean ± 2σ Across Samples\n"
        "(inter-sample variation — dominated by chemical diversity, not instrument noise)",
        fontsize=13, y=0.99,
    )

    rows = [
        (spectra, "Raw Absorbance",     "Absorbance (a.u.)"),
        (d1,      "1st Derivative",     "d(A)/d(ν)"),
        (d2,      "2nd Derivative",     "d²(A)/d(ν²)"),
    ]

    for ax, (data, title, ylabel) in zip(axes, rows):
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        ax.plot(wn, mu, color="steelblue", linewidth=1.2, label="Mean")
        ax.fill_between(wn, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.3, color="steelblue", label="±2σ (inter-sample)")
        ax.plot(wn, sigma, color="tomato", linewidth=0.9, linestyle="--",
                label="σ (inter-sample std)")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_instrument_noise(
    noise_stats: dict,
    spectra: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
    n_display: int = 5,
) -> None:
    """True instrument noise estimates: baseline region std and SG residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Instrument Noise Estimates\n"
        "(baseline region std and Savitzky-Golay residuals within each spectrum)",
        fontsize=13, y=0.99,
    )

    # Top-left: distribution of per-sample baseline std vs SG residual std
    ax = axes[0, 0]
    ax.violinplot(
        [noise_stats["baseline_std"], noise_stats["sg_residual_std"]],
        positions=[1, 2], showmedians=True,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"Baseline std\n({BASELINE_WN_LOW:.0f}–{BASELINE_WN_HIGH:.0f} cm⁻¹)",
        "SG residual std\n(whole spectrum)",
    ])
    ax.set_ylabel("Std dev (absorbance units)")
    ax.set_title("Per-sample Noise Estimates", fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")

    bm = np.median(noise_stats["baseline_std"])
    sm = np.median(noise_stats["sg_residual_std"])
    ax.text(1, bm, f" median\n {bm:.4f}", va="bottom", fontsize=8)
    ax.text(2, sm, f" median\n {sm:.4f}", va="bottom", fontsize=8)

    # Top-right: baseline region zoom
    ax = axes[0, 1]
    bwn = noise_stats["baseline_wn"]
    breg = noise_stats["baseline_region"]
    mu_b = breg.mean(axis=0)
    sig_b = breg.std(axis=0)
    n = min(n_display, len(breg))
    cmap = plt.cm.tab10(np.linspace(0, 1, n))
    for i in range(n):
        ax.plot(bwn, breg[i], color=cmap[i], alpha=0.7, linewidth=0.9)
    ax.fill_between(bwn, mu_b - 2 * sig_b, mu_b + 2 * sig_b,
                    alpha=0.2, color="gray", label="±2σ")
    ax.invert_xaxis()
    ax.set_title(f"Baseline Region ({BASELINE_WN_LOW:.0f}–{BASELINE_WN_HIGH:.0f} cm⁻¹)", fontsize=11)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Bottom-left: SG residuals for a few samples
    ax = axes[1, 0]
    sg_res = noise_stats["sg_residuals"]
    for i in range(n):
        ax.plot(wn, sg_res[i], color=cmap[i], alpha=0.7, linewidth=0.7)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.invert_xaxis()
    ax.set_title("SG Residuals (spectrum − smooth) — first few samples", fontsize=11)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Residual absorbance")
    ax.grid(True, alpha=0.25)

    # Bottom-right: pointwise std of SG residuals across all samples
    ax = axes[1, 1]
    sg_ptwise_std = sg_res.std(axis=0)
    ax.plot(wn, sg_ptwise_std, color="tomato", linewidth=0.9)
    ax.fill_between(wn, 0, sg_ptwise_std, alpha=0.2, color="tomato")
    ax.invert_xaxis()
    ax.set_title("Pointwise Std of SG Residuals (noise floor per wavenumber)", fontsize=11)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Std dev")
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_noise_in_exp6_channels(
    spectra: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    wn: np.ndarray,
    out_path: Path,
) -> None:
    """Noise distribution in the three Exp6 channels after max-abs normalization.

    SG residuals are computed on the normalized channels — this shows the
    effective noise level that the TTN actually receives per channel.
    Pointwise std of residuals reveals where along the spectrum noise is
    largest relative to the normalized signal.
    """
    raw_norm = maxabs_normalize(spectra)
    d1_norm  = maxabs_normalize(d1)
    d2_norm  = maxabs_normalize(d2)

    channels = [
        (raw_norm, "Channel 0: Raw (normalized)",          "steelblue"),
        (d1_norm,  "Channel 1: 1st Derivative (normalized)", "darkorange"),
        (d2_norm,  "Channel 2: 2nd Derivative (normalized)", "seagreen"),
    ]

    # SG residuals per normalized channel
    sg_residuals_per_channel = []
    sg_std_per_channel = []
    for ch_data, _, _ in channels:
        sg_smooth = np.stack([savgol_filter(s, window_length=15, polyorder=3) for s in ch_data])
        residuals = ch_data - sg_smooth
        sg_residuals_per_channel.append(residuals)
        sg_std_per_channel.append(residuals.std(axis=-1))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Noise in Experiment 6 Channels After Max-Abs Normalization\n"
        "(SG residuals on normalized channels = what the TTN sees as noise)",
        fontsize=13, y=0.99,
    )

    # Top row: per-sample noise std violin per channel
    ax_violin = axes[0, 0]
    vp = ax_violin.violinplot(sg_std_per_channel, positions=[1, 2, 3], showmedians=True)
    colors_v = ["steelblue", "darkorange", "seagreen"]
    for body, c in zip(vp["bodies"], colors_v):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    ax_violin.set_xticks([1, 2, 3])
    ax_violin.set_xticklabels(["Raw\n(norm.)", "1st Deriv.\n(norm.)", "2nd Deriv.\n(norm.)"])
    ax_violin.set_ylabel("SG residual std (normalized units)")
    ax_violin.set_title("Per-sample Noise Std per Channel", fontsize=11)
    ax_violin.grid(True, alpha=0.25, axis="y")
    for i, stds in enumerate(sg_std_per_channel):
        med = float(np.median(stds))
        ax_violin.text(i + 1, med, f" {med:.4f}", va="bottom", fontsize=8)

    # Top row: noise ratio across channels (how much louder is noise in d1/d2 vs raw)
    ax_ratio = axes[0, 1]
    eps = 1e-10
    raw_noise = sg_std_per_channel[0]
    d1_ratio  = sg_std_per_channel[1] / (raw_noise + eps)
    d2_ratio  = sg_std_per_channel[2] / (raw_noise + eps)
    ax_ratio.violinplot([d1_ratio, d2_ratio], positions=[1, 2], showmedians=True)
    ax_ratio.axhline(1.0, color="k", linewidth=0.8, linestyle="--", label="= raw level")
    ax_ratio.set_xticks([1, 2])
    ax_ratio.set_xticklabels(["1st Deriv. /\nRaw noise", "2nd Deriv. /\nRaw noise"])
    ax_ratio.set_ylabel("Noise ratio (normalized units)")
    ax_ratio.set_title("Noise Amplification After Normalization", fontsize=11)
    ax_ratio.legend(fontsize=8)
    ax_ratio.grid(True, alpha=0.25, axis="y")
    for i, ratio in enumerate([d1_ratio, d2_ratio]):
        med = float(np.median(ratio))
        ax_ratio.text(i + 1, med, f" {med:.2f}×", va="bottom", fontsize=9)

    # Top-right: empty — use for legend/summary text
    ax_txt = axes[0, 2]
    ax_txt.axis("off")
    summary_lines = [
        "Noise = SG residuals on normalized channel",
        "",
        f"Raw     median noise std: {np.median(sg_std_per_channel[0]):.4f}",
        f"1st d.  median noise std: {np.median(sg_std_per_channel[1]):.4f}",
        f"2nd d.  median noise std: {np.median(sg_std_per_channel[2]):.4f}",
        "",
        f"1st d. / Raw  median ratio: {np.median(d1_ratio):.2f}×",
        f"2nd d. / Raw  median ratio: {np.median(d2_ratio):.2f}×",
        "",
        "All channels in [-1, 1] after normalization.",
        "Ratio >1 = derivative channel is noisier.",
    ]
    ax_txt.text(0.05, 0.95, "\n".join(summary_lines),
                transform=ax_txt.transAxes, fontsize=9,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Bottom row: pointwise noise std along wavenumber per channel
    for col, (residuals, (_, title, color)) in enumerate(zip(sg_residuals_per_channel, channels)):
        ax = axes[1, col]
        ptwise_std = residuals.std(axis=0)
        ax.plot(wn, ptwise_std, color=color, linewidth=0.9)
        ax.fill_between(wn, 0, ptwise_std, alpha=0.2, color=color)
        ax.invert_xaxis()
        ax.set_title(f"Pointwise Noise Std — {title}", fontsize=10)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Std dev (norm. units)")
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_derivative_scale_change(
    spectra: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    out_path: Path,
) -> None:
    """Per-sample std ratio showing how differentiation scales signal spread.

    This is NOT a noise amplification plot — it shows the ratio of spread
    (std across the spectral axis) between the derivative and the raw signal.
    Because of max-abs normalization in Exp6, this ratio is partially absorbed
    before the model sees the data.
    """
    eps = 1e-10
    raw_std = spectra.std(axis=-1)
    d1_amp  = d1.std(axis=-1) / (raw_std + eps)
    d2_amp  = d2.std(axis=-1) / (raw_std + eps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Spectral Spread Change by Differentiation (std ratio vs. raw)\n"
        "Note: Exp6 applies per-sample max-abs normalization — this ratio is absorbed before the model",
        fontsize=12,
    )

    for ax, amp, label, color in [
        (axes[0], d1_amp, "1st Derivative std / Raw std", "darkorange"),
        (axes[1], d2_amp, "2nd Derivative std / Raw std", "seagreen"),
    ]:
        ax.hist(amp, bins=30, color=color, edgecolor="white", alpha=0.8)
        med = float(np.median(amp))
        ax.axvline(med, color="k", linestyle="--", linewidth=1.2, label=f"Median = {med:.3f}")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Ratio")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IR spectra variation and noise analysis")
    parser.add_argument("--data-dir",   type=Path, default=Path("data/raw"))
    parser.add_argument("--n-samples",  type=int,  default=50,
                        help="Total spectra to sample across all files (default: 50)")
    parser.add_argument("--n-display",  type=int,  default=10,
                        help="Individual traces to overlay per plot (default: 10)")
    parser.add_argument("--out-dir",    type=Path, default=Path("reports/noise_analysis"))
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading spectra from {args.data_dir} (n_samples={args.n_samples})")
    spectra, _ = load_raw_spectra(args.data_dir, args.n_samples, args.seed)
    print(f"Spectra shape: {spectra.shape}")

    wn = make_wavenumber_axis(spectra.shape[1])
    verify_wavenumber_axis(spectra.mean(axis=0), wn)

    d1 = first_derivative_exp6(spectra)
    d2 = second_derivative_exp6(spectra)

    print("\nEstimating instrument noise...")
    noise_stats = estimate_instrument_noise(spectra, wn)
    print(f"  Baseline region std   — median: {np.median(noise_stats['baseline_std']):.5f}")
    print(f"  SG residual std       — median: {np.median(noise_stats['sg_residual_std']):.5f}")

    print("\nGenerating plots...")

    plot_overview(
        spectra, d1, d2, wn,
        out_path=args.out_dir / "01_overview_raw_d1_d2.png",
        n_display=args.n_display,
    )
    plot_exp6_normalized_channels(
        spectra, d1, d2, wn,
        out_path=args.out_dir / "02_exp6_normalized_channels.png",
        n_display=args.n_display,
    )
    plot_inter_sample_variation(
        spectra, d1, d2, wn,
        out_path=args.out_dir / "03_inter_sample_variation.png",
    )
    plot_instrument_noise(
        noise_stats, spectra, wn,
        out_path=args.out_dir / "04_instrument_noise_estimates.png",
        n_display=args.n_display,
    )
    plot_noise_in_exp6_channels(
        spectra, d1, d2, wn,
        out_path=args.out_dir / "06_noise_in_exp6_channels.png",
    )
    plot_derivative_scale_change(
        spectra, d1, d2,
        out_path=args.out_dir / "05_derivative_spread_ratio.png",
    )

    print(f"\nAll plots saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
