from __future__ import annotations

import gc
import json
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
META_DATA_PATH = PROJECT_ROOT / "data" / "meta_data" / "meta_data_dict.json"
OUTPUT_DIR = PROJECT_ROOT / "jupyter" / "data_exploration" / "outputs"
N_CHUNKS = 10

META_DATA = json.loads(META_DATA_PATH.read_text())

X_AXIS_CONFIG = {
    "H-NMR": {
        "dimensions": np.asarray(META_DATA["h_nmr_spectra"]["dimensions"], dtype=np.float32),
        "label": "Chemical Shift (ppm)",
        "invert": True,
    },
    "C-NMR": {
        "dimensions": np.asarray(META_DATA["c_nmr_spectra"]["dimensions"], dtype=np.float32),
        "label": "Chemical Shift (ppm)",
        "invert": True,
    },
    "MS/MS": {
        "start": 0.0,
        "end": 1000.0,
        "label": "m/z",
        "invert": False,
    },
    "IR": {
        "dimensions": np.asarray(META_DATA["ir_spectra"]["dimensions"], dtype=np.float32),
        "label": "Wavenumber (cm^-1)",
        "invert": False,
    },
}


def is_valid_1d_spectrum(value: object) -> bool:
    return value is not None and len(value) > 10


def is_valid_msms(value: object) -> bool:
    return value is not None and len(value) > 0


def interpolate_linear(spectrum: np.ndarray, target_length: int) -> np.ndarray:
    if len(spectrum) == target_length:
        return spectrum.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, len(spectrum) - 1, len(spectrum), dtype=np.float32)
    new_x = np.linspace(0.0, len(spectrum) - 1, target_length, dtype=np.float32)
    return np.interp(new_x, old_x, spectrum).astype(np.float32)


def apply_snv(spectrum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    if std < eps:
        return (spectrum - mean).astype(np.float32)
    return ((spectrum - mean) / std).astype(np.float32)


def apply_quantile_normalization(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    sorted_values = np.sort(spectra, axis=1)
    mean_ranks = np.mean(sorted_values, axis=0)
    ranks = np.argsort(np.argsort(spectra, axis=1), axis=1)
    normalized = mean_ranks[ranks]
    return normalized.astype(np.float32)


def apply_pqn_normalization(spectra: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    row_sums = spectra.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums > eps, row_sums, 1.0)
    tic_normalized = spectra / safe_row_sums

    reference = np.median(tic_normalized, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = (tic_normalized > eps) & (reference > eps)
        quotients = np.where(mask, tic_normalized / reference, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dilution_factors = np.nanmedian(quotients, axis=1)

    dilution_factors = np.where(np.isnan(dilution_factors), 1.0, dilution_factors)
    safe_dilution = np.where(dilution_factors > eps, dilution_factors, 1.0)
    return (tic_normalized / safe_dilution[:, None]).astype(np.float32)


def make_msms_dense(spectrum: object, max_mz: int = 10000) -> np.ndarray:
    dense = np.zeros(max_mz, dtype=np.float32)
    for peak in spectrum:
        peak_pos = int(float(peak[0]) * 10)
        peak_pos = min(max(peak_pos, 0), max_mz - 1)
        dense[peak_pos] = float(peak[1])
    return dense


def get_axis(name: str, n_bins: int) -> np.ndarray:
    config = X_AXIS_CONFIG[name]
    if "dimensions" in config:
        return interpolate_linear(config["dimensions"], n_bins)
    return np.linspace(config["start"], config["end"], n_bins, dtype=np.float32)


def set_axis_limits(ax, name: str, x: np.ndarray | None = None) -> None:
    if x is not None:
        ax.set_xlim(float(x[0]), float(x[-1]))
        return

    config = X_AXIS_CONFIG[name]
    ax.set_xlim(float(config["start"]), float(config["end"]))


def find_target_sample(df: pd.DataFrame) -> int:
    mask = (
        df["h_nmr_spectra"].map(is_valid_1d_spectrum)
        & df["c_nmr_spectra"].map(is_valid_1d_spectrum)
        & df["msms_positive_40ev"].map(is_valid_msms)
        & df["ir_spectra"].map(is_valid_1d_spectrum)
    )
    matches = np.flatnonzero(mask.to_numpy())
    if len(matches) == 0:
        raise RuntimeError("Kein gemeinsames Sample mit H-NMR, C-NMR, MS/MS und IR gefunden.")
    return int(matches[0])


def load_dataframe() -> pd.DataFrame:
    parquet_files = sorted(DATA_DIR.glob("aligned_chunk_*.parquet"))[:N_CHUNKS]
    frames = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(
            parquet_file,
            columns=[
                "smiles",
                "h_nmr_spectra",
                "c_nmr_spectra",
                "msms_positive_40ev",
                "ir_spectra",
            ],
        ).copy()
        frame["source_chunk"] = parquet_file.name
        frame["source_row"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def select_valid_position(series: pd.Series, target_index: int, validator) -> int:
    mask = series.map(validator).to_numpy()
    if not mask[target_index]:
        raise RuntimeError("Zielsample ist fuer diesen Spektrentyp nicht valide.")
    return int(mask[: target_index + 1].sum() - 1)


def prepare_target_spectra(df: pd.DataFrame, target_index: int) -> tuple[dict, dict]:
    sample = df.iloc[target_index]
    spectra = {}

    h_raw = np.asarray(sample["h_nmr_spectra"], dtype=np.float32)
    h_cnn_snv = apply_snv(interpolate_linear(h_raw, 600))
    h_valid = df["h_nmr_spectra"].map(is_valid_1d_spectrum)
    h_pos = select_valid_position(df["h_nmr_spectra"], target_index, is_valid_1d_spectrum)
    h_all = np.stack(
        [interpolate_linear(np.asarray(spec, dtype=np.float32), 1800) for spec in df.loc[h_valid, "h_nmr_spectra"]]
    )
    h_mps = apply_quantile_normalization(h_all)[h_pos]
    spectra["H-NMR"] = {"raw": h_raw, "cnn": h_cnn_snv, "mps": h_mps}
    del h_all
    gc.collect()

    c_raw = np.asarray(sample["c_nmr_spectra"], dtype=np.float32)
    c_cnn_snv = apply_snv(interpolate_linear(c_raw, 600))
    c_valid = df["c_nmr_spectra"].map(is_valid_1d_spectrum)
    c_pos = select_valid_position(df["c_nmr_spectra"], target_index, is_valid_1d_spectrum)
    c_all = np.stack(
        [interpolate_linear(np.asarray(spec, dtype=np.float32), 1800) for spec in df.loc[c_valid, "c_nmr_spectra"]]
    )
    c_mps = apply_quantile_normalization(c_all)[c_pos]
    spectra["C-NMR"] = {"raw": c_raw, "cnn": c_cnn_snv, "mps": c_mps}
    del c_all
    gc.collect()

    ms_raw = np.stack(sample["msms_positive_40ev"]).astype(np.float32)
    ms_dense = make_msms_dense(ms_raw)
    ms_cnn_snv = apply_snv(interpolate_linear(ms_dense, 600))
    ms_valid = df["msms_positive_40ev"].map(is_valid_msms)
    ms_pos = select_valid_position(df["msms_positive_40ev"], target_index, is_valid_msms)
    ms_all = np.stack([make_msms_dense(spec) for spec in df.loc[ms_valid, "msms_positive_40ev"]])
    ms_mps = apply_pqn_normalization(ms_all)[ms_pos]
    spectra["MS/MS"] = {"raw": ms_raw, "cnn": ms_cnn_snv, "mps": ms_mps}
    del ms_all
    gc.collect()

    ir_raw = np.asarray(sample["ir_spectra"], dtype=np.float32)
    ir_cnn_snv = apply_snv(interpolate_linear(ir_raw, 600))
    ir_mps = apply_snv(interpolate_linear(ir_raw, 1800))
    spectra["IR"] = {"raw": ir_raw, "cnn": ir_cnn_snv, "mps": ir_mps}

    meta = {
        "smiles": sample["smiles"],
        "source_chunk": sample["source_chunk"],
        "source_row": int(sample["source_row"]),
        "loaded_rows": len(df),
    }
    return spectra, meta


def plot_single_sample(spectra: dict, meta: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))

    spectrum_titles = ["H-NMR", "C-NMR", "MS/MS", "IR"]
    row_titles = ["Raw", "CNN + SNV", "MPS-Normalisierung"]
    row_keys = ["raw", "cnn", "mps"]
    mps_labels = {"H-NMR": "Quantile", "C-NMR": "Quantile", "MS/MS": "PQN", "IR": "SNV"}

    wrapped_smiles = textwrap.fill(meta["smiles"], width=90)
    fig.suptitle(
        "Einzelnes Molekuel: Raw vs CNN + SNV vs MPS-Normalisierung\n"
        f"SMILES: {wrapped_smiles}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for col_idx, title in enumerate(spectrum_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold")

    row_order = ["H-NMR", "C-NMR", "MS/MS", "IR"]

    for row_idx, row_key in enumerate(row_keys):
        for col_idx, name in enumerate(row_order):
            ax = axes[row_idx, col_idx]
            values = spectra[name][row_key]
            axis_config = X_AXIS_CONFIG[name]

            if name == "MS/MS" and row_key == "raw":
                ax.vlines(values[:, 0], 0.0, values[:, 1], color="gray", linewidth=1.0)
                set_axis_limits(ax, name)
                ax.set_xlabel(axis_config["label"])
            else:
                x = get_axis(name, len(values))
                color = "gray" if row_key == "raw" else ("steelblue" if row_key == "cnn" else "coral")
                ax.plot(x, values, color=color, linewidth=1.0)
                ax.set_xlabel(axis_config["label"])
                set_axis_limits(ax, name, x)

            if col_idx == 0:
                ax.set_ylabel(f"{row_titles[row_idx]}\n\nIntensity")
            else:
                ax.set_ylabel("Intensity")

            if row_key == "cnn":
                ax.text(0.02, 0.95, f"SNV | n={len(values)}", transform=ax.transAxes, va="top", fontsize=8)
            if row_key == "mps":
                ax.text(
                    0.02,
                    0.95,
                    f"{mps_labels[name]} | n={len(values)}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
            if row_key == "raw":
                ax.text(0.02, 0.95, f"n={len(values)}", transform=ax.transAxes, va="top", fontsize=8)

            ax.grid(alpha=0.25)

    fig.text(
        0.5,
        0.01,
        f"Quelle: {meta['source_chunk']} | Row: {meta['source_row']} | Geladene Samples: {meta['loaded_rows']}",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(meta: dict, image_path: Path, output_path: Path) -> None:
    content = f"""# Einzelnes Sample: Raw vs CNN + SNV vs MPS

**SMILES**

`{meta["smiles"]}`

**Quelle**

- Chunk: `{meta["source_chunk"]}`
- Row im Chunk: `{meta["source_row"]}`
- Geladene Samples: `{meta["loaded_rows"]}` aus den ersten `{N_CHUNKS}` Chunks

**Plot**

![Einzelnes Sample Raw vs CNN + SNV vs MPS]({image_path.name})

**Hinweis**

- Die Zeilen zeigen `Raw`, `CNN + SNV` und `MPS-Normalisierung`.
- Die vier Spalten zeigen `H-NMR`, `C-NMR`, `MS/MS` und `IR`.
- Die Achsenskalierung fuer H-NMR, C-NMR und IR folgt direkt den in `data/meta_data/meta_data_dict.json` definierten `dimensions`.
- Fuer MS/MS gibt es dort keine festen `dimensions`; die dichte Darstellung folgt deshalb dem verwendeten Binning mit `10000` Bins bei `0.1 m/z` pro Bin.
"""
    output_path.write_text(content)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    target_index = find_target_sample(df)
    spectra, meta = prepare_target_spectra(df, target_index)

    image_path = OUTPUT_DIR / "single_sample_raw_cnn_mps.png"
    markdown_path = OUTPUT_DIR / "single_sample_raw_cnn_mps.md"

    plot_single_sample(spectra, meta, image_path)
    write_markdown(meta, image_path, markdown_path)

    print(f"Plot gespeichert: {image_path}")
    print(f"Dokument gespeichert: {markdown_path}")
    print(f"SMILES: {meta['smiles']}")


if __name__ == "__main__":
    main()
