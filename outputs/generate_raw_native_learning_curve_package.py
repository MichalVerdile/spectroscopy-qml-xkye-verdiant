#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


WORKSPACE = Path("/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant")
DATA_ROOT = WORKSPACE / "tmp/raw_native_learning_curve_source"
XGB_ROOT = WORKSPACE / "benchmark/preprocessing_training/raw_native_xgboost_external_learning_curves"
TIME_BOOK = WORKSPACE / "outputs/raw_native_paper_style/raw_native_seed41_45_average_results_paper_style.xlsx"
PACKAGE = WORKSPACE / "outputs/raw-native-paper-learning-curves-all-seeds-20261116"
ZIP_PATH = WORKSPACE / "outputs/raw-native-paper-learning-curves-all-seeds-20261116.zip"

SEEDS = (41, 42, 43, 44, 45)
MODELS = ("cnn", "mps", "ttn", "xgboost")
LABELS = {"cnn": "CNN", "mps": "MPS", "ttn": "TTN", "xgboost": "XGBoost"}
MODEL_COLORS = {"cnn": "#009E73", "mps": "#0072B2", "ttn": "#E69F00", "xgboost": "#D55E00"}
MODALITIES = (
    ("cnmr", "13C-NMR", "cnmr"),
    ("hnmr", "1H-NMR", "hnmr"),
    ("ir", "IR", "ir"),
    ("msms_neg", "MS/MS (-)", "neg_msms"),
    ("msms_pos", "MS/MS (+)", "pos_msms"),
)


def log_path(model: str, modality: str, output_dir: str, seed: int) -> Path:
    if model == "xgboost":
        return XGB_ROOT / modality / f"seed_{seed}" / "xgboost" / output_dir / "original/training_logs.csv"
    base = DATA_ROOT / modality / f"seed_{seed}"
    if model == "cnn":
        return base / "cnn" / output_dir / "original/training_logs.csv"
    if model == "mps":
        return base / "mps/results/training_log.csv"
    return base / "ttn/training_log.csv"


def metrics_path(model: str, modality: str, output_dir: str, seed: int) -> Path:
    if model == "xgboost":
        return XGB_ROOT / modality / f"seed_{seed}" / "xgboost" / output_dir / "original/metrics.json"
    base = DATA_ROOT / modality / f"seed_{seed}"
    if model == "cnn":
        return base / "cnn" / output_dir / "original/metrics.json"
    if model == "mps":
        return base / "mps/results/metrics.json"
    return base / "ttn/metrics.json"


def load_curve(model: str, modality: str, output_dir: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(log_path(model, modality, output_dir, seed))
    if model == "cnn":
        frame = frame.loc[frame["event"] == "epoch_finished"].copy()
    elif model == "xgboost":
        frame = frame.loc[frame["event"] == "learning_curve"].copy()
        frame = frame.rename(columns={"step": "epoch"})
    columns = ["epoch", "train_f1_micro", "train_f1_macro", "val_f1_micro", "val_f1_macro"]
    frame = frame[columns].dropna().sort_values("epoch").drop_duplicates("epoch", keep="last")
    if frame.empty:
        raise RuntimeError(f"Empty curve: {model}, {modality}, seed {seed}")
    return frame


def load_time_map() -> dict[tuple[str, str, int], float]:
    frame = pd.read_excel(TIME_BOOK, sheet_name="Seed data")
    modality_map = {
        "¹³C-NMR": "cnmr",
        "¹H-NMR": "hnmr",
        "IR": "ir",
        "MS/MS(-)": "msms_neg",
        "MS/MS(+)": "msms_pos",
    }
    model_map = {"CNN": "cnn", "MPS": "mps", "TTN": "ttn"}
    result = {}
    for row in frame.itertuples(index=False):
        if row.Model not in model_map:
            continue
        result[(modality_map[row.Modality], model_map[row.Model], int(row.Seed))] = float(row[3])
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.set_ylim(0, 1)
    axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def common_epoch_stack(curves: list[pd.DataFrame], column: str) -> tuple[np.ndarray, np.ndarray]:
    common = set(curves[0]["epoch"].astype(float))
    for frame in curves[1:]:
        common &= set(frame["epoch"].astype(float))
    x = np.array(sorted(common), dtype=float)
    if not len(x):
        raise RuntimeError("No common epochs across seeds")
    arrays = []
    for frame in curves:
        indexed = frame.set_index(frame["epoch"].astype(float))
        arrays.append(indexed.loc[x, column].to_numpy(float))
    return x, np.vstack(arrays)


def mean_sd_band(axis: plt.Axes, x: np.ndarray, array: np.ndarray, label: str, color: str) -> None:
    average = array.mean(axis=0)
    sd = array.std(axis=0, ddof=1)
    axis.plot(x, average, color=color, linewidth=1.8, label=label)
    axis.fill_between(x, np.clip(average - sd, 0, 1), np.clip(average + sd, 0, 1), color=color, alpha=0.16, linewidth=0)


def create_main_figures(status_rows: list[dict]) -> list[Path]:
    output = PACKAGE / "01_main_native_axes"
    pdfs = []
    for modality, label, output_dir in MODALITIES:
        fig, axes = plt.subplots(2, len(MODELS), figsize=(18.5, 7.7), sharey=True)
        for column, model in enumerate(MODELS):
            curves = [load_curve(model, modality, output_dir, seed) for seed in SEEDS]
            last_common = None
            points = None
            for row, metric in enumerate(("micro", "macro")):
                axis = axes[row, column]
                for split, color in (("train", "#0072B2"), ("val", "#D55E00")):
                    x, stack = common_epoch_stack(curves, f"{split}_f1_{metric}")
                    mean_sd_band(axis, x, stack, "Train" if split == "train" else "Validation", color)
                    last_common = float(x[-1])
                    points = len(x)
                style_axis(axis)
                if row == 0:
                    axis.set_title(LABELS[model], fontsize=12, fontweight="bold")
                if column == 0:
                    axis.set_ylabel(f"{metric.capitalize()}-F1", fontsize=11)
                if row == 1:
                    axis.set_xlabel("Boosting step" if model == "xgboost" else "Epoch", fontsize=10)
            status_rows.append({"collection": "main", "modality": label, "model": LABELS[model], "seed": "", "complete": True, "common_points": points, "last_native_step": last_common})
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
        fig.suptitle(f"{label} raw/native learning curves — mean ± SD over seeds 41–45", fontsize=14, fontweight="bold", y=0.995)
        fig.text(
            0.5,
            0.015,
            "Native optimization axes (epochs or boosting steps); shaded regions show ±1 sample standard deviation across five seeds.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.09, top=0.90, wspace=0.18, hspace=0.24)
        stem = output / f"main_{modality}_native_axes"
        save_figure(fig, stem)
        pdfs.append(stem.with_suffix(".pdf"))
    merge_pdfs(pdfs, output / "main_learning_curves_native_axes.pdf")
    return pdfs


def create_individual_figures(status_rows: list[dict]) -> list[Path]:
    output = PACKAGE / "04_appendix_individual_seeds"
    pdfs = []
    for seed in SEEDS:
        seed_dir = output / f"seed_{seed}"
        for modality, label, output_dir in MODALITIES:
            fig, axes = plt.subplots(2, len(MODELS), figsize=(18.5, 7.5), sharey=True)
            for column, model in enumerate(MODELS):
                frame = load_curve(model, modality, output_dir, seed)
                for row, metric in enumerate(("micro", "macro")):
                    axis = axes[row, column]
                    axis.plot(frame["epoch"], frame[f"train_f1_{metric}"], color="#0072B2", linewidth=1.7, label="Train")
                    axis.plot(frame["epoch"], frame[f"val_f1_{metric}"], color="#D55E00", linewidth=1.7, label="Validation")
                    style_axis(axis)
                    if row == 0:
                        axis.set_title(LABELS[model], fontsize=12, fontweight="bold")
                    if column == 0:
                        axis.set_ylabel(f"{metric.capitalize()}-F1")
                    if row == 1:
                        axis.set_xlabel("Boosting step" if model == "xgboost" else "Epoch")
                status_rows.append({"collection": "appendix", "modality": label, "model": LABELS[model], "seed": seed, "complete": True, "common_points": "", "last_native_step": float(frame["epoch"].iloc[-1])})
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
            fig.suptitle(f"{label} raw/native individual learning curves — Seed {seed}", fontsize=14, fontweight="bold", y=0.995)
            fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.90, wspace=0.18, hspace=0.23)
            stem = seed_dir / f"{modality}_seed_{seed}_individual"
            save_figure(fig, stem)
            pdfs.append(stem.with_suffix(".pdf"))
    merge_pdfs(pdfs, output / "appendix_individual_learning_curves.pdf")
    return pdfs


def create_normalized_figures(status_rows: list[dict]) -> list[Path]:
    output = PACKAGE / "05_supplement_normalized_qualitative"
    panels = (
        ("train_f1_micro", "Train Micro-F1"),
        ("train_f1_macro", "Train Macro-F1"),
        ("val_f1_micro", "Validation Micro-F1"),
        ("val_f1_macro", "Validation Macro-F1"),
    )
    pdfs = []
    for seed in SEEDS:
        seed_dir = output / f"seed_{seed}"
        for modality, label, output_dir in MODALITIES:
            fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.7), sharey=True)
            for model in MODELS:
                frame = load_curve(model, modality, output_dir, seed)
                minimum, maximum = frame["epoch"].min(), frame["epoch"].max()
                progress = np.zeros(len(frame)) if maximum == minimum else (frame["epoch"] - minimum) / (maximum - minimum) * 100.0
                for axis, (metric, _) in zip(axes, panels):
                    axis.plot(progress, frame[metric], color=MODEL_COLORS[model], linewidth=1.8, label=LABELS[model])
                status_rows.append({"collection": "normalized", "modality": label, "model": LABELS[model], "seed": seed, "complete": True, "common_points": "", "last_native_step": float(maximum)})
            for axis, (_, title) in zip(axes, panels):
                axis.set_title(title, fontsize=11, fontweight="bold")
                axis.set_xlim(0, 100)
                style_axis(axis)
                axis.set_xlabel("Normalized training progress (%)")
            axes[0].set_ylabel("F1 score")
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96))
            fig.suptitle(f"QUALITATIVE ONLY — {label} raw/native, Seed {seed}", fontsize=14, fontweight="bold", y=1.01)
            fig.text(0.5, 0.02, "Normalized progress compares curve shape, not epochs, wall-clock time, or computational cost.", ha="center", fontsize=9, color="#8B0000")
            fig.subplots_adjust(left=0.055, right=0.99, bottom=0.16, top=0.82, wspace=0.16)
            stem = seed_dir / f"{modality}_seed_{seed}_normalized_qualitative"
            save_figure(fig, stem)
            pdfs.append(stem.with_suffix(".pdf"))
    merge_pdfs(pdfs, output / "supplement_normalized_progress_qualitative.pdf")
    return pdfs


def merge_pdfs(paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pages = [Image.open(path.with_suffix(".png")).convert("RGB") for path in paths]
    pages[0].save(destination, save_all=True, append_images=pages[1:], resolution=150.0)
    for page in pages:
        page.close()


def collect_results(time_map: dict[tuple[str, str, int], float]) -> pd.DataFrame:
    records = []
    for modality, label, output_dir in MODALITIES:
        for model in MODELS:
            for seed in SEEDS:
                metrics = json.loads(metrics_path(model, modality, output_dir, seed).read_text())
                curve = load_curve(model, modality, output_dir, seed)
                records.append({
                    "modality_key": modality,
                    "modality": label,
                    "model": LABELS[model],
                    "seed": seed,
                    "test_f1_micro": float(metrics["test_f1_micro"]),
                    "test_f1_macro": float(metrics["test_f1_macro"]),
                    "final_val_f1_micro": float(curve["val_f1_micro"].iloc[-1]),
                    "final_val_f1_macro": float(curve["val_f1_macro"].iloc[-1]),
                    "training_min": time_map.get((modality, model, seed), np.nan),
                })
    return pd.DataFrame(records)


def create_validation_time(results: pd.DataFrame) -> None:
    output = PACKAGE / "02_validation_vs_training_time"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for (modality, model), group in results.groupby(["modality", "model"], sort=False):
        if group["training_min"].isna().all():
            continue
        for metric in ("micro", "macro"):
            records.append({
                "modality": modality,
                "model": model,
                "metric": metric,
                "time_mean_min": group["training_min"].mean(),
                "time_sd_min": group["training_min"].std(ddof=1),
                "val_mean": group[f"final_val_f1_{metric}"].mean(),
                "val_sd": group[f"final_val_f1_{metric}"].std(ddof=1),
            })
    summary = pd.DataFrame(records)
    summary.to_csv(output / "validation_vs_training_time_summary.csv", index=False)
    color_map = {LABELS[key]: value for key, value in MODEL_COLORS.items()}
    for metric in ("micro", "macro"):
        fig, axis = plt.subplots(figsize=(9.2, 6.2))
        subset = summary.loc[summary.metric == metric]
        for model in [LABELS[key] for key in MODELS]:
            model_rows = subset.loc[subset.model == model]
            axis.errorbar(model_rows.time_mean_min, model_rows.val_mean, xerr=model_rows.time_sd_min, yerr=model_rows.val_sd,
                          fmt="o", markersize=7, capsize=3, linewidth=1.2, color=color_map[model], label=model)
            for row in model_rows.itertuples():
                axis.annotate(row.modality, (row.time_mean_min, row.val_mean), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Training time (min; model fitting only)")
        axis.set_ylabel(f"Final validation {metric.capitalize()}-F1")
        axis.set_title(f"Raw/native validation {metric.capitalize()}-F1 vs training time", fontweight="bold")
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.85)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        save_figure(fig, output / f"validation_{metric}_f1_vs_training_time")


def create_result_tables(results: pd.DataFrame) -> None:
    output = PACKAGE / "03_result_table"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for (modality, model), group in results.groupby(["modality", "model"], sort=False):
        records.append({
            "modality": modality,
            "model": model,
            "test_micro_mean": group.test_f1_micro.mean(),
            "test_micro_sd": group.test_f1_micro.std(ddof=1),
            "test_macro_mean": group.test_f1_macro.mean(),
            "test_macro_sd": group.test_f1_macro.std(ddof=1),
            "training_min_mean": group.training_min.mean(),
            "training_min_sd": group.training_min.std(ddof=1),
        })
    summary = pd.DataFrame(records)
    summary.to_csv(output / "test_results_mean_sd_seeds_41_45.csv", index=False)
    formatted = summary.copy()
    formatted["Micro-F1"] = formatted.apply(lambda row: f"{row.test_micro_mean:.4f} ± {row.test_micro_sd:.4f}", axis=1)
    formatted["Macro-F1"] = formatted.apply(lambda row: f"{row.test_macro_mean:.4f} ± {row.test_macro_sd:.4f}", axis=1)
    formatted["Training time (min)"] = formatted.apply(
        lambda row: "pending Sanjose timing" if pd.isna(row.training_min_mean)
        else f"{row.training_min_mean:.1f} ± {row.training_min_sd:.1f}",
        axis=1,
    )
    formatted = formatted[["modality", "model", "Micro-F1", "Macro-F1", "Training time (min)"]]
    formatted.to_csv(output / "test_results_paper_formatted.csv", index=False)
    (output / "test_results_paper_table.md").write_text(formatted.to_markdown(index=False) + "\n")
    (output / "test_results_paper_table.tex").write_text(formatted.to_latex(index=False, escape=True) + "\n")


def write_readme() -> None:
    (PACKAGE / "README.md").write_text(
        "# Raw/native paper learning-curve package\n\n"
        "Input: native/raw spectral resolution. No smoothing, representation transform, or normalization. "
        "The shared train/validation/test splits and seeds 41–45 match the preprocessed comparison run.\n\n"
        "Included models: CNN, MPS, TTN, and XGBoost. Main figures use native epochs or XGBoost boosting steps "
        "and show mean ± one sample standard "
        "deviation. Test data are used only in the result tables. Normalized 0–100% figures are qualitative only. "
        "The XGBoost learning curves come from the isolated external-GPU source directory; external timings are not "
        "mixed into the timing comparison. XGBoost timing remains pending until the matching Sanjose timing job finishes. "
        "Training time denotes model fitting only and excludes loading, preprocessing, final test evaluation, and output writing.\n"
    )


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })
    PACKAGE.mkdir(parents=True, exist_ok=True)
    time_map = load_time_map()
    status_rows: list[dict] = []
    create_main_figures(status_rows)
    create_individual_figures(status_rows)
    create_normalized_figures(status_rows)
    results = collect_results(time_map)
    create_validation_time(results)
    create_result_tables(results)
    pd.DataFrame(status_rows).to_csv(PACKAGE / "artifact_status.csv", index=False)
    write_readme()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=PACKAGE.parent, base_dir=PACKAGE.name)
    print(f"PACKAGE={PACKAGE}")
    print(f"ZIP={ZIP_PATH}")
    print(f"FILES={sum(1 for path in PACKAGE.rglob('*') if path.is_file())}")


if __name__ == "__main__":
    main()
