#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


WORKSPACE = Path("/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant")
SOURCE_PACKAGE = WORKSPACE / "outputs/paper-learning-curves-20261109"
FINAL_PACKAGE = WORKSPACE / "outputs/paper-learning-curves-all-seeds-final-20261110"
DATA_ROOT = WORKSPACE / "outputs/learning-curves-seeds41-43-20261109/source_repo"
STANDARD = DATA_ROOT / "benchmark/preprocessing_training/dim_600/fir_gaussian_sqrt_snv"
XGB = DATA_ROOT / "benchmark/preprocessing_training/dim_600/fir_gaussian_sqrt_snv_xgb_learning_curves"

SEEDS = (41, 42, 43, 44, 45)
MODELS = ("cnn", "mps", "ttn", "xgboost")
LABELS = {"cnn": "CNN", "mps": "MPS", "ttn": "TTN", "xgboost": "XGBoost"}
COLORS = {"cnn": "#009E73", "mps": "#0072B2", "ttn": "#E69F00", "xgboost": "#D55E00"}
MODALITIES = (
    ("cnmr", "13C-NMR", "cnmr"),
    ("hnmr", "1H-NMR", "hnmr"),
    ("ir", "IR", "ir"),
    ("msms_neg", "MS/MS (-)", "neg_msms"),
    ("msms_pos", "MS/MS (+)", "pos_msms"),
)


def path_for(model: str, modality: str, output_dir: str, seed: int) -> Path:
    base = STANDARD / modality / f"seed_{seed}"
    if model == "cnn":
        return base / "cnn" / output_dir / "original/training_logs.csv"
    if model == "mps":
        return base / "mps/results/training_log.csv"
    if model == "ttn":
        return base / "ttn/training_log.csv"
    return XGB / modality / f"seed_{seed}" / "xgboost" / output_dir / "original/training_logs.csv"


def load(model: str, modality: str, output_dir: str, seed: int):
    frame = pd.read_csv(path_for(model, modality, output_dir, seed))
    if model == "cnn":
        frame = frame.loc[frame.event == "epoch_finished"].copy()
        xcol = "epoch"
    elif model == "xgboost":
        if not bool((frame.event == "training_finished").any()):
            raise RuntimeError(f"Incomplete XGBoost curve: {modality}, seed {seed}")
        frame = frame.loc[frame.event == "learning_curve"].copy()
        xcol = "step"
    else:
        xcol = "epoch"
    columns = [xcol, "train_f1_micro", "train_f1_macro", "val_f1_micro", "val_f1_macro"]
    frame = frame[columns].dropna().sort_values(xcol).drop_duplicates(xcol, keep="last")
    return frame, xcol


def band(axis, x, array, label, color):
    mean = array.mean(axis=0)
    sd = array.std(axis=0, ddof=1)
    axis.plot(x, mean, color=color, linewidth=1.8, label=label)
    axis.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=color, alpha=0.16, linewidth=0)


def regenerate_main_hnmr():
    target = FINAL_PACKAGE / "01_main_native_axes"
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.0), sharey=True)
    status = None
    for column, model in enumerate(MODELS):
        curves = [load(model, "hnmr", "hnmr", seed) for seed in SEEDS]
        common = set(curves[0][0][curves[0][1]].astype(float))
        for frame, xcol in curves[1:]:
            common &= set(frame[xcol].astype(float))
        x = np.array(sorted(common), dtype=float)
        for row, metric in enumerate(("micro", "macro")):
            axis = axes[row, column]
            for split, color in (("train", "#0072B2"), ("val", "#D55E00")):
                stack = []
                for frame, xcol in curves:
                    indexed = frame.set_index(frame[xcol].astype(float))
                    stack.append(indexed.loc[x, f"{split}_f1_{metric}"].to_numpy(float))
                band(axis, x, np.vstack(stack), "Train" if split == "train" else "Validation", color)
            axis.set_ylim(0, 1)
            axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.85)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(LABELS[model], fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{metric.capitalize()}-F1", fontsize=11)
            if row == 1:
                axis.set_xlabel("Boosting step" if model == "xgboost" else "Epoch", fontsize=10)
        if model == "xgboost":
            status = (len(x), float(x[-1]))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("1H-NMR learning curves — mean ± SD over seeds 41–45", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.015, "Native optimization axes; shaded regions show ±1 standard deviation across five seeds.", ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.09, top=0.90, wspace=0.18, hspace=0.24)
    stem = target / "main_hnmr_native_axes"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return status


def regenerate_appendix_hnmr(seed: int):
    target = FINAL_PACKAGE / "04_appendix_individual_seeds" / f"seed_{seed}"
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 7.7), sharey=True)
    xgb_last = None
    for column, model in enumerate(MODELS):
        frame, xcol = load(model, "hnmr", "hnmr", seed)
        for row, metric in enumerate(("micro", "macro")):
            axis = axes[row, column]
            axis.plot(frame[xcol], frame[f"train_f1_{metric}"], color="#0072B2", linewidth=1.7, label="Train")
            axis.plot(frame[xcol], frame[f"val_f1_{metric}"], color="#D55E00", linewidth=1.7, label="Validation")
            axis.set_ylim(0, 1)
            axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.85)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(LABELS[model], fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{metric.capitalize()}-F1")
            if row == 1:
                axis.set_xlabel("Boosting step" if model == "xgboost" else "Epoch")
        if model == "xgboost":
            xgb_last = float(frame[xcol].iloc[-1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"1H-NMR individual learning curves — Seed {seed}", fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.08, top=0.90, wspace=0.18, hspace=0.23)
    stem = target / f"hnmr_seed_{seed}_individual"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return xgb_last


def regenerate_normalized_hnmr(seed: int):
    target = FINAL_PACKAGE / "05_supplement_normalized_qualitative" / f"seed_{seed}"
    panels = (("train_f1_micro", "Train Micro-F1"), ("train_f1_macro", "Train Macro-F1"), ("val_f1_micro", "Validation Micro-F1"), ("val_f1_macro", "Validation Macro-F1"))
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.7), sharey=True)
    for model in MODELS:
        frame, xcol = load(model, "hnmr", "hnmr", seed)
        xmin, xmax = frame[xcol].min(), frame[xcol].max()
        progress = (frame[xcol] - xmin) / (xmax - xmin) * 100.0
        for axis, (metric, _) in zip(axes, panels):
            axis.plot(progress, frame[metric], color=COLORS[model], linewidth=1.8, label=LABELS[model])
    for axis, (_, title) in zip(axes, panels):
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Normalized training progress (%)")
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.85)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1 score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(f"QUALITATIVE ONLY — 1H-NMR, Seed {seed}", fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, 0.02, "Normalized progress compares curve shape, not epochs, boosting rounds, wall-clock time, or computational cost.", ha="center", fontsize=9, color="#8B0000")
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.16, top=0.82, wspace=0.16)
    stem = target / f"hnmr_seed_{seed}_normalized_qualitative"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if FINAL_PACKAGE.exists():
        shutil.rmtree(FINAL_PACKAGE)
    shutil.copytree(SOURCE_PACKAGE, FINAL_PACKAGE)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "axes.labelcolor": "#222222", "xtick.color": "#333333", "ytick.color": "#333333"})
    common_points, common_last = regenerate_main_hnmr()
    last44 = regenerate_appendix_hnmr(44)
    last45 = regenerate_appendix_hnmr(45)
    regenerate_normalized_hnmr(44)
    regenerate_normalized_hnmr(45)

    status_path = FINAL_PACKAGE / "artifact_status.csv"
    status = pd.read_csv(status_path)
    status.loc[(status.collection == "main") & (status.modality == "1H-NMR") & (status.model == "XGBoost"), ["complete", "common_points", "last_native_step"]] = [True, common_points, common_last]
    status.loc[(status.collection == "appendix") & (status.modality == "1H-NMR") & (status.model == "XGBoost") & (status.seed == 44), ["complete", "last_native_step"]] = [True, last44]
    status.loc[(status.collection == "appendix") & (status.modality == "1H-NMR") & (status.model == "XGBoost") & (status.seed == 45), ["complete", "last_native_step"]] = [True, last45]
    status.to_csv(status_path, index=False)
    if not bool(status.complete.all()):
        raise RuntimeError("Artifact status still contains incomplete rows")

    (FINAL_PACKAGE / "README.md").write_text(
        "# Paper learning-curve package (FINAL)\n\n"
        "Preprocessing: FIR + Gaussian smoothing + sqrt transformation + SNV normalization. Input dimension: 600.\n\n"
        "All CNN, MPS, TTN, and XGBoost learning curves for seeds 41–45 are complete. "
        "Main figures use native epochs or boosting steps and show mean ± one sample standard deviation. "
        "Test data are used only in the final result table. The normalized 0–100% figures are explicitly qualitative.\n"
    )
    print(f"FINAL_PACKAGE={FINAL_PACKAGE}")


if __name__ == "__main__":
    main()
