"""Evaluation script for the XGBoost multi-output functional-group model.

This script mirrors the TTN evaluation output style:
- evaluation_results.txt
- validation_results.txt
- summary_evaluation.txt
- test_error_analysis.csv / validation_error_analysis.csv
- test_error_analysis_metrics.json / validation_error_analysis_metrics.json
- test_error_analysis_detailed.png / validation_error_analysis_detailed.png
- test_top_problem_groups.png / validation_top_problem_groups.png

It evaluates a saved sklearn MultiOutputClassifier/XGBoost model produced by the XGB training script.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split



FUNCTIONAL_GROUPS = {
    "Acid anhydride": Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])"),
    "Acyl halide": Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"),
    "Alcohol": Chem.MolFromSmarts("[#6][OX2H]"),
    "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
    "Alkane": Chem.MolFromSmarts("[CX4;H3,H2]"),
    "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
    "Alkyne": Chem.MolFromSmarts("[CX2]#[CX2]"),
    "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
    "Amine": Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"),
    "Arene": Chem.MolFromSmarts("[cX3]1[cX3][cX3][cX3][cX3][cX3]1"),
    "Azo compound": Chem.MolFromSmarts("[#6][NX2]=[NX2][#6]"),
    "Carbamate": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2H0]"),
    "Carboxylic acid": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
    "Enamine": Chem.MolFromSmarts("[NX3][CX3]=[CX3]"),
    "Enol": Chem.MolFromSmarts("[OX2H][#6X3]=[#6]"),
    "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
    "Ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "Haloalkane": Chem.MolFromSmarts("[#6][F,Cl,Br,I]"),
    "Hydrazine": Chem.MolFromSmarts("[NX3][NX3]"),
    "Hydrazone": Chem.MolFromSmarts("[NX3][NX2]=[#6]"),
    "Imide": Chem.MolFromSmarts("[CX3](=[OX1])[NX3][CX3](=[OX1])"),
    "Imine": Chem.MolFromSmarts("[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"),
    "Isocyanate": Chem.MolFromSmarts("[NX2]=[C]=[O]"),
    "Isothiocyanate": Chem.MolFromSmarts("[NX2]=[C]=[S]"),
    "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "Nitrile": Chem.MolFromSmarts("[NX1]#[CX2]"),
    "Phenol": Chem.MolFromSmarts("[OX2H][cX3]:[c]"),
    "Phosphine": Chem.MolFromSmarts("[PX3]"),
    "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
    "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
    "Sulfonate": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]"),
    "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
    "Sulfonic acid": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H]"),
    "Sulfoxide": Chem.MolFromSmarts("[#16X3]=[OX1]"),
    "Thial": Chem.MolFromSmarts("[CX3H1](=S)[#6,H]"),
    "Thioamide": Chem.MolFromSmarts("[NX3][CX3]=[SX1]"),
    "Thiol": Chem.MolFromSmarts("[#16X2H]"),
}

COLUMN_MAPPING = {
    "h_nmr_spectra": ("h_nmr_spectra", "hnmr"),
    "c_nmr_spectra": ("c_nmr_spectra", "cnmr"),
    "ir_spectra": ("ir_spectra", "ir"),
    "pos_msms": ("msms_positive_40ev", "pos_msms"),
    "neg_msms": ("msms_negative_40ev", "neg_msms"),
}


def _configure_xgb_runtime() -> None:
    print("Using saved XGBoost MultiOutputClassifier model")

def match_group(mol: Chem.Mol, func_group) -> int:
    n = len(mol.GetSubstructMatches(func_group)) if type(func_group) is Chem.Mol else func_group(mol)
    return 0 if n == 0 else 1


def get_functional_groups(smiles: str) -> list[int] | None:
    RDLogger.DisableLog("rdApp.*")
    if smiles is None:
        return None
    smiles = str(smiles).strip().replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [match_group(mol, smarts) for smarts in FUNCTIONAL_GROUPS.values()]


def interpolate_to_600(spec) -> np.ndarray | None:
    if spec is None:
        return None
    spec = np.asarray(spec, dtype=np.float32).flatten()
    if len(spec) == 0:
        return np.zeros(600, dtype=np.float32)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    if len(spec) == 1:
        return np.full(600, spec[0], dtype=np.float32)
    old_x = np.arange(len(spec))
    new_x = np.linspace(min(old_x), max(old_x), 600)
    return interp1d(old_x, spec)(new_x).astype(np.float32)


def make_msms_spectrum(spectrum) -> np.ndarray | None:
    if spectrum is None:
        return None
    msms_spectrum = np.zeros(10000, dtype=np.float32)
    for peak in spectrum:
        if peak is None or len(peak) < 2:
            continue
        peak_pos = max(0, min(9999, int(peak[0] * 10)))
        msms_spectrum[peak_pos] = peak[1]
    return msms_spectrum


def load_data_for_column(analytical_data: Path, actual_col: str) -> pd.DataFrame:
    columns_to_load = ["smiles", actual_col]
    parquet_files = sorted(analytical_data.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    training_data = None

    for i, parquet_file in enumerate(parquet_files):
        print(f"Loading file {i + 1}/{len(parquet_files)}: {parquet_file.name}...", end=" ", flush=True)
        data = pd.read_parquet(parquet_file, columns=columns_to_load)
        print(f"[{len(data)} samples]")
        if actual_col in ["msms_positive_40ev", "msms_negative_40ev"]:
            data[actual_col] = [make_msms_spectrum(s) for s in data[actual_col]]
        data["func_group"] = [get_functional_groups(s) for s in data["smiles"]]
        data[actual_col] = [interpolate_to_600(s) for s in data[actual_col]]
        data = data.dropna(subset=[actual_col, "func_group"])
        training_data = data if training_data is None else pd.concat((training_data, data), ignore_index=True)
        del data

    if training_data is None:
        raise ValueError(f"No training data could be loaded for column: {actual_col}")
    return training_data


def _load_thresholds(thresholds_path: Path | None, functional_groups: list[str], fallback: float) -> np.ndarray:
    if thresholds_path is None:
        return np.full(len(functional_groups), fallback, dtype=np.float32)
    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "thresholds" in payload and isinstance(payload["thresholds"], dict):
        return np.asarray([payload["thresholds"].get(name, fallback) for name in functional_groups], dtype=np.float32)
    if isinstance(payload, dict) and "thresholds" in payload:
        return np.asarray(payload["thresholds"], dtype=np.float32)
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32)
    raise ValueError(f"Unsupported thresholds format: {thresholds_path}")


def _load_xgb_model(model_path: Path):
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities when available, otherwise binary predictions."""
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        if isinstance(probas, list):
            cols = []
            for proba in probas:
                arr = np.asarray(proba)
                cols.append(arr[:, 1] if arr.ndim == 2 and arr.shape[1] > 1 else arr.reshape(-1))
            return np.column_stack(cols).astype(np.float32)
        arr = np.asarray(probas)
        if arr.ndim == 3:
            return arr[:, :, 1].T.astype(np.float32)
        return arr.astype(np.float32)
    return np.asarray(model.predict(X), dtype=np.float32)


def _count_xgb_parameters(model) -> int:
    # Tree models do not expose neural-network-style parameter counts.
    try:
        return int(sum(est.get_booster().num_boosted_rounds() for est in model.estimators_))
    except Exception:
        return 0

def evaluate_arrays(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    thresh = thresholds if thresholds is not None else 0.5
    y_pred = (y_prob >= thresh).astype(np.int32)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    true_positives = np.logical_and(y_true == 1, y_pred == 1).sum(axis=0)
    false_positives = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0)
    false_negatives = np.logical_and(y_true == 1, y_pred == 0).sum(axis=0)
    true_negatives = np.logical_and(y_true == 0, y_pred == 0).sum(axis=0)
    specificity_per_class = np.divide(
        true_negatives,
        true_negatives + false_positives,
        out=np.zeros_like(true_negatives, dtype=float),
        where=(true_negatives + false_positives) != 0,
    )
    false_positive_indices = [
        np.flatnonzero((y_true[:, index] == 0) & (y_pred[:, index] == 1))
        for index in range(y_true.shape[1])
    ]
    false_negative_indices = [
        np.flatnonzero((y_true[:, index] == 1) & (y_pred[:, index] == 0))
        for index in range(y_true.shape[1])
    ]

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "specificity_per_class": specificity_per_class,
        "true_positives_per_class": true_positives,
        "false_positives_per_class": false_positives,
        "false_negatives_per_class": false_negatives,
        "true_negatives_per_class": true_negatives,
        "false_positive_indices_per_class": false_positive_indices,
        "false_negative_indices_per_class": false_negative_indices,
    }


def build_error_analysis_dataframe(results: dict, functional_groups: list[str]) -> pd.DataFrame:
    rows = []
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    for index, name in enumerate(functional_groups):
        yt = y_true[:, index].astype(np.int32)
        positives = int(yt.sum())
        total = int(len(yt))
        tp = int(results["true_positives_per_class"][index])
        fp = int(results["false_positives_per_class"][index])
        fn = int(results["false_negatives_per_class"][index])
        tn = int(results["true_negatives_per_class"][index])
        rows.append({
            "label_idx": index,
            "label_name": name,
            "positive_samples": positives,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "f1": float(results["f1_per_class"][index]),
            "precision": float(results["precision_per_class"][index]),
            "recall": float(results["recall_per_class"][index]),
            "specificity": float(results["specificity_per_class"][index]),
            "error_rate": fn / positives if positives > 0 else 0.0,
            "binary_error_rate": (fp + fn) / total if total > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)


def build_global_error_metrics(results: dict) -> dict[str, float]:
    metrics = results["metrics"].copy()
    metrics["hamming_accuracy"] = float((results["y_true"] == results["y_pred"]).mean())
    return metrics


def plot_error_analysis_detail(df: pd.DataFrame, metrics: dict[str, float], title: str, color: str, out_path: Path) -> None:
    values = df["error_rate"] * 100.0
    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    colors = ["#B00020" if x > 50 else color if x > 20 else "#BFE8E4" for x in values]
    ax1.barh(range(len(df)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df["label_name"], fontsize=8)
    ax1.set_xlabel("Missed-positive error rate: FN / positives (%)")
    ax1.set_title(f"{title} - All Functional Groups")
    ax1.set_xlim(0, max(105, float(values.max()) + 5))
    ax1.grid(axis="x", alpha=0.3)
    for i, value in enumerate(values):
        if value >= 1:
            ax1.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=7)
    ax1.invert_yaxis()

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(values, bins=18, color=color, alpha=0.75, edgecolor="black")
    ax2.axvline(values.mean(), color="black", linestyle="--", label=f"Mean {values.mean():.1f}%")
    ax2.axvline(values.median(), color="black", linestyle=":", label=f"Median {values.median():.1f}%")
    ax2.set_xlabel("Error rate (%)")
    ax2.set_ylabel("# Functional Groups")
    ax2.set_title("Error-Rate Distribution")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    top = df.head(10)
    top_values = top["error_rate"] * 100.0
    ax3.bar(range(len(top)), top_values, color=color, edgecolor="black")
    ax3.set_xticks(range(len(top)))
    ax3.set_xticklabels(top["label_name"], rotation=45, ha="right")
    ax3.set_ylabel("Error rate (%)")
    ax3.set_title("Top 10 Problem Groups")
    ax3.set_ylim(0, max(105, float(top_values.max()) + 8))
    ax3.grid(axis="y", alpha=0.3)
    for i, value in enumerate(top_values):
        ax3.text(i, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    max_row = df.iloc[0]
    min_row = df.iloc[-1]
    summary = (
        f"{title.upper()} ERROR ANALYSIS\n\n"
        f"Global metrics:\n"
        f"  F1 micro:          {metrics['f1_micro']:.4f}\n"
        f"  F1 macro:          {metrics['f1_macro']:.4f}\n"
        f"  Precision micro:   {metrics['precision_micro']:.4f}\n"
        f"  Recall micro:      {metrics['recall_micro']:.4f}\n"
        f"  Hamming accuracy:  {metrics['hamming_accuracy']:.4f}\n\n"
        f"Error-rate statistics:\n"
        f"  Mean:              {values.mean():.2f}%\n"
        f"  Median:            {values.median():.2f}%\n"
        f"  Std:               {values.std(ddof=1):.2f}%\n"
        f"  Max:               {values.max():.1f}% ({max_row['label_name']})\n"
        f"  Min:               {values.min():.1f}% ({min_row['label_name']})\n\n"
        f"Problem classes:\n"
        f"  Severe (>50%):     {(values > 50).sum()} groups\n"
        f"  High (20-50%):     {((values > 20) & (values <= 50)).sum()} groups\n"
        f"  Moderate (5-20%):  {((values > 5) & (values <= 20)).sum()} groups\n"
        f"  Low (<=5%):        {(values <= 5).sum()} groups\n"
    )
    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, va="top", family="monospace", fontsize=10,
             bbox={"boxstyle": "round", "facecolor": "#F4F4F4", "alpha": 0.95})
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_problem_groups(df: pd.DataFrame, title: str, color: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    top = df.head(15)
    values = top["error_rate"] * 100.0
    ax.barh(range(len(top)), values, color=color, edgecolor="black")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label_name"], fontsize=10)
    ax.set_xlabel("Error rate: FN / positives (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(105, float(values.max()) + 5))
    ax.grid(axis="x", alpha=0.3)
    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_error_analysis_artifacts(results: dict, functional_groups: list[str], output_dir: Path, split_name: str, title: str, color: str) -> tuple[Path, Path, Path, Path]:
    df = build_error_analysis_dataframe(results, functional_groups)
    metrics = build_global_error_metrics(results)
    csv_path = output_dir / f"{split_name}_error_analysis.csv"
    metrics_path = output_dir / f"{split_name}_error_analysis_metrics.json"
    detailed_plot_path = output_dir / f"{split_name}_error_analysis_detailed.png"
    top_plot_path = output_dir / f"{split_name}_top_problem_groups.png"
    df.to_csv(csv_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_error_analysis_detail(df, metrics, title, color, detailed_plot_path)
    plot_top_problem_groups(df, f"{title} - Top 15 Problem Groups", color, top_plot_path)
    return csv_path, metrics_path, detailed_plot_path, top_plot_path


def print_detailed_results(results: dict, functional_groups: list[str], output_file=None) -> None:
    def print_line(text: str) -> None:
        print(text)
        if output_file:
            output_file.write(text + "\n")

    metrics = results["metrics"]
    print_line("=" * 80)
    print_line("Overall Metrics")
    print_line("=" * 80)
    print_line(f"Accuracy:          {metrics['accuracy']:.4f}")
    print_line(f"F1 Micro:          {metrics['f1_micro']:.4f}")
    print_line(f"F1 Macro:          {metrics['f1_macro']:.4f}")
    print_line(f"F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print_line(f"F1 Samples:        {metrics['f1_samples']:.4f}")
    print_line(f"Precision Micro:   {metrics['precision_micro']:.4f}")
    print_line(f"Precision Macro:   {metrics['precision_macro']:.4f}")
    print_line(f"Precision Weighted: {metrics['precision_weighted']:.4f}")
    print_line(f"Recall Micro:      {metrics['recall_micro']:.4f}")
    print_line(f"Recall Macro:      {metrics['recall_macro']:.4f}")
    print_line(f"Recall Weighted:   {metrics['recall_weighted']:.4f}")

    print_line("\n" + "=" * 80)
    print_line("Per-Class Metrics")
    print_line("=" * 80)
    print_line(f"{'Functional Group':<25} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Spec':>8} {'Support':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print_line("-" * 120)
    for i, fg_name in enumerate(functional_groups):
        support = int(results["y_true"][:, i].sum())
        print_line(
            f"{fg_name:<25} "
            f"{results['f1_per_class'][i]:>8.4f} "
            f"{results['precision_per_class'][i]:>8.4f} "
            f"{results['recall_per_class'][i]:>8.4f} "
            f"{results['specificity_per_class'][i]:>8.4f} "
            f"{support:>8} "
            f"{results['true_positives_per_class'][i]:>6} "
            f"{results['false_positives_per_class'][i]:>6} "
            f"{results['false_negatives_per_class'][i]:>6} "
            f"{results['true_negatives_per_class'][i]:>6}"
        )

    total_errors_per_class = results["false_positives_per_class"] + results["false_negatives_per_class"]
    worst_indices = np.argsort(total_errors_per_class)[::-1][:5]
    print_line("\n" + "=" * 80)
    print_line("Error Analysis")
    print_line("=" * 80)
    for index in worst_indices:
        fp_idx = results["false_positive_indices_per_class"][index][:10].tolist()
        fn_idx = results["false_negative_indices_per_class"][index][:10].tolist()
        print_line(
            f"{functional_groups[index]}: errors={int(total_errors_per_class[index])}, "
            f"FP={int(results['false_positives_per_class'][index])}, "
            f"FN={int(results['false_negatives_per_class'][index])}, "
            f"example FP indices={fp_idx}, example FN indices={fn_idx}"
        )

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    print_line("\n" + "=" * 80)
    print_line("Prediction Statistics")
    print_line("=" * 80)
    print_line(f"Total samples: {len(y_true)}")
    print_line(f"Average labels per sample (true): {y_true.sum(axis=1).mean():.2f}")
    print_line(f"Average labels per sample (pred): {y_pred.sum(axis=1).mean():.2f}")
    print_line(f"Samples with exact match: {(y_true == y_pred).all(axis=1).sum()} ({(y_true == y_pred).all(axis=1).mean()*100:.1f}%)")


def _resolve_model_path(results_dir: Path, output_short_name: str) -> Path:
    preferred = results_dir / f"{output_short_name}_xgboost_model.pickle"
    if preferred.exists():
        return preferred
    candidates = sorted(results_dir.glob("*xgboost_model.pickle"))
    if len(candidates) == 1:
        return candidates[0]
    candidates = sorted(results_dir.glob("*.pickle"))
    candidates = [path for path in candidates if path.name != "results.pickle"]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No XGBoost model pickle found in {results_dir}")
    raise ValueError(f"Multiple candidate model pickles found in {results_dir}; pass --model_path explicitly.")


def _load_results_payload(results_pickle_path: Path | None) -> dict:
    if results_pickle_path is None or not results_pickle_path.exists():
        return {}
    with results_pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {results_pickle_path} to contain a dict, got {type(payload)!r}")
    return payload


def _first_array(payload: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return np.asarray(payload[key])
    return None


def _best_fold_index_from_payload(payload: dict) -> int | None:
    if "best_fold_idx" in payload and payload["best_fold_idx"] is not None:
        return int(payload["best_fold_idx"])
    if "best_fold" in payload and payload["best_fold"] is not None:
        return int(payload["best_fold"]) - 1
    return None


def _has_exact_saved_split_arrays(payload: dict, mode: str) -> bool:
    if not payload:
        return False
    has_test = (
        _first_array(payload, ("X_test", "x_test", "test_X")) is not None
        and _first_array(payload, ("y_test", "test_targets", "tgt")) is not None
    )
    has_val = (
        _first_array(payload, ("X_val", "x_val", "val_X")) is not None
        and _first_array(payload, ("y_val", "val_targets", "val_tgt")) is not None
    )
    return bool(has_test and has_val)


def _evaluate_from_saved_binary_predictions(payload: dict, mode: str) -> tuple[dict, dict, dict] | None:
    """Build evaluation results directly from predictions/targets saved by older training runs.

    Older results.pickle files often contain only binary predictions and targets, not
    the original X_test/X_val arrays. In that case exact model re-inference is
    impossible if the parquet order changed, but exact reproduction of the stored
    0.5-threshold metrics is still possible.
    """
    if not payload:
        return None

    test_y = _first_array(payload, ("y_test", "test_targets", "tgt"))
    test_pred = _first_array(payload, ("test_predictions", "test_pred", "pred"))
    if test_y is None or test_pred is None:
        return None

    val_y = _first_array(payload, ("y_val", "val_targets", "val_tgt"))
    val_pred = _first_array(payload, ("val_predictions", "val_pred"))

    best_fold_idx = _best_fold_index_from_payload(payload)
    if mode == "k_fold" and (val_y is None or val_pred is None):
        all_cv_targets = payload.get("all_cv_targets")
        all_cv_predictions = payload.get("all_cv_predictions")
        if best_fold_idx is not None and all_cv_targets is not None and all_cv_predictions is not None:
            if len(all_cv_targets) > best_fold_idx and len(all_cv_predictions) > best_fold_idx:
                val_y = np.asarray(all_cv_targets[best_fold_idx])
                val_pred = np.asarray(all_cv_predictions[best_fold_idx])

    if val_y is None or val_pred is None:
        return None

    test_y = np.asarray(test_y, dtype=np.int32)
    test_pred = np.asarray(test_pred, dtype=np.float32)
    val_y = np.asarray(val_y, dtype=np.int32)
    val_pred = np.asarray(val_pred, dtype=np.float32)

    if test_y.shape != test_pred.shape:
        raise ValueError(f"Saved test prediction shape {test_pred.shape} does not match target shape {test_y.shape}.")
    if val_y.shape != val_pred.shape:
        raise ValueError(f"Saved validation prediction shape {val_pred.shape} does not match target shape {val_y.shape}.")

    test_thresholds = np.full(test_y.shape[1], 0.5, dtype=np.float32)
    val_thresholds = np.full(val_y.shape[1], 0.5, dtype=np.float32)
    test_results = evaluate_arrays(test_y, test_pred, thresholds=test_thresholds)
    val_results = evaluate_arrays(val_y, val_pred, thresholds=val_thresholds)

    split_info = {
        "source": "saved binary predictions/targets from results.pickle",
        "seed": payload.get("seed"),
        "n_folds": payload.get("n_folds"),
        "best_fold": None if best_fold_idx is None else best_fold_idx + 1,
        "val_size": int(len(val_y)),
        "test_size": int(len(test_y)),
        "note": "No model re-inference was run because this results.pickle does not contain exact X_test/X_val arrays.",
    }
    return test_results, val_results, split_info


def _normalise_X_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array.reshape(array.shape[0], array.shape[1])
    return array


def _assert_same_targets(name: str, reconstructed: np.ndarray, saved: np.ndarray | None) -> None:
    if saved is None:
        return
    saved = np.asarray(saved)
    if reconstructed.shape != saved.shape or not np.array_equal(reconstructed, saved):
        raise ValueError(
            f"Reconstructed {name} targets do not match the targets saved in results.pickle. "
            "This means the evaluation would use a different split than training. "
            "Use the seed/n_folds from results.pickle, keep the exact same parquet input/order, "
            "or regenerate results.pickle with X_test/X_val saved."
        )


def _prepare_splits(
    X_data: np.ndarray,
    y_data: np.ndarray,
    seed: int | None,
    mode: str,
    n_folds: int | None,
    results_pickle_path: Path | None,
):
    payload = _load_results_payload(results_pickle_path)

    # Prefer split metadata from results.pickle, so evaluation uses the same
    # data split that was used during training.
    if payload:
        saved_seed = payload.get("seed")
        if saved_seed is not None:
            if seed is None:
                seed = int(saved_seed)
                print(f"Using seed from results.pickle: {seed}")
            elif int(seed) != int(saved_seed):
                print(
                    f"WARNING: CLI seed={seed} differs from results.pickle seed={saved_seed}. "
                    "Using the CLI seed, but target verification must still pass."
                )

        saved_n_folds = payload.get("n_folds")
        if saved_n_folds is not None and mode == "k_fold":
            if n_folds is None or int(n_folds) <= 1:
                n_folds = int(saved_n_folds)
                print(f"Using n_folds from results.pickle: {n_folds}")
            elif int(n_folds) != int(saved_n_folds):
                print(
                    f"WARNING: CLI n_folds={n_folds} differs from results.pickle n_folds={saved_n_folds}. "
                    "Using the CLI value, but target verification must still pass."
                )

    if seed is None:
        seed = 42
    if n_folds is None:
        n_folds = 1

    saved_X_test = _first_array(payload, ("X_test", "x_test", "test_X"))
    saved_y_test = _first_array(payload, ("y_test", "test_targets", "tgt"))
    saved_X_val = _first_array(payload, ("X_val", "x_val", "val_X"))
    saved_y_val = _first_array(payload, ("y_val", "val_targets", "val_tgt"))

    # Best case for new results.pickle files: exact arrays are stored.
    if saved_X_test is not None and saved_y_test is not None:
        X_test = _normalise_X_array(saved_X_test)
        y_test = np.asarray(saved_y_test, dtype=np.int32)

        if saved_X_val is not None and saved_y_val is not None:
            X_val = _normalise_X_array(saved_X_val)
            y_val = np.asarray(saved_y_val, dtype=np.int32)
            return X_val, y_val, X_test, y_test, {
                "source": "results.pickle arrays",
                "seed": seed,
                "val_size": len(X_val),
                "test_size": len(X_test),
            }

        print(
            "results.pickle contains X_test/y_test but no X_val/y_val. "
            "Using saved test data and reconstructing validation only."
        )
    else:
        X_test = y_test = None

    X_train_full, reconstructed_X_test, y_train_full, reconstructed_y_test = train_test_split(
        X_data, y_data, test_size=0.1, random_state=seed, shuffle=True
    )

    if X_test is None:
        X_test = reconstructed_X_test
        y_test = reconstructed_y_test
    _assert_same_targets("test", y_test, saved_y_test)

    if mode == "original":
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=1 / 9, random_state=seed, shuffle=True
        )
        _assert_same_targets("validation", y_val, saved_y_val)
        return X_val, y_val, X_test, y_test, {
            "source": "reconstructed split verified against results.pickle targets",
            "seed": seed,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }

    if mode != "k_fold":
        raise ValueError("mode must be either 'original' or 'k_fold'")

    best_fold = None
    if "best_fold_idx" in payload:
        best_fold = int(payload["best_fold_idx"]) + 1
    elif "best_fold" in payload and payload["best_fold"] is not None:
        best_fold = int(payload["best_fold"])
    if best_fold is None:
        best_fold = 1
        print("Could not find best_fold in results.pickle; using fold 1 for validation reconstruction.")

    kfold = KFold(n_splits=int(n_folds), shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(kfold.split(X_train_full), start=1):
        if fold_idx == best_fold:
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]

            if saved_y_val is None and "all_cv_targets" in payload:
                all_cv_targets = payload.get("all_cv_targets")
                if all_cv_targets is not None and len(all_cv_targets) >= best_fold:
                    saved_y_val = np.asarray(all_cv_targets[best_fold - 1])
            _assert_same_targets("validation", y_val, saved_y_val)

            return X_val, y_val, X_test, y_test, {
                "source": "reconstructed split verified against results.pickle targets",
                "seed": seed,
                "n_folds": int(n_folds),
                "best_fold": best_fold,
                "val_size": len(X_val),
                "test_size": len(X_test),
            }
    raise ValueError(f"best_fold={best_fold} is outside n_folds={n_folds}")



@click.command()
@click.option("--analytical_data", type=click.Path(exists=True, path_type=Path), required=True, default="data/raw")
@click.option("--base_out_path", type=click.Path(path_type=Path), default=None, help="Base output path used by the training script.")
@click.option("--results_dir", type=click.Path(path_type=Path), default=None, help="Exact XGBoost output directory, e.g. .../ir/original or .../ir/k_fold.")
@click.option("--column", type=click.Choice(list(COLUMN_MAPPING.keys())), default="ir_spectra", show_default=True)
@click.option("--mode", type=click.Choice(["original", "k_fold"]), default="original", show_default=True)
@click.option("--model_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--results_pickle_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--thresholds_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True, help="Seed used for split reconstruction. Defaults to the seed stored in results.pickle when available.")
@click.option("--n_folds", type=int, default=1, show_default=True, help="K-fold count. Defaults to n_folds stored in results.pickle when available.")
def main(
    analytical_data: Path,
    base_out_path: Path | None,
    results_dir: Path | None,
    column: str,
    mode: str,
    model_path: Path | None,
    results_pickle_path: Path | None,
    thresholds_path: Path | None,
    threshold: float,
    seed: int | None,
    n_folds: int | None,
) -> None:
    print("=" * 80)
    print("XGBoost IR Evaluation")
    print("=" * 80)
    _configure_xgb_runtime()

    actual_col, output_short_name = COLUMN_MAPPING[column]
    if results_dir is None:
        if base_out_path is None:
            raise click.UsageError("Pass either --results_dir or --base_out_path.")
        results_dir = base_out_path / output_short_name / mode
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_path or _resolve_model_path(results_dir, output_short_name)
    results_pickle_path = results_pickle_path or (results_dir / "results.pickle")

    print(f"Column: {column} ({actual_col})")
    print(f"Mode: {mode}")
    print(f"Model path: {model_path}")
    print(f"Results directory: {results_dir}")

    functional_group_names = list(FUNCTIONAL_GROUPS.keys())
    thresholds = _load_thresholds(thresholds_path, functional_group_names, threshold)
    print(f"Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}")

    payload = _load_results_payload(results_pickle_path)
    use_saved_binary_predictions = False
    parameter_count = 0
    load_seconds = 0.0

    saved_binary_evaluation = _evaluate_from_saved_binary_predictions(payload, mode)
    has_exact_saved_arrays = _has_exact_saved_split_arrays(payload, mode)

    if saved_binary_evaluation is not None and not has_exact_saved_arrays:
        print(
            "\nresults.pickle contains saved predictions/targets but no exact X_test/X_val arrays.\n"
            "Using the saved binary predictions directly, so the reported metrics match the historical run.\n"
            "No model re-inference is performed in this compatibility mode; custom thresholds are ignored.\n"
            "Regenerate results.pickle with the patched training script to enable exact model re-inference."
        )
        test_results, val_results, split_info = saved_binary_evaluation
        use_saved_binary_predictions = True
        print(f"Split info: {split_info}")

    else:
        if has_exact_saved_arrays:
            print("results.pickle contains exact X/y split arrays. Skipping parquet loading.")
            X_data = np.empty((0, 600), dtype=np.float32)
            y_data = np.empty((0, len(functional_group_names)), dtype=np.int32)
        else:
            load_start = time.perf_counter()
            training_data = load_data_for_column(analytical_data, actual_col)
            load_seconds = time.perf_counter() - load_start
            X_data = np.stack(training_data[actual_col].to_list()).astype(np.float32)
            y_data = np.stack(training_data["func_group"].to_list()).astype(np.int32)

            print(f"Total samples loaded: {len(training_data)}")
            print(f"Input shape: {X_data.shape}")
            print(f"Target shape: {y_data.shape}")
            print(f"Data loading time: {load_seconds:.2f}s")

        try:
            X_val, y_val, X_test, y_test, split_info = _prepare_splits(
                X_data, y_data, seed=seed, mode=mode, n_folds=n_folds, results_pickle_path=results_pickle_path
            )
        except ValueError as exc:
            saved_binary_evaluation = _evaluate_from_saved_binary_predictions(payload, mode)
            if saved_binary_evaluation is None:
                raise
            print(
                "\nWARNING: Exact split reconstruction failed:\n"
                f"  {exc}\n"
                "Falling back to saved binary predictions/targets from results.pickle.\n"
                "No model re-inference is performed in this compatibility mode; custom thresholds are ignored.\n"
                "Regenerate results.pickle with the patched training script to save X_test/X_val for exact re-inference."
            )
            test_results, val_results, split_info = saved_binary_evaluation
            use_saved_binary_predictions = True
            print(f"Split info: {split_info}")
        else:
            print(f"Split info: {split_info}")

            print("\nLoading XGBoost model...")
            model = _load_xgb_model(model_path)
            parameter_count = _count_xgb_parameters(model)
            print(f"Model loaded. Boosted rounds across outputs: {parameter_count:,}")

            print("\n" + "=" * 80)
            print("Evaluating on Test Set")
            print("=" * 80)
            test_prob = _predict_probabilities(model, X_test)
            test_results = evaluate_arrays(y_test, test_prob, thresholds=thresholds)

            print("\n" + "=" * 80)
            print("Evaluating on Validation Set")
            print("=" * 80)
            val_prob = _predict_probabilities(model, X_val)
            val_results = evaluate_arrays(y_val, val_prob, thresholds=thresholds)

    eval_output_path = results_dir / "evaluation_results.txt"
    with eval_output_path.open("w", encoding="utf-8") as handle:
        print_detailed_results(test_results, functional_group_names, handle)
    test_csv_path, test_metrics_path, test_plot_path, test_top_plot_path = save_error_analysis_artifacts(
        test_results, functional_group_names, results_dir, "test", "XGBoost IR Test Set", "#4ECDC4"
    )

    val_output_path = results_dir / "validation_results.txt"
    with val_output_path.open("w", encoding="utf-8") as handle:
        print_detailed_results(val_results, functional_group_names, handle)
    val_csv_path, val_metrics_path, val_plot_path, val_top_plot_path = save_error_analysis_artifacts(
        val_results, functional_group_names, results_dir, "validation", "XGBoost IR Validation Set", "#FF6B6B"
    )

    summary_path = results_dir / "summary_evaluation.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("XGBoost IR - Evaluation Summary\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Model Information:\n")
        handle.write(f"  Model path: {model_path}\n")
        handle.write(f"  Results directory: {results_dir}\n")
        handle.write(f"  Column: {column} ({actual_col})\n")
        handle.write(f"  Mode: {mode}\n")
        handle.write(f"  Seed: {seed}\n")
        handle.write(f"  Boosted rounds across outputs: {parameter_count:,}\n")
        if use_saved_binary_predictions:
            handle.write("  Evaluation mode: saved binary predictions from results.pickle; no model re-inference; thresholds ignored.\n")
        handle.write(f"  Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}\n")
        handle.write(f"  Split info: {split_info}\n\n")

        handle.write("Test Set Performance:\n")
        for metric, value in test_results["metrics"].items():
            handle.write(f"  {metric}: {value:.4f}\n")
        handle.write(f"  hamming_accuracy: {build_global_error_metrics(test_results)['hamming_accuracy']:.4f}\n")

        handle.write("\nValidation Set Performance:\n")
        for metric, value in val_results["metrics"].items():
            handle.write(f"  {metric}: {value:.4f}\n")
        handle.write(f"  hamming_accuracy: {build_global_error_metrics(val_results)['hamming_accuracy']:.4f}\n")

        handle.write("\nTop 5 Best Performing Functional Groups (by F1 score):\n")
        for index in np.argsort(test_results["f1_per_class"])[::-1][:5]:
            handle.write(f"  {functional_group_names[index]}: F1={test_results['f1_per_class'][index]:.4f} (support={int(test_results['y_true'][:, index].sum())})\n")

        handle.write("\nBottom 5 Worst Performing Functional Groups (by F1 score):\n")
        for index in np.argsort(test_results["f1_per_class"])[:5]:
            handle.write(f"  {functional_group_names[index]}: F1={test_results['f1_per_class'][index]:.4f} (support={int(test_results['y_true'][:, index].sum())})\n")

        total_errors_per_class = test_results["false_positives_per_class"] + test_results["false_negatives_per_class"]
        handle.write("\nTop 5 Functional Groups by Total Errors:\n")
        for index in np.argsort(total_errors_per_class)[::-1][:5]:
            handle.write(
                f"  {functional_group_names[index]}: errors={int(total_errors_per_class[index])}, "
                f"TP={int(test_results['true_positives_per_class'][index])}, "
                f"FP={int(test_results['false_positives_per_class'][index])}, "
                f"FN={int(test_results['false_negatives_per_class'][index])}, "
                f"TN={int(test_results['true_negatives_per_class'][index])}, "
                f"specificity={test_results['specificity_per_class'][index]:.4f}, "
                f"example_fp_indices={test_results['false_positive_indices_per_class'][index][:10].tolist()}, "
                f"example_fn_indices={test_results['false_negative_indices_per_class'][index][:10].tolist()}\n"
            )

        handle.write("\nGenerated Error Analysis Artifacts:\n")
        handle.write(f"  Test CSV: {test_csv_path}\n")
        handle.write(f"  Test metrics JSON: {test_metrics_path}\n")
        handle.write(f"  Test plot: {test_plot_path}\n")
        handle.write(f"  Test top-problem plot: {test_top_plot_path}\n")
        handle.write(f"  Validation CSV: {val_csv_path}\n")
        handle.write(f"  Validation metrics JSON: {val_metrics_path}\n")
        handle.write(f"  Validation plot: {val_plot_path}\n")
        handle.write(f"  Validation top-problem plot: {val_top_plot_path}\n")

    print(f"\nDetailed results saved to: {eval_output_path}")
    print(f"Validation results saved to: {val_output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Test error-analysis CSV saved to: {test_csv_path}")
    print(f"Validation error-analysis CSV saved to: {val_csv_path}")
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
