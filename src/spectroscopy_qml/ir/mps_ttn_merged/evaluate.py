from __future__ import annotations

from pathlib import Path
import json
import sys

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.mps_classifier.model import MPSFunctionalGroupClassifier  # noqa: E402
from spectroscopy_qml.ir.mps_ttn_merged.config import PATH_CONFIG  # noqa: E402
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (  # noqa: E402
    TTNIRClassifier10_2,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    IRSpectraDataset,
    load_ir_data,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (  # noqa: E402
    count_available_data_files,
    resolve_used_file_count,
)


def _strip_compile_prefix_if_needed(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def _create_eval_dataloader(
    X_split: np.ndarray,
    y_split: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = IRSpectraDataset(X_split, y_split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )


def _get_required_arg(run_args: dict, key: str):
    if key not in run_args:
        raise KeyError(f"Selector checkpoint args are missing required key: {key}")
    return run_args[key]


def _resolve_split_path(run_args: dict, output_dir: Path, data_dir: Path) -> Path:
    configured_split_path = run_args.get("split_path")
    if configured_split_path not in (None, "", False):
        return Path(configured_split_path)

    total_data_files = count_available_data_files(data_dir)
    used_data_files = resolve_used_file_count(total_data_files, run_args.get("max_files"))
    split_suffix = "all" if run_args.get("max_files") is None else f"files{used_data_files}"
    return output_dir / f"data_split_seed{int(run_args['seed'])}_{split_suffix}.npz"


def _load_split_indices(split_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(split_path, allow_pickle=False)
    return (
        payload["val_indices"].astype(np.int64, copy=False),
        payload["test_indices"].astype(np.int64, copy=False),
    )


def _build_mps_model(run_args: dict) -> MPSFunctionalGroupClassifier:
    return MPSFunctionalGroupClassifier(
        input_dim=int(_get_required_arg(run_args, "input_dim")),
        num_sites=int(_get_required_arg(run_args, "mps_num_sites")),
        physical_dim=int(_get_required_arg(run_args, "mps_physical_dim")),
        bond_dim=int(_get_required_arg(run_args, "mps_bond_dim")),
        num_classes=int(_get_required_arg(run_args, "num_labels")),
        dropout_rate=float(_get_required_arg(run_args, "mps_dropout_rate")),
        classifier_head=str(_get_required_arg(run_args, "mps_classifier_head")),
        num_sites_2=int(_get_required_arg(run_args, "mps_num_sites_2")),
        physical_dim_2=int(_get_required_arg(run_args, "mps_physical_dim_2")),
        bond_dim_2=int(_get_required_arg(run_args, "mps_bond_dim_2")),
    )


def _build_ttn_model(run_args: dict) -> TTNIRClassifier10_2:
    return TTNIRClassifier10_2(
        num_labels=int(_get_required_arg(run_args, "num_labels")),
        chi=int(_get_required_arg(run_args, "ttn_chi")),
        input_dim=int(_get_required_arg(run_args, "input_dim")),
        segment_window_size=int(_get_required_arg(run_args, "ttn_segment_window_size")),
        segment_stride=int(_get_required_arg(run_args, "ttn_segment_stride")),
        segment_mode=str(_get_required_arg(run_args, "ttn_segment_mode")),
        segment_offset=run_args.get("ttn_segment_offset"),
        segment_state_normalize=bool(_get_required_arg(run_args, "ttn_segment_state_normalize")),
        merge_mode=str(_get_required_arg(run_args, "ttn_merge_mode")),
        merge_residual_weight=float(_get_required_arg(run_args, "ttn_merge_residual_weight")),
        merge_renormalize_output=bool(_get_required_arg(run_args, "ttn_merge_renormalize_output")),
        lorentz_gamma=float(_get_required_arg(run_args, "ttn_lorentz_gamma")),
        lorentz_kernel_half_width=int(_get_required_arg(run_args, "ttn_lorentz_kernel_half_width")),
        lorentz_norm_mode=str(_get_required_arg(run_args, "ttn_lorentz_norm_mode")),
    )


def _count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _load_branch_models(
    run_args: dict,
    mps_model_path: Path,
    ttn_model_path: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict, dict]:
    mps_checkpoint = torch.load(mps_model_path, map_location=device, weights_only=False)
    ttn_checkpoint = torch.load(ttn_model_path, map_location=device, weights_only=False)

    mps_model = _build_mps_model(run_args)
    ttn_model = _build_ttn_model(run_args)
    mps_model.load_state_dict(
        _strip_compile_prefix_if_needed(mps_checkpoint["model_state_dict"])
    )
    ttn_model.load_state_dict(
        _strip_compile_prefix_if_needed(ttn_checkpoint["model_state_dict"])
    )
    return mps_model.to(device), ttn_model.to(device), mps_checkpoint, ttn_checkpoint


def _load_branch_datasets(run_args: dict, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common_kwargs = {
        "data_dir": data_dir,
        "target_length": int(_get_required_arg(run_args, "input_dim")),
        "max_files": run_args.get("max_files"),
    }
    X_mps, y_mps = load_ir_data(
        **common_kwargs,
        apply_snv=bool(_get_required_arg(run_args, "mps_apply_snv")),
        apply_savgol=bool(_get_required_arg(run_args, "mps_apply_savgol")),
        savgol_window_length=int(_get_required_arg(run_args, "mps_savgol_window_length")),
        savgol_polyorder=int(_get_required_arg(run_args, "mps_savgol_polyorder")),
    )
    X_ttn, y_ttn = load_ir_data(
        **common_kwargs,
        apply_snv=bool(_get_required_arg(run_args, "ttn_apply_snv")),
        apply_savgol=bool(_get_required_arg(run_args, "ttn_apply_savgol")),
        savgol_window_length=int(_get_required_arg(run_args, "ttn_savgol_window_length")),
        savgol_polyorder=int(_get_required_arg(run_args, "ttn_savgol_polyorder")),
    )
    if X_mps.shape != X_ttn.shape:
        raise RuntimeError(f"Branch datasets have different shapes: MPS {X_mps.shape}, TTN {X_ttn.shape}.")
    if y_mps.shape != y_ttn.shape or not np.array_equal(y_mps, y_ttn):
        raise RuntimeError("Branch label arrays do not match exactly; evaluation is invalid.")
    return X_mps, y_mps, X_ttn, y_ttn


def evaluate_selector_model(
    mps_model: nn.Module,
    ttn_model: nn.Module,
    mps_dataloader: DataLoader,
    ttn_dataloader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray,
    selected_branch_names: list[str],
) -> dict:
    mps_model.eval()
    ttn_model.eval()
    use_mps = np.asarray([branch_name == "mps" for branch_name in selected_branch_names], dtype=bool)
    all_labels = []
    all_preds = []
    all_probs = []
    all_probs_mps = []
    all_probs_ttn = []

    with torch.no_grad():
        for (spectra_mps, labels_mps), (spectra_ttn, labels_ttn) in zip(mps_dataloader, ttn_dataloader, strict=True):
            if labels_mps.shape != labels_ttn.shape or not torch.equal(labels_mps, labels_ttn):
                raise RuntimeError("MPS and TTN evaluation batches are misaligned.")

            logits_mps = mps_model(spectra_mps.to(device))
            logits_ttn = ttn_model(spectra_ttn.to(device))
            probs_mps_np = torch.sigmoid(logits_mps).cpu().numpy()
            probs_ttn_np = torch.sigmoid(logits_ttn).cpu().numpy()
            probs_np = np.where(use_mps[None, :], probs_mps_np, probs_ttn_np)
            preds = (probs_np >= thresholds).astype(float)

            all_labels.append(labels_mps.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs_np)
            all_probs_mps.append(probs_mps_np)
            all_probs_ttn.append(probs_ttn_np)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)
    y_prob_mps = np.vstack(all_probs_mps)
    y_prob_ttn = np.vstack(all_probs_ttn)

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
        "y_prob_mps": y_prob_mps,
        "y_prob_ttn": y_prob_ttn,
        "metrics": metrics,
        "selected_branch_names": selected_branch_names,
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
    selected_branch_names = results["selected_branch_names"]

    for index, name in enumerate(functional_groups):
        yt = y_true[:, index].astype(np.int32)
        positives = int(yt.sum())
        total = int(len(yt))
        tp = int(results["true_positives_per_class"][index])
        fp = int(results["false_positives_per_class"][index])
        fn = int(results["false_negatives_per_class"][index])
        tn = int(results["true_negatives_per_class"][index])
        rows.append(
            {
                "label_idx": index,
                "label_name": name,
                "selected_branch": selected_branch_names[index],
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
            }
        )

    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)


def build_global_error_metrics(results: dict) -> dict[str, float]:
    metrics = results["metrics"].copy()
    metrics["hamming_accuracy"] = float((results["y_true"] == results["y_pred"]).mean())
    metrics["selected_mps_labels"] = float(sum(name == "mps" for name in results["selected_branch_names"]))
    metrics["selected_ttn_labels"] = float(sum(name == "ttn" for name in results["selected_branch_names"]))
    return metrics


def plot_error_analysis_detail(
    df: pd.DataFrame,
    metrics: dict[str, float],
    title: str,
    color: str,
    out_path: Path,
) -> None:
    values = df["error_rate"] * 100.0
    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    colors = ["#B00020" if x > 50 else color if x > 20 else "#BFE8E4" for x in values]
    ax1.barh(range(len(df)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df["label_name"], fontsize=8)
    ax1.set_xlabel("Missed-positive error rate: FN / positives (%)")
    ax1.set_title(f"{title} - All 37 Functional Groups")
    ax1.set_xlim(0, max(105, float(values.max()) + 5))
    ax1.grid(axis="x", alpha=0.3)
    for i, value in enumerate(values):
        if value >= 1:
            ax1.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=7)
    ax1.invert_yaxis()

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(values, bins=18, color=color, alpha=0.75, edgecolor="black")
    ax2.axvline(values.mean(), color="black", linestyle="--", label=f"Mean {values.mean():.1f}%")
    ax2.axvline(
        values.median(),
        color="black",
        linestyle=":",
        label=f"Median {values.median():.1f}%",
    )
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
        f"  Hamming accuracy:  {metrics['hamming_accuracy']:.4f}\n"
        f"  MPS labels:        {int(metrics['selected_mps_labels'])}\n"
        f"  TTN labels:        {int(metrics['selected_ttn_labels'])}\n\n"
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
    ax4.text(
        0.02,
        0.98,
        summary,
        transform=ax4.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#F4F4F4", "alpha": 0.95},
    )
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


def save_error_analysis_artifacts(
    results: dict,
    functional_groups: list[str],
    output_dir: Path,
    split_name: str,
    title: str,
    color: str,
) -> tuple[Path, Path, Path, Path]:
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


def print_detailed_results(results: dict, functional_groups: list[str], output_file=None):
    def print_line(text: str):
        print(text)
        if output_file:
            output_file.write(text + "\n")

    metrics = results["metrics"]
    selected_branch_names = results["selected_branch_names"]

    print_line("=" * 80)
    print_line("Overall Metrics")
    print_line("=" * 80)
    print_line(f"Accuracy:           {metrics['accuracy']:.4f}")
    print_line(f"F1 Micro:           {metrics['f1_micro']:.4f}")
    print_line(f"F1 Macro:           {metrics['f1_macro']:.4f}")
    print_line(f"F1 Weighted:        {metrics['f1_weighted']:.4f}")
    print_line(f"F1 Samples:         {metrics['f1_samples']:.4f}")
    print_line(f"Precision Micro:    {metrics['precision_micro']:.4f}")
    print_line(f"Precision Macro:    {metrics['precision_macro']:.4f}")
    print_line(f"Precision Weighted: {metrics['precision_weighted']:.4f}")
    print_line(f"Recall Micro:       {metrics['recall_micro']:.4f}")
    print_line(f"Recall Macro:       {metrics['recall_macro']:.4f}")
    print_line(f"Recall Weighted:    {metrics['recall_weighted']:.4f}")
    print_line(f"Selected MPS Labels: {sum(name == 'mps' for name in selected_branch_names)}")
    print_line(f"Selected TTN Labels: {sum(name == 'ttn' for name in selected_branch_names)}")

    print_line("\n" + "=" * 80)
    print_line("Per-Class Metrics")
    print_line("=" * 80)
    print_line(
        f"{'Functional Group':<25} {'Branch':<8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Spec':>8} {'Support':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    )
    print_line("-" * 128)

    for i, fg_name in enumerate(functional_groups):
        support = int(results["y_true"][:, i].sum())
        print_line(
            f"{fg_name:<25} "
            f"{selected_branch_names[i]:<8} "
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

    total_errors_per_class = (
        results["false_positives_per_class"] + results["false_negatives_per_class"]
    )
    worst_indices = np.argsort(total_errors_per_class)[::-1][:5]

    print_line("\n" + "=" * 80)
    print_line("Error Analysis")
    print_line("=" * 80)
    for index in worst_indices:
        false_positive_indices = results["false_positive_indices_per_class"][index][:10].tolist()
        false_negative_indices = results["false_negative_indices_per_class"][index][:10].tolist()
        print_line(
            f"{functional_groups[index]} ({selected_branch_names[index]}): "
            f"errors={int(total_errors_per_class[index])}, "
            f"FP={int(results['false_positives_per_class'][index])}, "
            f"FN={int(results['false_negatives_per_class'][index])}, "
            f"example FP indices={false_positive_indices}, "
            f"example FN indices={false_negative_indices}"
        )

    print_line("\n" + "=" * 80)
    print_line("Prediction Statistics")
    print_line("=" * 80)

    y_true = results["y_true"]
    y_pred = results["y_pred"]

    print_line(f"Total samples: {len(y_true)}")
    print_line(f"Average labels per sample (true): {y_true.sum(axis=1).mean():.2f}")
    print_line(f"Average labels per sample (pred): {y_pred.sum(axis=1).mean():.2f}")
    print_line(
        f"Samples with exact match: {(y_true == y_pred).all(axis=1).sum()} "
        f"({(y_true == y_pred).all(axis=1).mean() * 100:.1f}%)"
    )


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to selector checkpoint (default: use selector_model_path from config)",
)
@click.option(
    "--mps_model_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to MPS branch checkpoint (default: infer from output_dir)",
)
@click.option(
    "--ttn_model_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to TTN branch checkpoint (default: infer from output_dir)",
)
@click.option(
    "--split_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to fixed split artifact (default: infer from training args)",
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to data directory (default: use training args)",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for evaluation artifacts (default: selector checkpoint directory)",
)
@click.option(
    "--eval_batch_size",
    type=int,
    default=None,
    help="Evaluation batch size (default: min of the two branch training batch sizes)",
)
def main(model_path, mps_model_path, ttn_model_path, split_path, data_dir, output_dir, eval_batch_size):
    """Evaluate the merged MPS + TTN selector model."""

    print("=" * 80)
    print("Merged MPS + TTN Selector Evaluation")
    print("=" * 80)

    model_path = model_path or Path(PATH_CONFIG.selector_model_path)
    output_dir = output_dir or model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print(f"\nLoading selector checkpoint from: {model_path}")
    selector_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    run_args = selector_checkpoint.get("args", {})
    if not isinstance(run_args, dict):
        raise TypeError("Selector checkpoint args payload must be a dictionary.")

    mps_model_path = mps_model_path or (output_dir / Path(PATH_CONFIG.mps_model_path).name)
    ttn_model_path = ttn_model_path or (output_dir / Path(PATH_CONFIG.ttn_model_path).name)
    data_dir = data_dir or Path(run_args.get("data_dir", Path(__file__).parents[4] / "data" / "raw"))
    if not data_dir.exists():
        project_root = Path(__file__).parents[4]
        data_dir = project_root / "data" / "raw"
    split_path = split_path or _resolve_split_path(run_args, output_dir, data_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split artifact does not exist: {split_path}")

    thresholds = np.asarray(selector_checkpoint.get("thresholds"), dtype=np.float32)
    selected_branch_names = list(selector_checkpoint.get("selected_branch_names", []))
    if thresholds.ndim != 1:
        raise ValueError("Selector checkpoint thresholds must be a 1D array.")
    if len(selected_branch_names) != len(thresholds):
        raise ValueError(
            "Selector checkpoint selected_branch_names length does not match threshold count."
        )

    print(f"MPS branch checkpoint: {mps_model_path}")
    print(f"TTN branch checkpoint: {ttn_model_path}")
    print(f"Data directory:        {data_dir}")
    print(f"Split artifact:        {split_path}")
    print(f"Thresholds:            mean={thresholds.mean():.3f}, std={thresholds.std():.3f}")
    print(f"Labels assigned to MPS: {sum(name == 'mps' for name in selected_branch_names)}")
    print(f"Labels assigned to TTN: {sum(name == 'ttn' for name in selected_branch_names)}")

    print("\nLoading branch datasets...")
    X_mps, y_mps, X_ttn, y_ttn = _load_branch_datasets(run_args, data_dir)
    val_indices, test_indices = _load_split_indices(split_path)

    batch_size = eval_batch_size or min(
        int(_get_required_arg(run_args, "mps_batch_size")),
        int(_get_required_arg(run_args, "ttn_batch_size")),
    )
    num_workers = int(run_args.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    val_loader_mps = _create_eval_dataloader(
        X_mps[val_indices], y_mps[val_indices], batch_size, num_workers, pin_memory
    )
    val_loader_ttn = _create_eval_dataloader(
        X_ttn[val_indices], y_ttn[val_indices], batch_size, num_workers, pin_memory
    )
    test_loader_mps = _create_eval_dataloader(
        X_mps[test_indices], y_mps[test_indices], batch_size, num_workers, pin_memory
    )
    test_loader_ttn = _create_eval_dataloader(
        X_ttn[test_indices], y_ttn[test_indices], batch_size, num_workers, pin_memory
    )

    print("\nLoading branch models...")
    mps_model, ttn_model, mps_checkpoint, ttn_checkpoint = _load_branch_models(
        run_args,
        mps_model_path,
        ttn_model_path,
        device,
    )

    functional_groups = list(FUNCTIONAL_GROUPS.keys())

    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)
    test_results = evaluate_selector_model(
        mps_model,
        ttn_model,
        test_loader_mps,
        test_loader_ttn,
        device,
        thresholds,
        selected_branch_names,
    )

    eval_output_path = output_dir / "evaluation_results.txt"
    with eval_output_path.open("w", encoding="utf-8") as handle:
        print_detailed_results(test_results, functional_groups, handle)

    (
        test_csv_path,
        test_metrics_path,
        test_detailed_plot_path,
        test_top_plot_path,
    ) = save_error_analysis_artifacts(
        test_results,
        functional_groups,
        output_dir,
        split_name="test",
        title="Merged Selector Test Set",
        color="#4ECDC4",
    )

    print(f"\nDetailed results saved to: {eval_output_path}")
    print(f"Test error-analysis CSV saved to: {test_csv_path}")
    print(f"Test error-analysis metrics saved to: {test_metrics_path}")
    print(f"Test error-analysis plot saved to: {test_detailed_plot_path}")
    print(f"Test top-problem plot saved to: {test_top_plot_path}")

    print("\n" + "=" * 80)
    print("Evaluating on Validation Set")
    print("=" * 80)
    val_results = evaluate_selector_model(
        mps_model,
        ttn_model,
        val_loader_mps,
        val_loader_ttn,
        device,
        thresholds,
        selected_branch_names,
    )

    val_output_path = output_dir / "validation_results.txt"
    with val_output_path.open("w", encoding="utf-8") as handle:
        print_detailed_results(val_results, functional_groups, handle)

    (
        val_csv_path,
        val_metrics_path,
        val_detailed_plot_path,
        val_top_plot_path,
    ) = save_error_analysis_artifacts(
        val_results,
        functional_groups,
        output_dir,
        split_name="validation",
        title="Merged Selector Validation Set",
        color="#FF6B6B",
    )

    print(f"\nValidation results saved to: {val_output_path}")
    print(f"Validation error-analysis CSV saved to: {val_csv_path}")
    print(f"Validation error-analysis metrics saved to: {val_metrics_path}")
    print(f"Validation error-analysis plot saved to: {val_detailed_plot_path}")
    print(f"Validation top-problem plot saved to: {val_top_plot_path}")

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("Merged MPS + TTN Selector - Evaluation Summary\n")
        handle.write("=" * 80 + "\n\n")

        handle.write("Model Information:\n")
        handle.write(f"  Selector model path: {model_path}\n")
        handle.write(f"  MPS model path: {mps_model_path}\n")
        handle.write(f"  TTN model path: {ttn_model_path}\n")
        handle.write(f"  MPS best epoch: {selector_checkpoint.get('mps_best_epoch', mps_checkpoint.get('best_epoch', 'n/a'))}\n")
        handle.write(f"  TTN best epoch: {selector_checkpoint.get('ttn_best_epoch', ttn_checkpoint.get('best_epoch', 'n/a'))}\n")
        handle.write(f"  MPS parameters: {_count_trainable_parameters(mps_model):,}\n")
        handle.write(f"  TTN parameters: {_count_trainable_parameters(ttn_model):,}\n")
        handle.write(f"  Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}\n")
        handle.write(f"  Selected MPS labels: {sum(name == 'mps' for name in selected_branch_names)}\n")
        handle.write(f"  Selected TTN labels: {sum(name == 'ttn' for name in selected_branch_names)}\n\n")

        handle.write("Test Set Performance:\n")
        for metric, value in test_results["metrics"].items():
            handle.write(f"  {metric}: {value:.4f}\n")
        handle.write(
            f"  hamming_accuracy: {build_global_error_metrics(test_results)['hamming_accuracy']:.4f}\n"
        )

        handle.write("\nValidation Set Performance:\n")
        for metric, value in val_results["metrics"].items():
            handle.write(f"  {metric}: {value:.4f}\n")
        handle.write(
            f"  hamming_accuracy: {build_global_error_metrics(val_results)['hamming_accuracy']:.4f}\n"
        )

        handle.write("\nTop 5 Best Performing Functional Groups (by F1 score):\n")
        top_indices = np.argsort(test_results["f1_per_class"])[::-1][:5]
        for idx in top_indices:
            fg_name = functional_groups[idx]
            f1 = test_results["f1_per_class"][idx]
            support = int(test_results["y_true"][:, idx].sum())
            branch_name = selected_branch_names[idx]
            handle.write(f"  {fg_name}: F1={f1:.4f} (support={support}, branch={branch_name})\n")

        handle.write("\nBottom 5 Worst Performing Functional Groups (by F1 score):\n")
        bottom_indices = np.argsort(test_results["f1_per_class"])[:5]
        for idx in bottom_indices:
            fg_name = functional_groups[idx]
            f1 = test_results["f1_per_class"][idx]
            support = int(test_results["y_true"][:, idx].sum())
            branch_name = selected_branch_names[idx]
            handle.write(f"  {fg_name}: F1={f1:.4f} (support={support}, branch={branch_name})\n")

        total_errors_per_class = (
            test_results["false_positives_per_class"] + test_results["false_negatives_per_class"]
        )
        worst_error_indices = np.argsort(total_errors_per_class)[::-1][:5]

        handle.write("\nTop 5 Functional Groups by Total Errors:\n")
        for idx in worst_error_indices:
            fg_name = functional_groups[idx]
            fp_examples = test_results["false_positive_indices_per_class"][idx][:10].tolist()
            fn_examples = test_results["false_negative_indices_per_class"][idx][:10].tolist()
            handle.write(
                f"  {fg_name}: "
                f"branch={selected_branch_names[idx]}, "
                f"errors={int(total_errors_per_class[idx])}, "
                f"TP={int(test_results['true_positives_per_class'][idx])}, "
                f"FP={int(test_results['false_positives_per_class'][idx])}, "
                f"FN={int(test_results['false_negatives_per_class'][idx])}, "
                f"TN={int(test_results['true_negatives_per_class'][idx])}, "
                f"specificity={test_results['specificity_per_class'][idx]:.4f}, "
                f"example_fp_indices={fp_examples}, "
                f"example_fn_indices={fn_examples}\n"
            )

        handle.write("\nGenerated Error Analysis Artifacts:\n")
        handle.write(f"  Test CSV: {test_csv_path}\n")
        handle.write(f"  Test metrics JSON: {test_metrics_path}\n")
        handle.write(f"  Test detail plot: {test_detailed_plot_path}\n")
        handle.write(f"  Test top-plot: {test_top_plot_path}\n")
        handle.write(f"  Validation CSV: {val_csv_path}\n")
        handle.write(f"  Validation metrics JSON: {val_metrics_path}\n")
        handle.write(f"  Validation detail plot: {val_detailed_plot_path}\n")
        handle.write(f"  Validation top-plot: {val_top_plot_path}\n")

    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()