"""Evaluation script for boosted TTN C-NMR classifiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[3]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.boosted_model import (  # noqa: E402
    build_boosted_ttn_from_metadata,
)
from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.helpers.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_cnmr_data,
)
from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.helpers.train_helpers import (  # noqa: E402
    count_available_data_files,
    resolve_device,
    resolve_used_file_count,
)


class SpectraDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, index: int):
        return (
            torch.as_tensor(self.X[index], dtype=torch.float32),
            torch.as_tensor(self.y[index], dtype=torch.float32),
        )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(output_dir: Path) -> Path:
    if (output_dir / "run_config.json").exists():
        return output_dir

    child_run_dirs = [
        child for child in output_dir.iterdir()
        if child.is_dir() and (child / "run_config.json").exists()
    ]
    if len(child_run_dirs) == 1:
        return child_run_dirs[0]
    if len(child_run_dirs) == 0:
        raise FileNotFoundError(
            f"No run_config.json found in {output_dir} or its direct child directories."
        )
    raise ValueError(
        f"Multiple run directories found under {output_dir}. Pass a specific --output_dir."
    )


def parse_specialist_indices(raw_value) -> list[int] | None:
    if raw_value in (None, "", "None", False):
        return None
    if isinstance(raw_value, list):
        return [int(value) for value in raw_value]
    return [int(index.strip()) for index in str(raw_value).split(",") if index.strip()]


def resolve_cache_path_from_config(run_config: dict) -> Path:
    configured = run_config.get("cache_path")
    if configured not in (None, "", "None"):
        return Path(configured)

    cache_dir = Path("data/cache")
    file_suffix = "all" if run_config.get("max_files") is None else f"files{int(run_config['max_files'])}"
    snv_suffix = "snv" if bool(run_config.get("apply_snv", True)) else "raw"
    return cache_dir / f"cnmr_spectra_len{int(run_config['input_dim'])}_{snv_suffix}_{file_suffix}.npz"


def resolve_split_path(run_config: dict, run_dir: Path, data_dir: Path) -> Path:
    configured_split_path = run_config.get("split_path")
    if configured_split_path:
        return Path(configured_split_path)

    total_data_files = count_available_data_files(data_dir)
    used_data_files = resolve_used_file_count(total_data_files, run_config.get("max_files"))
    split_suffix = "all" if run_config.get("max_files") is None else f"files{used_data_files}"
    return run_dir / f"data_split_seed{int(run_config['seed'])}_{split_suffix}.npz"


def load_thresholds(thresholds_path: Path, label_names: list[str]) -> np.ndarray:
    payload = _load_json(thresholds_path)
    return np.asarray([payload["thresholds"][name] for name in label_names], dtype=np.float32)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray,
) -> dict:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for spectra, labels in dataloader:
        spectra = spectra.to(device)
        probs = model(spectra, apply_sigmoid=True)
        all_probs.append(probs.detach().float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0).astype(np.int32)
    y_prob = np.concatenate(all_probs, axis=0).astype(np.float32)
    y_pred = (y_prob >= thresholds).astype(np.int32)

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

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "metrics": {
            "accuracy": accuracy_score(y_true, y_pred),
            "hamming_accuracy": float((y_true == y_pred).mean()),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        },
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0),
        "specificity_per_class": specificity_per_class,
        "true_positives_per_class": true_positives,
        "false_positives_per_class": false_positives,
        "false_negatives_per_class": false_negatives,
        "true_negatives_per_class": true_negatives,
    }


def build_per_class_dataframe(results: dict, label_names: list[str]) -> pd.DataFrame:
    rows = []
    y_true = results["y_true"]

    for index, name in enumerate(label_names):
        support = int(y_true[:, index].sum())
        tp = int(results["true_positives_per_class"][index])
        fp = int(results["false_positives_per_class"][index])
        fn = int(results["false_negatives_per_class"][index])
        tn = int(results["true_negatives_per_class"][index])
        rows.append(
            {
                "label_index": index,
                "label_name": name,
                "support": support,
                "f1": float(results["f1_per_class"][index]),
                "precision": float(results["precision_per_class"][index]),
                "recall": float(results["recall_per_class"][index]),
                "specificity": float(results["specificity_per_class"][index]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "missed_positive_error_rate": fn / support if support > 0 else 0.0,
                "binary_error_rate": (fp + fn) / max(1, len(y_true)),
            }
        )

    return pd.DataFrame(rows)


def save_split_results(results: dict, label_names: list[str], output_dir: Path, split_name: str) -> None:
    metrics_path = output_dir / f"{split_name}_metrics.json"
    per_class_path = output_dir / f"{split_name}_per_class_metrics.csv"
    predictions_path = output_dir / f"{split_name}_predictions.npz"

    metrics_path.write_text(json.dumps(results["metrics"], indent=2) + "\n", encoding="utf-8")
    build_per_class_dataframe(results, label_names).to_csv(per_class_path, index=False)
    np.savez_compressed(
        predictions_path,
        y_true=results["y_true"],
        y_prob=results["y_prob"],
        y_pred=results["y_pred"],
    )

    print(f"{split_name.capitalize()} metrics:      {metrics_path}")
    print(f"{split_name.capitalize()} per-class CSV:{per_class_path}")
    print(f"{split_name.capitalize()} predictions:  {predictions_path}")


@click.command()
@click.option("--model_path", type=click.Path(path_type=Path), default=None)
@click.option("--data_dir", type=click.Path(path_type=Path), default=None)
@click.option("--output_dir", type=click.Path(path_type=Path), default=Path("c_nmr/tree_tensor_network/boosted_results"))
@click.option("--run_config_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--thresholds_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--split_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto", show_default=True)
def main(
    model_path: Path | None,
    data_dir: Path | None,
    output_dir: Path,
    run_config_path: Path | None,
    thresholds_path: Path | None,
    split_path: Path | None,
    device: str,
) -> None:
    run_dir = run_config_path.parent if run_config_path is not None else _resolve_run_dir(output_dir)
    run_config_path = run_config_path or (run_dir / "run_config.json")
    thresholds_path = thresholds_path or (run_dir / "selected_thresholds.json")
    model_path = model_path or (run_dir / "boosted_ttn_cnmr.pt")

    run_config = _load_json(run_config_path)
    data_dir = data_dir or Path(run_config["data_dir"])
    if not data_dir.exists():
        project_root = Path(__file__).parents[6]
        data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    eval_device = resolve_device(device)
    print("=" * 80)
    print("Boosted TTN C-NMR Evaluation")
    print("=" * 80)
    print(f"Device:     {eval_device}")
    print(f"Run config: {run_config_path}")
    print(f"Model:      {model_path}")

    checkpoint = torch.load(model_path, map_location=eval_device)
    label_names = list(checkpoint["label_names"])
    model = build_boosted_ttn_from_metadata(
        model_kwargs=checkpoint["model_kwargs"],
        base_scores=checkpoint["base_scores"],
        learning_rate=float(checkpoint["learning_rate"]),
        estimators_per_label=checkpoint["estimators_per_label"],
        freeze=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(eval_device)

    thresholds = load_thresholds(thresholds_path, label_names)
    print(f"Threshold mean={thresholds.mean():.4f}, std={thresholds.std():.4f}")

    print("\nLoading data...")
    X, y = load_cnmr_data(
        data_dir=data_dir,
        target_length=int(run_config["input_dim"]),
        max_files=run_config.get("max_files"),
        apply_snv=bool(run_config.get("apply_snv", True)),
        cache_path=resolve_cache_path_from_config(run_config),
        overwrite_cache=False,
    )

    specialist_indices = parse_specialist_indices(run_config.get("specialist_indices"))
    if specialist_indices is not None:
        y = y[:, specialist_indices]

    split_path = split_path or resolve_split_path(run_config, run_dir, data_dir)
    payload = np.load(split_path, allow_pickle=False)
    val_indices = payload["val_indices"].astype(np.int64, copy=False)
    test_indices = payload["test_indices"].astype(np.int64, copy=False)

    batch_size = int(run_config.get("predict_batch_size", run_config.get("batch_size", 512)))
    val_loader = DataLoader(
        SpectraDataset(X[val_indices], y[val_indices]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        SpectraDataset(X[test_indices], y[test_indices]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("\nEvaluating validation set...")
    val_results = evaluate_model(model, val_loader, eval_device, thresholds)
    print(f"Validation F1 micro: {val_results['metrics']['f1_micro']:.4f}")
    print(f"Validation F1 macro: {val_results['metrics']['f1_macro']:.4f}")

    print("\nEvaluating test set...")
    test_results = evaluate_model(model, test_loader, eval_device, thresholds)
    print(f"Test F1 micro: {test_results['metrics']['f1_micro']:.4f}")
    print(f"Test F1 macro: {test_results['metrics']['f1_macro']:.4f}")

    save_split_results(val_results, label_names, run_dir, "validation")
    save_split_results(test_results, label_names, run_dir, "test")

    summary_path = run_dir / "evaluation_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Boosted TTN Evaluation Summary\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"Model path: {model_path}\n")
        handle.write(f"Run config: {run_config_path}\n")
        handle.write(f"Split path: {split_path}\n")
        handle.write(f"Functional groups: {len(label_names)}\n\n")
        handle.write("Validation metrics:\n")
        for key, value in val_results["metrics"].items():
            handle.write(f"  {key}: {value:.6f}\n")
        handle.write("\nTest metrics:\n")
        for key, value in test_results["metrics"].items():
            handle.write(f"  {key}: {value:.6f}\n")

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
