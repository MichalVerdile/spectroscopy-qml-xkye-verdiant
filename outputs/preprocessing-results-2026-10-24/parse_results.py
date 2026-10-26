#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path


def num(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def section(text, start, end=None):
    tail = text.split(start, 1)[1] if start in text else ""
    return tail.split(end, 1)[0] if end and end in tail else tail


def labeled_number(text, label):
    match = re.search(rf"^\s*{re.escape(label)}:\s*([-+0-9.eE]+)", text, re.MULTILINE)
    return num(match.group(1)) if match else None


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
rows = []

for metrics_path in sorted(root.glob("benchmark/preprocessing_training/dim_600/*/*/seed_41/**/metrics.json")):
    parts = metrics_path.parts
    dim_idx = parts.index("dim_600")
    config = parts[dim_idx + 1]
    spectrum = parts[dim_idx + 2]
    model = next((m for m in ("cnn", "xgboost", "mps", "ttn") if m in parts), None)
    if model is None:
        continue

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = {
        "preprocessing": config,
        "spectrum": spectrum,
        "model": model,
        "seed": int(metrics.get("seed", 41)),
        "input_dim": int(metrics.get("input_dim", 600)),
        "train_size": None,
        "val_size": None,
        "test_size": None,
        "val_f1_micro": None,
        "val_f1_macro": None,
        "val_f1_blend": None,
        "val_loss": None,
        "test_f1_micro": num(metrics.get("test_f1_micro")),
        "test_f1_macro": num(metrics.get("test_f1_macro")),
        "test_f1_blend": None,
        "test_loss": None,
        "test_accuracy": None,
        "test_f1_weighted": None,
        "test_precision_micro": None,
        "test_precision_macro": None,
        "test_recall_micro": None,
        "test_recall_macro": None,
        "best_epoch": None,
        "epochs_completed": None,
        "fit_minutes": None,
        "metric_source": str(metrics_path.relative_to(root)),
    }

    if model in ("cnn", "xgboost"):
        log_path = metrics_path.with_name("training_logs.csv")
        logs = read_csv(log_path)
        final = next((r for r in reversed(logs) if r.get("event") == "training_finished"), None)
        if final:
            for field in ("train_size", "val_size", "test_size"):
                value = num(final.get(field))
                row[field] = int(value) if value is not None else None
            row["val_f1_micro"] = num(final.get("val_f1_micro"))
            row["val_f1_macro"] = num(final.get("val_f1_macro"))
            row["test_f1_micro"] = num(final.get("test_f1_micro")) or row["test_f1_micro"]
            row["test_f1_macro"] = num(final.get("test_f1_macro")) or row["test_f1_macro"]
            fit_seconds = num(final.get("fit_seconds"))
            row["fit_minutes"] = fit_seconds / 60 if fit_seconds is not None else None
        epoch_rows = [r for r in logs if r.get("event") == "epoch_finished" and num(r.get("val_f1_micro")) is not None]
        if epoch_rows:
            best = max(epoch_rows, key=lambda r: (num(r.get("val_f1_micro")) + num(r.get("val_f1_macro"))) / 2)
            row["best_epoch"] = int(num(best.get("epoch")))
            row["epochs_completed"] = max(int(num(r.get("epoch"))) for r in epoch_rows)
            row["val_loss"] = num(best.get("val_loss"))

    elif model == "mps":
        summary_path = metrics_path.with_name("summary.txt")
        text = summary_path.read_text(encoding="utf-8")
        checkpoint = section(text, "Best Fold Checkpoint:", "Per-Fold Best Validation Results:")
        test = section(text, "Test Set Results:")
        row["best_epoch"] = int(labeled_number(checkpoint, "Epoch"))
        row["val_loss"] = labeled_number(checkpoint, "Loss")
        row["val_f1_micro"] = labeled_number(checkpoint, "F1 Micro")
        row["val_f1_macro"] = labeled_number(checkpoint, "F1 Macro")
        row["test_loss"] = labeled_number(test, "Loss")
        row["test_accuracy"] = labeled_number(test, "Accuracy")
        row["test_f1_weighted"] = labeled_number(test, "F1 Weighted")
        row["test_precision_micro"] = labeled_number(test, "Precision Micro")
        row["test_precision_macro"] = labeled_number(test, "Precision Macro")
        row["test_recall_micro"] = labeled_number(test, "Recall Micro")
        row["test_recall_macro"] = labeled_number(test, "Recall Macro")
        row["epochs_completed"] = int(labeled_number(text, "Total epochs trained across folds"))
        row["fit_minutes"] = labeled_number(text, "Training time")

    elif model == "ttn":
        summary_path = metrics_path.with_name("summary.txt")
        text = summary_path.read_text(encoding="utf-8")
        best_epoch = labeled_number(text, "Best epoch")
        row["best_epoch"] = int(best_epoch) if best_epoch is not None else None
        completed_match = re.search(r"Epochs completed:\s*(\d+)", text)
        row["epochs_completed"] = int(completed_match.group(1)) if completed_match else None
        row["val_loss"] = labeled_number(text, "Best val loss")
        row["test_loss"] = labeled_number(text, "Test loss")
        row["test_precision_micro"] = labeled_number(text, "Test precision_micro")
        row["test_recall_micro"] = labeled_number(text, "Test recall_micro")
        elapsed = labeled_number(text, "Elapsed seconds")
        row["fit_minutes"] = elapsed / 60 if elapsed is not None else None
        log_path = metrics_path.with_name("training_log.csv")
        logs = read_csv(log_path)
        best = next((r for r in logs if int(num(r.get("epoch"))) == row["best_epoch"]), None)
        if best:
            row["val_f1_micro"] = num(best.get("val_f1_micro"))
            row["val_f1_macro"] = num(best.get("val_f1_macro"))

    if row["val_f1_micro"] is not None and row["val_f1_macro"] is not None:
        row["val_f1_blend"] = (row["val_f1_micro"] + row["val_f1_macro"]) / 2
    if row["test_f1_micro"] is not None and row["test_f1_macro"] is not None:
        row["test_f1_blend"] = (row["test_f1_micro"] + row["test_f1_macro"]) / 2
    rows.append(row)

expected_configs = [
    "none", "snv", "minmax", "max", "area", "l2", "quantile", "pqn",
    "savgol", "gaussian", "moving_average", "lorentz", "sqrt", "log1p",
    "derivative", "bin_mean", "bin_max", "fir",
]
assert len(rows) == 360, f"Expected 360 completed results, got {len(rows)}"
for config in expected_configs:
    assert sum(r["preprocessing"] == config for r in rows) == 20, config

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} rows to {output}")
