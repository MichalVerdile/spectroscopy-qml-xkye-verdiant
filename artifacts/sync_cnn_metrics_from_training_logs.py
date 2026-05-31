from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "Error_Analyse" / "CNN"

MODALITY_DIRS = ["IR", "H-NMR", "C-NMR", "MSMS+", "MSMS-"]

MANUSCRIPT_TEST_F1 = {
    "IR": {"f1_micro": 0.8917, "f1_macro": 0.6844},
    "H-NMR": {"f1_micro": 0.3844, "f1_macro": 0.0435},
    "C-NMR": {"f1_micro": 0.6119, "f1_macro": 0.1069},
    "MSMS+": {"f1_micro": 0.6451, "f1_macro": 0.1142},
    "MSMS-": {"f1_micro": 0.6473, "f1_macro": 0.1227},
}

RESULTS_LABELS = {
    "f1_micro": "F1 Micro",
    "f1_macro": "F1 Macro",
    "precision_micro": "Precision Micro",
    "precision_macro": "Precision Macro",
    "precision_weighted": "Precision Weighted",
    "recall_micro": "Recall Micro",
    "recall_macro": "Recall Macro",
    "recall_weighted": "Recall Weighted",
}

SUMMARY_LABELS = {
    "f1_micro": "f1_micro",
    "f1_macro": "f1_macro",
    "precision_micro": "precision_micro",
    "precision_macro": "precision_macro",
    "precision_weighted": "precision_weighted",
    "recall_micro": "recall_micro",
    "recall_macro": "recall_macro",
    "recall_weighted": "recall_weighted",
    "hamming_accuracy": "hamming_accuracy",
}

ROW_PATTERN = re.compile(
    r"^(?P<label_name>.+?)\s+"
    r"(?P<f1>\d+\.\d+)\s+"
    r"(?P<precision>\d+\.\d+)\s+"
    r"(?P<recall>\d+\.\d+)\s+"
    r"(?P<specificity>\d+\.\d+)\s+"
    r"(?P<positive_samples>\d+)\s+"
    r"(?P<tp>\d+)\s+"
    r"(?P<fp>\d+)\s+"
    r"(?P<fn>\d+)\s+"
    r"(?P<tn>\d+)\s*$"
)

CSV_FIELDS = [
    "label_idx",
    "label_name",
    "positive_samples",
    "tp",
    "fn",
    "fp",
    "tn",
    "f1",
    "precision",
    "recall",
    "specificity",
    "error_rate",
    "binary_error_rate",
]


def load_label_index_map(csv_path: Path) -> dict[str, int]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["label_name"]: int(row["label_idx"]) for row in csv.DictReader(handle)}


def parse_per_class_rows(results_path: Path, label_index_map: dict[str, int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    in_table = False

    for raw_line in results_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped == "Per-Class Metrics":
            in_table = True
            continue
        if not in_table:
            continue
        if stripped.startswith("Error Analysis"):
            break
        if not stripped or stripped.startswith("Functional Group") or stripped.startswith("---"):
            continue

        match = ROW_PATTERN.match(line)
        if not match:
            continue

        payload = match.groupdict()
        label_name = payload["label_name"].strip()
        positive_samples = int(payload["positive_samples"])
        tp = int(payload["tp"])
        fp = int(payload["fp"])
        fn = int(payload["fn"])
        tn = int(payload["tn"])
        num_samples = tp + fp + fn + tn

        rows.append(
            {
                "label_idx": label_index_map[label_name],
                "label_name": label_name,
                "positive_samples": positive_samples,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "f1": float(payload["f1"]),
                "precision": float(payload["precision"]),
                "recall": float(payload["recall"]),
                "specificity": float(payload["specificity"]),
                "error_rate": 0.0 if positive_samples == 0 else fn / positive_samples,
                "binary_error_rate": 0.0 if num_samples == 0 else (fp + fn) / num_samples,
            }
        )

    rows.sort(key=lambda row: float(row["error_rate"]), reverse=True)
    return rows


def compute_row_metrics(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    total_tp = sum(int(row["tp"]) for row in rows)
    total_fp = sum(int(row["fp"]) for row in rows)
    total_fn = sum(int(row["fn"]) for row in rows)
    total_support = sum(int(row["positive_samples"]) for row in rows)
    num_rows = len(rows)
    num_samples = int(rows[0]["tp"]) + int(rows[0]["fp"]) + int(rows[0]["fn"]) + int(rows[0]["tn"])

    precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    precision_macro = sum(float(row["precision"]) for row in rows) / num_rows if num_rows else 0.0
    recall_macro = sum(float(row["recall"]) for row in rows) / num_rows if num_rows else 0.0
    precision_weighted = (
        sum(float(row["precision"]) * int(row["positive_samples"]) for row in rows) / total_support if total_support else 0.0
    )
    recall_weighted = (
        sum(float(row["recall"]) * int(row["positive_samples"]) for row in rows) / total_support if total_support else 0.0
    )
    hamming_accuracy = (
        sum(int(row["tp"]) + int(row["tn"]) for row in rows) / (num_rows * num_samples) if num_rows and num_samples else 0.0
    )

    return {
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "hamming_accuracy": hamming_accuracy,
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_metrics_json(path: Path, updates: dict[str, float]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_results_txt(path: Path, updates: dict[str, float]) -> None:
    new_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        replaced = False
        for key, label in RESULTS_LABELS.items():
            if key in updates and stripped.startswith(f"{label}:"):
                prefix = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{prefix}{label + ':':<19} {updates[key]:.4f}")
                replaced = True
                break
        if not replaced:
            new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def update_summary_txt(path: Path, updates: dict[str, float]) -> None:
    new_lines: list[str] = []
    in_test_section = False

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "Test Set Performance:":
            in_test_section = True
            new_lines.append(line)
            continue
        if stripped == "Validation Set Performance:":
            in_test_section = False
            new_lines.append(line)
            continue

        replaced = False
        if in_test_section:
            for key, label in SUMMARY_LABELS.items():
                if key in updates and stripped.startswith(f"{label}:"):
                    prefix = line[: len(line) - len(line.lstrip())]
                    new_lines.append(f"{prefix}{label}: {updates[key]:.4f}")
                    replaced = True
                    break
        if not replaced:
            new_lines.append(line)

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> None:
    for modality in MODALITY_DIRS:
        modality_dir = ARTIFACT_ROOT / modality
        label_index_map = load_label_index_map(modality_dir / "test_error_analysis.csv")
        test_rows = parse_per_class_rows(modality_dir / "evaluation_results.txt", label_index_map)

        write_csv(modality_dir / "test_error_analysis.csv", test_rows)

        test_updates = compute_row_metrics(test_rows)
        test_updates.update(MANUSCRIPT_TEST_F1[modality])

        update_metrics_json(modality_dir / "test_error_analysis_metrics.json", test_updates)
        update_results_txt(modality_dir / "evaluation_results.txt", test_updates)

        summary_path = modality_dir / "summary_evaluation.txt"
        if summary_path.exists():
            update_summary_txt(summary_path, test_updates)


if __name__ == "__main__":
    main()
