from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

XGB_SUMMARY_PATH = ROOT / "benchmark" / "xgb" / "results_600" / "summary.txt"
TTN_IR_SUMMARY_PATH = ROOT / "src" / "spectroscopy_qml" / "ir" / "tree_tensor_network" / "results_600" / "summary.txt"

XGB_TARGETS = {
    "c_nmr_test": ROOT / "artifacts" / "Error_Analyse" / "XGB" / "C_NMR",
    "c_nmr_val": ROOT / "artifacts" / "Error_Analyse" / "XGB" / "C_NMR",
    "ir_test": ROOT / "artifacts" / "Error_Analyse" / "XGB" / "IR",
    "ir_val": ROOT / "artifacts" / "Error_Analyse" / "XGB" / "IR",
}

RESULTS_LABELS = {
    "accuracy": "Accuracy",
    "f1_micro": "F1 Micro",
    "f1_macro": "F1 Macro",
    "f1_weighted": "F1 Weighted",
    "f1_samples": "F1 Samples",
    "precision_micro": "Precision Micro",
    "precision_macro": "Precision Macro",
    "precision_weighted": "Precision Weighted",
    "recall_micro": "Recall Micro",
    "recall_macro": "Recall Macro",
    "recall_weighted": "Recall Weighted",
}

SUMMARY_LABELS = {
    "accuracy": "accuracy",
    "f1_micro": "f1_micro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "f1_samples": "f1_samples",
    "precision_micro": "precision_micro",
    "precision_macro": "precision_macro",
    "precision_weighted": "precision_weighted",
    "recall_micro": "recall_micro",
    "recall_macro": "recall_macro",
    "recall_weighted": "recall_weighted",
    "hamming_accuracy": "hamming_accuracy",
}

XGB_SUMMARY_KEY_MAP = {
    "Accuracy": "accuracy",
    "F1 Score (micro)": "f1_micro",
    "F1 Score (macro)": "f1_macro",
    "F1 Score (weighted)": "f1_weighted",
    "F1 Score (samples)": "f1_samples",
    "Precision (micro)": "precision_micro",
    "Precision (macro)": "precision_macro",
    "Precision (weighted)": "precision_weighted",
    "Recall (micro)": "recall_micro",
    "Recall (macro)": "recall_macro",
    "Recall (weighted)": "recall_weighted",
    "Hamming Accuracy": "hamming_accuracy",
}


def parse_xgb_summary(path: Path) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    current: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("Result: "):
            current = line.split(":", 1)[1].strip()
            sections[current] = {}
            continue
        if current is None or not line.startswith("  "):
            continue
        stripped = line.strip()
        if ": " not in stripped:
            continue
        label, value = stripped.split(":", 1)
        if label not in XGB_SUMMARY_KEY_MAP:
            continue
        try:
            sections[current][XGB_SUMMARY_KEY_MAP[label]] = float(value.strip())
        except ValueError:
            continue

    return sections


def parse_ttn_ir_summary(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    key_map = {
        "Test f1_micro": "f1_micro",
        "Test f1_macro": "f1_macro",
        "Test precision_micro": "precision_micro",
        "Test recall_micro": "recall_micro",
    }

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if ":" not in stripped:
            continue
        label, value = stripped.split(":", 1)
        if label not in key_map:
            continue
        metrics[key_map[label]] = float(value.strip())

    return metrics


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
                left = f"{label}:"
                new_lines.append(f"{prefix}{left:<19} {updates[key]:.4f}")
                replaced = True
                break
        if not replaced:
            new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def update_summary_evaluation(path: Path, section_updates: dict[str, dict[str, float]]) -> None:
    new_lines: list[str] = []
    section: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "Test Set Performance:":
            section = "test"
            new_lines.append(line)
            continue
        if stripped == "Validation Set Performance:":
            section = "validation"
            new_lines.append(line)
            continue

        updates = section_updates.get(section or "")
        replaced = False
        if updates:
            for key, label in SUMMARY_LABELS.items():
                if key in updates and stripped.startswith(f"{label}:"):
                    prefix = line[: len(line) - len(line.lstrip())]
                    new_lines.append(f"{prefix}{label}: {updates[key]:.4f}")
                    replaced = True
                    break
        if not replaced:
            new_lines.append(line)

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def sync_xgb() -> None:
    parsed = parse_xgb_summary(XGB_SUMMARY_PATH)
    targets = {
        ROOT / "artifacts" / "Error_Analyse" / "XGB" / "C_NMR": {
            "test": parsed["c_nmr_test"],
            "validation": parsed["c_nmr_val"],
        },
        ROOT / "artifacts" / "Error_Analyse" / "XGB" / "IR": {
            "test": parsed["ir_test"],
            "validation": parsed["ir_val"],
        },
    }

    for artifact_dir, split_map in targets.items():
        update_metrics_json(artifact_dir / "test_error_analysis_metrics.json", split_map["test"])
        update_metrics_json(artifact_dir / "validation_error_analysis_metrics.json", split_map["validation"])
        update_results_txt(artifact_dir / "evaluation_results.txt", split_map["test"])
        update_results_txt(artifact_dir / "validation_results.txt", split_map["validation"])
        update_summary_evaluation(artifact_dir / "summary_evaluation.txt", split_map)


def sync_ttn_ir() -> None:
    test_updates = parse_ttn_ir_summary(TTN_IR_SUMMARY_PATH)
    artifact_dir = ROOT / "artifacts" / "Error_Analyse" / "TTN" / "IR"
    update_metrics_json(artifact_dir / "test_error_analysis_metrics.json", test_updates)
    update_results_txt(artifact_dir / "evaluation_results.txt", test_updates)
    update_summary_evaluation(artifact_dir / "summary_evaluation.txt", {"test": test_updates})
    (artifact_dir / "summary.txt").write_text(TTN_IR_SUMMARY_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    sync_xgb()
    sync_ttn_ir()


if __name__ == "__main__":
    main()
