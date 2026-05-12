from pathlib import Path
import sys
import re
import csv
from io import StringIO

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from scipy.interpolate import interp1d
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


src_dir = Path(__file__).parents[3]
sys.path.insert(0, str(src_dir))

from spectroscopy_qml.ir.mps_classifier.config import (
    DATA_CONFIG,
    TRAINING_CONFIG,
    PATH_CONFIG,
)
from spectroscopy_qml.ir.mps_classifier.data_loader import (
    IRSpectraDataset,
    load_ir_data,
    FUNCTIONAL_GROUPS,
)
from spectroscopy_qml.ir.mps_classifier.model import MPSFunctionalGroupClassifier
from spectroscopy_qml.ir.mps_classifier.train import _get_model_config_kwargs


FUNCTIONAL_GROUP_NAMES = list(FUNCTIONAL_GROUPS.keys())

csv_text = """PASTE_YOUR_CSV_HERE"""

TARGET_FORMULAS_FROM_CSV = [
    row["molecular_formula"]
    for row in csv.DictReader(StringIO(csv_text))
    if "molecular_formula" in row and row["molecular_formula"]
]

TARGET_FORMULAS = (
    "C27H25F3N2O3",
    "C28H20N4O2",
)

IDENTIFIER_COLUMNS = [
    "formula",
    "molecular_formula",
    "smiles",
    "canonical_smiles",
    "inchi",
    "inchikey",
    "compound_id",
    "id",
    "name",
]


def interpolate_to_600(spec):
    old_x = np.arange(len(spec))
    new_x = np.linspace(old_x.min(), old_x.max(), 600)
    return interp1d(old_x, spec)(new_x)


def normalize_formula(value):
    if value is None or pd.isna(value):
        return None
    return str(value).replace(" ", "").strip()


def labels_from_binary(row):
    idx = np.where(row == 1)[0]
    return [FUNCTIONAL_GROUP_NAMES[i] for i in idx]


def top_probability_labels(prob_row, max_labels=10):
    idx = np.argsort(prob_row)[::-1][:max_labels]
    return [f"{FUNCTIONAL_GROUP_NAMES[i]} ({prob_row[i]:.3f})" for i in idx]


def label_difference(y_true_row, y_pred_row):
    true_set = set(np.where(y_true_row == 1)[0])
    pred_set = set(np.where(y_pred_row == 1)[0])

    correct = sorted(true_set & pred_set)
    false_pos = sorted(pred_set - true_set)
    false_neg = sorted(true_set - pred_set)

    return {
        "correct_labels": [FUNCTIONAL_GROUP_NAMES[i] for i in correct],
        "false_positive_labels": [FUNCTIONAL_GROUP_NAMES[i] for i in false_pos],
        "false_negative_labels": [FUNCTIONAL_GROUP_NAMES[i] for i in false_neg],
    }


def error_counts(y_true_row, y_pred_row):
    fp = int(np.logical_and(y_pred_row == 1, y_true_row == 0).sum())
    fn = int(np.logical_and(y_pred_row == 0, y_true_row == 1).sum())

    return {
        "false_positive_count": fp,
        "false_negative_count": fn,
        "total_error_count": fp + fn,
    }


def classify_case(y_true_row, y_pred_row):
    counts = error_counts(y_true_row, y_pred_row)
    fp = counts["false_positive_count"]
    fn = counts["false_negative_count"]

    if fp == 0 and fn == 0:
        return "correct"
    if fp > 0 and fn > 0:
        return "mixed error"
    if fp > 0:
        return "false positive"
    return "false negative"


def build_model_from_checkpoint(checkpoint, device):
    config = checkpoint["config"]
    kwargs = _get_model_config_kwargs(config)

    model = MPSFunctionalGroupClassifier(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def run_inference(model, test_loader, device, thresholds):
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for spectra, labels in test_loader:
            spectra = spectra.to(device)

            logits = model(spectra)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= thresholds).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(labels.numpy().astype(int))

    return (
        np.vstack(all_targets),
        np.vstack(all_preds),
        np.vstack(all_probs),
    )


def load_raw_spectra_with_metadata(data_dir, seed, test_size, max_files=None):
    spectra = []
    metadata = []

    parquet_files = sorted(Path(data_dir).glob("*.parquet"))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    for parquet_file in parquet_files:
        available_columns = pd.read_parquet(parquet_file).columns
        identifier_columns = [c for c in IDENTIFIER_COLUMNS if c in available_columns]

        data = pd.read_parquet(
            parquet_file,
            columns=["ir_spectra"] + identifier_columns,
        )

        for row_idx, row in data.iterrows():
            spectra.append(interpolate_to_600(row["ir_spectra"]))

            meta = {
                "source_file": parquet_file.name,
                "source_row": int(row_idx),
                "global_index": len(spectra) - 1,
            }

            for col in identifier_columns:
                meta[col] = row[col]

            metadata.append(meta)

    X_raw = np.stack(spectra)
    indices = np.arange(len(X_raw))

    _, X_test_raw, _, test_indices = train_test_split(
        X_raw,
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    test_metadata = [metadata[i] for i in test_indices]

    return X_test_raw, test_metadata


def find_test_indices_by_formula(metadata, formula):
    target = normalize_formula(formula)
    matches = []

    for test_idx, meta in enumerate(metadata):
        for key in ["formula", "molecular_formula"]:
            value = normalize_formula(meta.get(key))
            if value == target:
                matches.append(test_idx)
                break

    return matches


def choose_formula_match(matches, y_true, y_pred, match_strategy):
    if len(matches) == 0:
        raise ValueError("choose_formula_match() received an empty matches list.")

    fp = np.logical_and(y_pred == 1, y_true == 0).sum(axis=1)
    fn = np.logical_and(y_pred == 0, y_true == 1).sum(axis=1)
    total = fp + fn

    matches = np.asarray(matches)

    if match_strategy == "first":
        return int(matches[0])

    if match_strategy == "mixed":
        mixed = matches[(fp[matches] > 0) & (fn[matches] > 0)]
        if len(mixed) > 0:
            return int(mixed[np.argmax(total[mixed])])
        print("No mixed case found for this formula. Falling back to worst error.")

    if match_strategy == "worst" or match_strategy == "mixed":
        return int(matches[np.argmax(total[matches])])

    if match_strategy == "correct":
        correct = matches[total[matches] == 0]
        if len(correct) > 0:
            return int(correct[0])
        print("No correct case found for this formula. Falling back to first match.")
        return int(matches[0])

    return int(matches[0])


def identifier_string(meta):
    parts = []

    for col in IDENTIFIER_COLUMNS:
        value = meta.get(col)
        if value is not None and not pd.isna(value):
            parts.append(f"{col}: {value}")

    parts.append(f"source_file: {meta['source_file']}")
    parts.append(f"source_row: {meta['source_row']}")
    parts.append(f"global_index: {meta['global_index']}")

    return " | ".join(parts)


def sanitize_filename(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def plot_label_confusion(y_true, y_pred, out_path, top_n=20):
    n_labels = y_true.shape[1]
    confusion = np.zeros((n_labels, n_labels), dtype=float)

    for true_row, pred_row in zip(y_true, y_pred):
        true_idx = np.where(true_row == 1)[0]
        pred_idx = np.where(pred_row == 1)[0]

        for i in true_idx:
            for j in pred_idx:
                if i != j and true_row[j] == 0:
                    confusion[i, j] += 1

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion),
        where=row_sums != 0,
    )

    support = y_true.sum(axis=0)
    active = np.where(support > 0)[0]
    error_mass = confusion.sum(axis=1)

    selected = active[np.argsort(error_mass[active])[-top_n:]]
    selected = selected[np.argsort(support[selected])]

    mat = confusion_norm[np.ix_(selected, selected)]
    names = [FUNCTIONAL_GROUP_NAMES[i] for i in selected]

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Row-normalized false-positive co-prediction")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("Predicted false-positive label")
    plt.ylabel("Ground-truth label")
    plt.title("IR functional-group confusion structure — MPS")
    plt.tight_layout()

    plt.savefig(out_path / "mps_ir_confusion_heatmap.pdf")
    plt.savefig(out_path / "mps_ir_confusion_heatmap.png", dpi=300)
    plt.close()


def plot_per_label_f1(y_true, y_pred, out_path):
    rows = []

    for i, name in enumerate(FUNCTIONAL_GROUP_NAMES):
        score = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        support = int(y_true[:, i].sum())
        rows.append((name, score, support))

    rows = sorted(rows, key=lambda x: x[1])

    names = [r[0] for r in rows]
    values = [r[1] for r in rows]

    plt.figure(figsize=(11, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.ylabel("F1")
    plt.title("IR per-label F1 scores — MPS")
    plt.tight_layout()

    plt.savefig(out_path / "mps_ir_per_label_f1.pdf")
    plt.savefig(out_path / "mps_ir_per_label_f1.png", dpi=300)
    plt.close()


def build_selected_spectrum_row(
    row_number,
    idx,
    formula,
    y_true,
    y_pred,
    metadata,
):
    diff = label_difference(y_true[idx], y_pred[idx])
    counts = error_counts(y_true[idx], y_pred[idx])
    meta = metadata[idx]

    formula_value = meta.get("formula")
    if formula_value is None or pd.isna(formula_value):
        formula_value = meta.get("molecular_formula")
    if formula_value is None or pd.isna(formula_value):
        formula_value = formula

    smiles_value = meta.get("smiles")
    if smiles_value is None or pd.isna(smiles_value):
        smiles_value = meta.get("canonical_smiles")

    return {
        "row_number": int(row_number),
        "test_index": int(idx),
        "case_type": classify_case(y_true[idx], y_pred[idx]),
        "true_labels": "; ".join(labels_from_binary(y_true[idx])),
        "predicted_binary_labels": "; ".join(labels_from_binary(y_pred[idx])),
        "correct_labels": "; ".join(diff["correct_labels"]),
        "false_positive_labels": "; ".join(diff["false_positive_labels"]),
        "false_negative_labels": "; ".join(diff["false_negative_labels"]),
        "false_positive_count": counts["false_positive_count"],
        "false_negative_count": counts["false_negative_count"],
        "total_error_count": counts["total_error_count"],
        "source_file": meta.get("source_file"),
        "source_row": meta.get("source_row"),
        "global_index": meta.get("global_index"),
        "formula": formula_value,
        "smiles": smiles_value,
    }


def plot_selected_formula_spectrum(
    row_number,
    idx,
    formula,
    matches,
    X_test_raw,
    y_true,
    y_pred,
    y_probs,
    metadata,
    match_strategy,
    out_path,
):
    case = classify_case(y_true[idx], y_pred[idx])
    diff = label_difference(y_true[idx], y_pred[idx])
    counts = error_counts(y_true[idx], y_pred[idx])
    meta = metadata[idx]

    true_labels = ", ".join(labels_from_binary(y_true[idx])) or "none"
    pred_binary = ", ".join(labels_from_binary(y_pred[idx])) or "none"
    top_probs = ", ".join(top_probability_labels(y_probs[idx], 5)) or "none"

    correct_labels = ", ".join(diff["correct_labels"]) or "none"
    fp_labels = ", ".join(diff["false_positive_labels"]) or "none"
    fn_labels = ", ".join(diff["false_negative_labels"]) or "none"

    x_axis = np.linspace(4000, 400, 600)

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, X_test_raw[idx])
    plt.gca().invert_xaxis()

    plt.title(
        f"{row_number}. {case.upper()} | formula: {formula} | test index: {idx}\n"
        f"matches in test split: {len(matches)} | match strategy: {match_strategy}\n"
        f"FP: {counts['false_positive_count']} | "
        f"FN: {counts['false_negative_count']} | "
        f"total errors: {counts['total_error_count']}\n"
        f"true: {true_labels}\n"
        f"binary pred: {pred_binary}\n"
        f"top probabilities: {top_probs}\n"
        f"correct: {correct_labels}\n"
        f"false positives: {fp_labels}\n"
        f"false negatives: {fn_labels}\n"
        f"{identifier_string(meta)}",
        fontsize=8,
    )

    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("Intensity")
    plt.tight_layout()

    safe_formula = sanitize_filename(formula)
    base_name = f"mps_ir_formula_spectrum_{row_number:02d}_{safe_formula}_testidx_{idx}"

    plt.savefig(out_path / f"{base_name}.pdf")
    plt.savefig(out_path / f"{base_name}.png", dpi=300)
    plt.close()

    return base_name


def select_target_spectra_by_formula(
    formulas,
    metadata,
    y_true,
    y_pred,
    match_strategy,
):
    selected = []
    skipped = []

    for row_number, formula in enumerate(formulas, start=1):
        matches = find_test_indices_by_formula(metadata, formula)

        if len(matches) == 0:
            print(
                f"No spectrum with formula '{formula}' was found in the TEST split. "
                "Skipping."
            )
            skipped.append(
                {
                    "row_number": row_number,
                    "formula": formula,
                    "reason": "not found in test split",
                }
            )
            continue

        idx = choose_formula_match(
            matches=matches,
            y_true=y_true,
            y_pred=y_pred,
            match_strategy=match_strategy,
        )

        selected.append(
            {
                "row_number": row_number,
                "formula": formula,
                "test_index": idx,
                "matches": matches,
            }
        )

    selected_indices = [item["test_index"] for item in selected]

    if len(set(selected_indices)) != len(selected_indices):
        print("The formula selection produced duplicate test indices.")

    print(f"Selected {len(selected)} formula(s). Skipped {len(skipped)} formula(s).")

    return selected, skipped


def save_selected_spectra(
    selected,
    X_test_raw,
    y_true,
    y_pred,
    y_probs,
    metadata,
    match_strategy,
    out_path,
):
    rows = []
    match_rows = []

    for item in selected:
        row_number = item["row_number"]
        formula = item["formula"]
        idx = item["test_index"]
        matches = item["matches"]

        plot_base_name = plot_selected_formula_spectrum(
            row_number=row_number,
            idx=idx,
            formula=formula,
            matches=matches,
            X_test_raw=X_test_raw,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            metadata=metadata,
            match_strategy=match_strategy,
            out_path=out_path,
        )

        row = build_selected_spectrum_row(
            row_number=row_number,
            idx=idx,
            formula=formula,
            y_true=y_true,
            y_pred=y_pred,
            metadata=metadata,
        )
        row["plot_png"] = f"{plot_base_name}.png"
        row["plot_pdf"] = f"{plot_base_name}.pdf"

        rows.append(row)

        for match_idx in matches:
            match_meta = metadata[match_idx]

            formula_value = match_meta.get("formula")
            if formula_value is None or pd.isna(formula_value):
                formula_value = match_meta.get("molecular_formula")

            smiles_value = match_meta.get("smiles")
            if smiles_value is None or pd.isna(smiles_value):
                smiles_value = match_meta.get("canonical_smiles")

            match_rows.append(
                {
                    "searched_formula": formula,
                    "selected": int(match_idx == idx),
                    "test_index": int(match_idx),
                    "case_type": classify_case(y_true[match_idx], y_pred[match_idx]),
                    "true_labels": "; ".join(labels_from_binary(y_true[match_idx])),
                    "predicted_binary_labels": "; ".join(
                        labels_from_binary(y_pred[match_idx])
                    ),
                    "source_file": match_meta.get("source_file"),
                    "source_row": match_meta.get("source_row"),
                    "global_index": match_meta.get("global_index"),
                    "formula": formula_value,
                    "smiles": smiles_value,
                }
            )

    selected_df = pd.DataFrame(rows)

    selected_df.to_csv(
        out_path / "mps_ir_selected_spectra.csv",
        index=False,
    )

    pd.DataFrame(match_rows).to_csv(
        out_path / "mps_ir_selected_spectra_all_formula_matches.csv",
        index=False,
    )

    return selected_df


@click.command()
@click.option(
    "--checkpoint_path",
    type=click.Path(exists=True, path_type=Path),
    default=PATH_CONFIG.best_model_path,
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, path_type=Path),
    default=PATH_CONFIG.data_dir,
)
@click.option(
    "--out_path",
    type=click.Path(path_type=Path),
    default="src/spectroscopy_qml/ir/mps_classifier/results_test/error_analysis",
)
@click.option("--device", type=str, default=None)
@click.option("--threshold", type=float, default=None)
@click.option("--top_n_confusion", type=int, default=20)
@click.option(
    "--formula",
    "formulas",
    multiple=True,
    default=TARGET_FORMULAS,
    help=(
        "Chemical formula to search in the test split. "
        "Can be passed multiple times. "
        "Missing formulas are skipped automatically."
    ),
)
@click.option(
    "--match_strategy",
    type=click.Choice(["first", "mixed", "worst", "correct"], case_sensitive=False),
    default="mixed",
    help=(
        "How to choose if multiple test spectra have the same formula. "
        "'mixed' prefers a sample with both false positives and false negatives."
    ),
)
def main(
    checkpoint_path,
    data_dir,
    out_path,
    device,
    threshold,
    top_n_confusion,
    formulas,
    match_strategy,
):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    formulas = tuple(formulas)

    print(f"Received {len(formulas)} formula(s) to search.")

    device = device or TRAINING_CONFIG.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    device = torch.device(device)
    print(f"Using device: {device}")

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model_from_checkpoint(checkpoint, device)

    if threshold is not None:
        thresholds = np.full(model.num_classes, threshold)
        print(f"Using fixed threshold: {threshold}")
    else:
        thresholds = checkpoint.get("thresholds", np.full(model.num_classes, 0.5))
        print(
            f"Using checkpoint thresholds: mean={thresholds.mean():.3f}, "
            f"std={thresholds.std():.3f}"
        )

    print("Loading processed IR data for inference...")
    X, y = load_ir_data(
        Path(data_dir),
        target_length=DATA_CONFIG.target_length,
        max_files=DATA_CONFIG.max_files,
        apply_snv=DATA_CONFIG.apply_snv,
        apply_savgol=DATA_CONFIG.apply_savgol,
        savgol_window_length=DATA_CONFIG.savgol_window_length,
        savgol_polyorder=DATA_CONFIG.savgol_polyorder,
    )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TRAINING_CONFIG.test_ratio,
        random_state=TRAINING_CONFIG.random_seed,
        shuffle=True,
    )

    test_dataset = IRSpectraDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
        persistent_workers=True if TRAINING_CONFIG.num_workers > 0 else False,
    )

    print("Running inference...")
    y_true, y_pred, y_probs = run_inference(model, test_loader, device, thresholds)

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)

    print("\nTest metrics")
    print(f"Micro-F1:        {micro_f1:.4f}")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"Precision micro: {precision_micro:.4f}")
    print(f"Recall micro:    {recall_micro:.4f}")

    np.savez(
        out_path / "mps_ir_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        thresholds=thresholds,
    )

    plot_label_confusion(y_true, y_pred, out_path, top_n=top_n_confusion)
    plot_per_label_f1(y_true, y_pred, out_path)

    print("Loading raw IR spectra and metadata...")
    X_test_raw, metadata = load_raw_spectra_with_metadata(
        data_dir=data_dir,
        seed=TRAINING_CONFIG.random_seed,
        test_size=TRAINING_CONFIG.test_ratio,
        max_files=DATA_CONFIG.max_files,
    )

    if len(X_test_raw) != len(y_true):
        print(
            f"Mismatch: loaded {len(X_test_raw)} raw spectra "
            f"but inference contains {len(y_true)} samples."
        )

    print("\nSelecting spectra by formula...")
    selected, skipped = select_target_spectra_by_formula(
        formulas=formulas,
        metadata=metadata,
        y_true=y_true,
        y_pred=y_pred,
        match_strategy=match_strategy,
    )

    if len(skipped) > 0:
        skipped_df = pd.DataFrame(skipped)
        skipped_df.to_csv(
            out_path / "mps_ir_skipped_formulas.csv",
            index=False,
        )
        print(f"Saved skipped formulas CSV: {out_path / 'mps_ir_skipped_formulas.csv'}")

    if len(selected) == 0:
        print("No requested formulas were found in the test split. Nothing to plot.")
        print(f"\nSaved available outputs to: {out_path}")
        return

    selected_df = save_selected_spectra(
        selected=selected,
        X_test_raw=X_test_raw,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        metadata=metadata,
        match_strategy=match_strategy,
        out_path=out_path,
    )

    print()
    print("=" * 80)
    print("Selected spectra")
    print("=" * 80)

    for _, row in selected_df.iterrows():
        print(
            f"{row['row_number']}. "
            f"formula={row['formula']} | "
            f"test_index={row['test_index']} | "
            f"case_type={row['case_type']} | "
            f"FP={row['false_positive_count']} | "
            f"FN={row['false_negative_count']} | "
            f"total={row['total_error_count']} | "
            f"source={row['source_file']}:{row['source_row']}"
        )

    print("=" * 80)

    print(f"\nSaved figures and CSVs to: {out_path}")
    print(f"Saved selected CSV: {out_path / 'mps_ir_selected_spectra.csv'}")
    print(
        "Saved all formula matches CSV: "
        f"{out_path / 'mps_ir_selected_spectra_all_formula_matches.csv'}"
    )


if __name__ == "__main__":
    main()