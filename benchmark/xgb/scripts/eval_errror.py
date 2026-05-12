from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from scipy.interpolate import interp1d
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")


functional_groups = {
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
    "Imine": Chem.MolFromSmarts(
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"
    ),
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

FUNCTIONAL_GROUP_NAMES = list(functional_groups.keys())

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

COLUMN_MAPPING = {
    "h_nmr_spectra": ("h_nmr_spectra", "hnmr"),
    "c_nmr_spectra": ("c_nmr_spectra", "cnmr"),
    "ir_spectra": ("ir_spectra", "ir"),
    "pos_msms": ("msms_positive_40ev", "pos_msms"),
    "neg_msms": ("msms_negative_40ev", "neg_msms"),
}

# Default SMILES to search in the rebuilt test split.
# You can leave this empty and pass targets via --smiles or --smiles_csv.
TARGET_SMILES: tuple[str, ...] = (
    "C#Cc1ccc(CN2CCOCC2)cc1",
    "CC1(C)C=CN(c2ccccc2)C(=O)C1",
    "CCNc1cc(-c2ccoc2)ccc1C",
    "COc1ccc(Nc2ncccc2C(=O)O)cc1Cl",
    "COc1ccccc1C(=O)Cn1cc(Cl)cnc1=O",
    "O=C(Nc1noc(C2CC2)c1Cl)Oc1ccccc1",
    "O=C(O)CC(=O)NCCc1ccccn1",
    "CN1C(=O)OC(O)N1Cc1ccccc1",
    "CCOc1cc(C#N)ccc1COON",
    "O=C1Cc2ccccc2N1c1ncc[nH]1",
    "O=C1Nc2ccccc2Cn2ccnc21",
    "Cn1cnc2c(O)nc3ccccc3c21",
    "O=C(O)c1cc(CCCc2ccccc2)on1",
    "CC1CC(=O)C=CN1C(=O)Oc1ccccc1",
    "CCOC(=O)C1=Nc2ccc(C)cc2C(=O)C1",
)


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def normalize_formula(value) -> str | None:
    if is_missing(value):
        return None
    return str(value).replace(" ", "").strip()


def normalize_smiles(value) -> str | None:
    """Return canonical RDKit SMILES, or None for invalid/missing input."""
    if is_missing(value):
        return None

    smiles = str(value).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def sanitize_filename(value) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def unique_smiles_preserve_order(values: Iterable[str | None]) -> tuple[str, ...]:
    seen = set()
    result = []

    for value in values:
        normalized = normalize_smiles(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    return tuple(result)


def match_group(mol: Chem.Mol, func_group) -> int:
    if func_group is None:
        return 0

    if isinstance(func_group, Chem.Mol):
        n_matches = len(mol.GetSubstructMatches(func_group))
    else:
        n_matches = func_group(mol)

    return 0 if n_matches == 0 else 1


def get_functional_groups(smiles: str) -> list[int] | None:
    if smiles is None:
        return None

    smiles = str(smiles).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return [match_group(mol, smarts) for smarts in functional_groups.values()]


def molecular_formula_from_smiles(smiles: str | None) -> str | None:
    if is_missing(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None

    return rdMolDescriptors.CalcMolFormula(mol)


def make_msms_spectrum(spectrum) -> np.ndarray | None:
    if spectrum is None:
        return None

    msms_spectrum = np.zeros(10000, dtype=np.float32)

    for peak in spectrum:
        if peak is None or len(peak) < 2:
            continue

        peak_pos = int(peak[0] * 10)
        peak_pos = min(max(peak_pos, 0), 9999)
        msms_spectrum[peak_pos] = peak[1]

    return msms_spectrum


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
    new_x = np.linspace(old_x.min(), old_x.max(), 600)
    return interp1d(old_x, spec, kind="linear")(new_x).astype(np.float32)


def labels_from_binary(row: np.ndarray) -> list[str]:
    idx = np.where(row == 1)[0]
    return [FUNCTIONAL_GROUP_NAMES[i] for i in idx]


def top_probability_labels(prob_row: np.ndarray | None, max_labels: int = 10) -> list[str]:
    if prob_row is None:
        return []

    safe_probs = np.nan_to_num(prob_row, nan=-np.inf)
    idx = np.argsort(safe_probs)[::-1][:max_labels]
    return [f"{FUNCTIONAL_GROUP_NAMES[i]} ({prob_row[i]:.3f})" for i in idx]


def label_difference(y_true_row: np.ndarray, y_pred_row: np.ndarray) -> dict[str, list[str]]:
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


def error_counts(y_true_row: np.ndarray, y_pred_row: np.ndarray) -> dict[str, int]:
    fp = int(np.logical_and(y_pred_row == 1, y_true_row == 0).sum())
    fn = int(np.logical_and(y_pred_row == 0, y_true_row == 1).sum())

    return {
        "false_positive_count": fp,
        "false_negative_count": fn,
        "total_error_count": fp + fn,
    }


def classify_case(y_true_row: np.ndarray, y_pred_row: np.ndarray) -> str:
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


def identifier_string(meta: dict) -> str:
    parts = []

    for col in IDENTIFIER_COLUMNS + ["computed_molecular_formula", "normalized_smiles"]:
        value = meta.get(col)
        if not is_missing(value):
            parts.append(f"{col}: {value}")

    parts.append(f"source_file: {meta['source_file']}")
    parts.append(f"source_row: {meta['source_row']}")
    parts.append(f"global_index: {meta['global_index']}")

    return " | ".join(parts)


def get_parquet_columns(parquet_file: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(parquet_file).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(parquet_file).columns)


def deduplicate_columns(columns: Iterable[str]) -> list[str]:
    seen = set()
    result = []

    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        result.append(col)

    return result


def load_data_for_column_with_metadata(
    analytical_data: Path,
    actual_col: str,
    max_files: int | None = None,
    sort_files: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    spectra = []
    labels = []
    metadata = []

    parquet_files = list(Path(analytical_data).glob("*.parquet"))
    if sort_files:
        parquet_files = sorted(parquet_files)

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    print(f"Found {len(parquet_files)} parquet files")

    for file_idx, parquet_file in enumerate(parquet_files, start=1):
        available_columns = get_parquet_columns(parquet_file)

        if actual_col not in available_columns:
            print(f"Skipping {parquet_file.name}: missing column '{actual_col}'")
            continue

        if "smiles" not in available_columns:
            print(f"Skipping {parquet_file.name}: missing required column 'smiles'")
            continue

        identifier_columns = [c for c in IDENTIFIER_COLUMNS if c in available_columns]
        columns_to_load = deduplicate_columns(["smiles", actual_col, *identifier_columns])

        print(
            f"Loading file {file_idx}/{len(parquet_files)}: {parquet_file.name}...",
            end=" ",
            flush=True,
        )

        data = pd.read_parquet(parquet_file, columns=columns_to_load)
        data["_source_file"] = parquet_file.name
        data["_source_row"] = data.index.astype(int)

        print(f"[{len(data)} samples]", flush=True)

        if actual_col in ["msms_positive_40ev", "msms_negative_40ev"]:
            data[actual_col] = [make_msms_spectrum(s) for s in data[actual_col].to_list()]

        data["func_group"] = [get_functional_groups(s) for s in data["smiles"].to_list()]
        data[actual_col] = [interpolate_to_600(s) for s in data[actual_col].to_list()]
        data = data.dropna(subset=[actual_col, "func_group"])

        for _, row in data.iterrows():
            spectra.append(row[actual_col])
            labels.append(row["func_group"])

            meta = {
                "source_file": row["_source_file"],
                "source_row": int(row["_source_row"]),
                "global_index": len(spectra) - 1,
            }

            for col in identifier_columns:
                meta[col] = row[col]

            smiles_value = meta.get("smiles") or row.get("smiles")
            if "smiles" not in meta:
                meta["smiles"] = smiles_value

            meta["normalized_smiles"] = normalize_smiles(smiles_value)
            meta["computed_molecular_formula"] = molecular_formula_from_smiles(smiles_value)
            metadata.append(meta)

        del data

    if len(spectra) == 0:
        raise ValueError(f"No usable samples were loaded for column: {actual_col}")

    X = np.stack(spectra).astype(np.float32)
    y = np.stack(labels).astype(np.int32)

    return X, y, metadata


def load_pickle_model(model_path: Path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


def set_xgboost_device(model, device: str | None):
    if not device:
        return model

    estimators = getattr(model, "estimators_", None)
    targets = estimators if estimators is not None else [model]

    for estimator in targets:
        if estimator is None or not hasattr(estimator, "set_params"):
            continue
        try:
            estimator.set_params(device=device)
        except Exception as exc:
            print(f"Could not set XGBoost device='{device}' for one estimator: {exc}")

    return model


def positive_class_probability(estimator, X: np.ndarray) -> np.ndarray:
    probs = estimator.predict_proba(X)

    if probs.ndim == 1:
        return probs.astype(float)

    classes = list(getattr(estimator, "classes_", []))
    if 1 in classes:
        return probs[:, classes.index(1)].astype(float)

    if probs.shape[1] == 2:
        return probs[:, 1].astype(float)

    return np.zeros(X.shape[0], dtype=float)


def predict_proba_multioutput(model, X: np.ndarray) -> np.ndarray | None:
    estimators = getattr(model, "estimators_", None)

    if estimators is not None:
        probs = [positive_class_probability(estimator, X) for estimator in estimators]
        return np.column_stack(probs)

    if not hasattr(model, "predict_proba"):
        return None

    raw_probs = model.predict_proba(X)

    if isinstance(raw_probs, list):
        probs = []
        for label_probs in raw_probs:
            if label_probs.ndim == 1:
                probs.append(label_probs.astype(float))
            elif label_probs.shape[1] >= 2:
                probs.append(label_probs[:, 1].astype(float))
            else:
                probs.append(np.zeros(X.shape[0], dtype=float))
        return np.column_stack(probs)

    if raw_probs.ndim == 3:
        return raw_probs[:, :, 1].astype(float)

    return raw_probs.astype(float)


def run_inference(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    all_probs = []
    all_preds = []

    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))
        X_batch = X_test[start:end]

        probs = predict_proba_multioutput(model, X_batch)
        if probs is None:
            preds = model.predict(X_batch).astype(int)
        else:
            preds = (probs >= threshold).astype(int)
            all_probs.append(probs)

        all_preds.append(preds)

    y_pred = np.vstack(all_preds).astype(int)
    y_probs = np.vstack(all_probs) if all_probs else None

    return y_test.astype(int), y_pred, y_probs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match_ratio": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = []

    for i, name in enumerate(FUNCTIONAL_GROUP_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        tp = int(np.logical_and(yt == 1, yp == 1).sum())
        fp = int(np.logical_and(yt == 0, yp == 1).sum())
        fn = int(np.logical_and(yt == 1, yp == 0).sum())
        tn = int(np.logical_and(yt == 0, yp == 0).sum())

        rows.append(
            {
                "label": name,
                "support": int(yt.sum()),
                "predicted_support": int(yp.sum()),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "threshold": threshold,
            }
        )

    return pd.DataFrame(rows)


def plot_label_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    prefix: str,
    top_n: int = 20,
):
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

    if len(active) == 0:
        print("Skipping confusion heatmap: no active labels in y_true.")
        return

    error_mass = confusion.sum(axis=1)
    n_selected = min(top_n, len(active))
    selected = active[np.argsort(error_mass[active])[-n_selected:]]
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
    plt.title("Functional-group confusion structure — XGBoost")
    plt.tight_layout()

    plt.savefig(out_path / f"{prefix}_confusion_heatmap.pdf")
    plt.savefig(out_path / f"{prefix}_confusion_heatmap.png", dpi=300)
    plt.close()


def plot_per_label_f1(per_label_df: pd.DataFrame, out_path: Path, prefix: str):
    plot_df = per_label_df.sort_values("f1", ascending=True)

    plt.figure(figsize=(11, 6))
    plt.bar(range(len(plot_df)), plot_df["f1"].to_numpy())
    plt.xticks(range(len(plot_df)), plot_df["label"].to_list(), rotation=90, fontsize=8)
    plt.ylabel("F1")
    plt.title("Per-label F1 scores — XGBoost")
    plt.tight_layout()

    plt.savefig(out_path / f"{prefix}_per_label_f1.pdf")
    plt.savefig(out_path / f"{prefix}_per_label_f1.png", dpi=300)
    plt.close()


def plot_per_label_precision_recall(per_label_df: pd.DataFrame, out_path: Path, prefix: str):
    plot_df = per_label_df.sort_values("f1", ascending=True)
    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, plot_df["precision"].to_numpy(), width=width, label="Precision")
    plt.bar(x + width / 2, plot_df["recall"].to_numpy(), width=width, label="Recall")
    plt.xticks(x, plot_df["label"].to_list(), rotation=90, fontsize=8)
    plt.ylabel("Score")
    plt.title("Per-label precision and recall — XGBoost")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path / f"{prefix}_per_label_precision_recall.pdf")
    plt.savefig(out_path / f"{prefix}_per_label_precision_recall.png", dpi=300)
    plt.close()


def plot_error_type_counts(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, prefix: str):
    case_types = [classify_case(t, p) for t, p in zip(y_true, y_pred)]
    counts = pd.Series(case_types).value_counts().reindex(
        ["correct", "false positive", "false negative", "mixed error"],
        fill_value=0,
    )

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Number of spectra")
    plt.title("Prediction error types — XGBoost")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(out_path / f"{prefix}_error_type_counts.pdf")
    plt.savefig(out_path / f"{prefix}_error_type_counts.png", dpi=300)
    plt.close()

    counts.rename_axis("case_type").reset_index(name="count").to_csv(
        out_path / f"{prefix}_error_type_counts.csv",
        index=False,
    )


def plot_sample_error_histogram(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, prefix: str):
    total_errors = np.array([error_counts(t, p)["total_error_count"] for t, p in zip(y_true, y_pred)])
    max_errors = int(total_errors.max()) if len(total_errors) else 0
    bins = np.arange(max_errors + 2) - 0.5

    plt.figure(figsize=(8, 5))
    plt.hist(total_errors, bins=bins)
    plt.xlabel("False positives + false negatives per spectrum")
    plt.ylabel("Number of spectra")
    plt.title("Per-spectrum error count distribution — XGBoost")
    plt.tight_layout()

    plt.savefig(out_path / f"{prefix}_sample_error_histogram.pdf")
    plt.savefig(out_path / f"{prefix}_sample_error_histogram.png", dpi=300)
    plt.close()


def plot_support_vs_f1(per_label_df: pd.DataFrame, out_path: Path, prefix: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(per_label_df["support"], per_label_df["f1"])

    support_quantile = per_label_df["support"].quantile(0.2)
    f1_quantile = per_label_df["f1"].quantile(0.2)

    for _, row in per_label_df.iterrows():
        if row["support"] <= support_quantile or row["f1"] <= f1_quantile:
            plt.annotate(row["label"], (row["support"], row["f1"]), fontsize=7)

    plt.xlabel("Label support in test set")
    plt.ylabel("F1")
    plt.title("Support vs. F1 — XGBoost")
    plt.tight_layout()

    plt.savefig(out_path / f"{prefix}_support_vs_f1.pdf")
    plt.savefig(out_path / f"{prefix}_support_vs_f1.png", dpi=300)
    plt.close()


def plot_probability_distributions(
    y_true: np.ndarray,
    y_probs: np.ndarray | None,
    out_path: Path,
    prefix: str,
):
    if y_probs is None:
        print("Skipping probability histogram: model probabilities are unavailable.")
        return

    positive_probs = y_probs[y_true == 1]
    negative_probs = y_probs[y_true == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(negative_probs, bins=40, alpha=0.6, label="True negative label positions")
    plt.hist(positive_probs, bins=40, alpha=0.6, label="True positive label positions")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Predicted probability distributions — XGBoost")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path / f"{prefix}_probability_distributions.pdf")
    plt.savefig(out_path / f"{prefix}_probability_distributions.png", dpi=300)
    plt.close()


def first_available_formula(meta: dict, fallback: str | None = None) -> str | None:
    for key in ["formula", "molecular_formula", "computed_molecular_formula"]:
        value = normalize_formula(meta.get(key))
        if value:
            return value
    return fallback


def first_available_smiles(meta: dict) -> str | None:
    for key in ["normalized_smiles", "canonical_smiles", "smiles"]:
        value = meta.get(key)
        normalized = normalize_smiles(value)
        if normalized:
            return normalized
    return None


def save_global_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None,
    threshold: float,
    metadata: list[dict],
    out_path: Path,
    prefix: str,
    save_all_predictions: bool,
):
    metrics = compute_metrics(y_true, y_pred)
    metrics_df = pd.DataFrame([{**metrics, "threshold": threshold}])
    metrics_df.to_csv(out_path / f"{prefix}_metrics.csv", index=False)

    per_label_df = compute_per_label_metrics(y_true, y_pred, threshold)
    per_label_df.to_csv(out_path / f"{prefix}_per_label_metrics.csv", index=False)

    npz_kwargs = {
        "y_true": y_true,
        "y_pred": y_pred,
        "threshold": np.array([threshold], dtype=float),
    }
    if y_probs is not None:
        npz_kwargs["y_probs"] = y_probs

    np.savez(out_path / f"{prefix}_predictions.npz", **npz_kwargs)

    if save_all_predictions:
        rows = []
        for idx, meta in enumerate(metadata):
            diff = label_difference(y_true[idx], y_pred[idx])
            counts = error_counts(y_true[idx], y_pred[idx])
            row_probs = y_probs[idx] if y_probs is not None else None

            rows.append(
                {
                    "test_index": idx,
                    "case_type": classify_case(y_true[idx], y_pred[idx]),
                    "true_labels": "; ".join(labels_from_binary(y_true[idx])),
                    "predicted_binary_labels": "; ".join(labels_from_binary(y_pred[idx])),
                    "top_probability_labels": "; ".join(top_probability_labels(row_probs, 10)),
                    "correct_labels": "; ".join(diff["correct_labels"]),
                    "false_positive_labels": "; ".join(diff["false_positive_labels"]),
                    "false_negative_labels": "; ".join(diff["false_negative_labels"]),
                    "false_positive_count": counts["false_positive_count"],
                    "false_negative_count": counts["false_negative_count"],
                    "total_error_count": counts["total_error_count"],
                    "source_file": meta.get("source_file"),
                    "source_row": meta.get("source_row"),
                    "global_index": meta.get("global_index"),
                    "formula": first_available_formula(meta),
                    "smiles": first_available_smiles(meta),
                }
            )

        pd.DataFrame(rows).to_csv(out_path / f"{prefix}_all_test_predictions.csv", index=False)

    return metrics, per_label_df


def find_test_indices_by_smiles(metadata: list[dict], smiles: str) -> list[int]:
    target = normalize_smiles(smiles)
    matches = []

    if target is None:
        return matches

    for test_idx, meta in enumerate(metadata):
        candidate = first_available_smiles(meta)
        if candidate == target:
            matches.append(test_idx)

    return matches


def choose_match(
    matches: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    match_strategy: str,
) -> int:
    if len(matches) == 0:
        raise ValueError("choose_match() received an empty matches list.")

    fp = np.logical_and(y_pred == 1, y_true == 0).sum(axis=1)
    fn = np.logical_and(y_pred == 0, y_true == 1).sum(axis=1)
    total = fp + fn

    matches_array = np.asarray(matches)

    if match_strategy == "first":
        return int(matches_array[0])

    if match_strategy == "mixed":
        mixed = matches_array[(fp[matches_array] > 0) & (fn[matches_array] > 0)]
        if len(mixed) > 0:
            return int(mixed[np.argmax(total[mixed])])
        print("No mixed case found for this SMILES. Falling back to worst error.")

    if match_strategy in ["worst", "mixed"]:
        return int(matches_array[np.argmax(total[matches_array])])

    if match_strategy == "correct":
        correct = matches_array[total[matches_array] == 0]
        if len(correct) > 0:
            return int(correct[0])
        print("No correct case found for this SMILES. Falling back to first match.")
        return int(matches_array[0])

    return int(matches_array[0])


def build_selected_spectrum_row(
    row_number: int,
    idx: int,
    searched_smiles: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None,
    metadata: list[dict],
) -> dict:
    diff = label_difference(y_true[idx], y_pred[idx])
    counts = error_counts(y_true[idx], y_pred[idx])
    meta = metadata[idx]
    row_probs = y_probs[idx] if y_probs is not None else None

    return {
        "row_number": int(row_number),
        "test_index": int(idx),
        "case_type": classify_case(y_true[idx], y_pred[idx]),
        "true_labels": "; ".join(labels_from_binary(y_true[idx])),
        "predicted_binary_labels": "; ".join(labels_from_binary(y_pred[idx])),
        "top_probability_labels": "; ".join(top_probability_labels(row_probs, 10)),
        "correct_labels": "; ".join(diff["correct_labels"]),
        "false_positive_labels": "; ".join(diff["false_positive_labels"]),
        "false_negative_labels": "; ".join(diff["false_negative_labels"]),
        "false_positive_count": counts["false_positive_count"],
        "false_negative_count": counts["false_negative_count"],
        "total_error_count": counts["total_error_count"],
        "source_file": meta.get("source_file"),
        "source_row": meta.get("source_row"),
        "global_index": meta.get("global_index"),
        "searched_smiles": normalize_smiles(searched_smiles),
        "matched_smiles": first_available_smiles(meta),
        "formula": first_available_formula(meta),
    }


def axis_config(actual_col: str) -> tuple[np.ndarray, str, str]:
    if actual_col == "ir_spectra":
        return np.linspace(4000, 400, 600), "Wavenumber / cm$^{-1}$", "Intensity"

    if actual_col == "h_nmr_spectra":
        return np.linspace(12, 0, 600), r"$^1$H chemical shift / ppm", "Intensity"

    if actual_col == "c_nmr_spectra":
        return np.linspace(220, 0, 600), r"$^{13}$C chemical shift / ppm", "Intensity"

    if actual_col in ["msms_positive_40ev", "msms_negative_40ev"]:
        return np.linspace(0, 999.9, 600), "m/z", "Intensity"

    return np.arange(600), "Interpolated point index", "Value"


def plot_selected_smiles_spectrum(
    row_number: int,
    idx: int,
    searched_smiles: str,
    matches: list[int],
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None,
    metadata: list[dict],
    match_strategy: str,
    actual_col: str,
    out_path: Path,
    prefix: str,
) -> str:
    case = classify_case(y_true[idx], y_pred[idx])
    diff = label_difference(y_true[idx], y_pred[idx])
    counts = error_counts(y_true[idx], y_pred[idx])
    meta = metadata[idx]

    matched_smiles = first_available_smiles(meta) or "unknown"
    formula = first_available_formula(meta) or "unknown"

    true_labels = ", ".join(labels_from_binary(y_true[idx])) or "none"
    pred_binary = ", ".join(labels_from_binary(y_pred[idx])) or "none"
    row_probs = y_probs[idx] if y_probs is not None else None
    top_probs = ", ".join(top_probability_labels(row_probs, 5)) or "none"

    correct_labels = ", ".join(diff["correct_labels"]) or "none"
    fp_labels = ", ".join(diff["false_positive_labels"]) or "none"
    fn_labels = ", ".join(diff["false_negative_labels"]) or "none"

    x_axis, x_label, y_label = axis_config(actual_col)

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, X_test[idx])

    if actual_col == "ir_spectra":
        plt.gca().invert_xaxis()

    plt.title(
        f"{row_number}. {case.upper()} | searched SMILES: {searched_smiles} | test index: {idx}\n"
        f"matched SMILES: {matched_smiles} | formula: {formula}\n"
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

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    safe_smiles = sanitize_filename(searched_smiles)
    base_name = f"{prefix}_smiles_spectrum_{row_number:02d}_{safe_smiles}_testidx_{idx}"

    plt.savefig(out_path / f"{base_name}.pdf")
    plt.savefig(out_path / f"{base_name}.png", dpi=300)
    plt.close()

    return base_name


def select_target_spectra_by_smiles(
    smiles_targets: Iterable[str],
    metadata: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    match_strategy: str,
) -> tuple[list[dict], list[dict]]:
    selected = []
    skipped = []

    for row_number, smiles in enumerate(smiles_targets, start=1):
        normalized = normalize_smiles(smiles)
        if normalized is None:
            print(f"Invalid SMILES '{smiles}'. Skipping.")
            skipped.append(
                {
                    "row_number": row_number,
                    "smiles": smiles,
                    "reason": "invalid SMILES",
                }
            )
            continue

        matches = find_test_indices_by_smiles(metadata, normalized)

        if len(matches) == 0:
            print(f"No spectrum with SMILES '{normalized}' was found in the TEST split. Skipping.")
            skipped.append(
                {
                    "row_number": row_number,
                    "smiles": normalized,
                    "reason": "not found in test split",
                }
            )
            continue

        idx = choose_match(
            matches=matches,
            y_true=y_true,
            y_pred=y_pred,
            match_strategy=match_strategy,
        )

        selected.append(
            {
                "row_number": row_number,
                "smiles": normalized,
                "test_index": idx,
                "matches": matches,
            }
        )

    selected_indices = [item["test_index"] for item in selected]

    if len(set(selected_indices)) != len(selected_indices):
        print("The SMILES selection produced duplicate test indices.")

    print(f"Selected {len(selected)} SMILES target(s). Skipped {len(skipped)} SMILES target(s).")

    return selected, skipped


def save_selected_spectra(
    selected: list[dict],
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None,
    metadata: list[dict],
    match_strategy: str,
    actual_col: str,
    out_path: Path,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    match_rows = []

    for item in selected:
        row_number = item["row_number"]
        searched_smiles = item["smiles"]
        idx = item["test_index"]
        matches = item["matches"]

        plot_base_name = plot_selected_smiles_spectrum(
            row_number=row_number,
            idx=idx,
            searched_smiles=searched_smiles,
            matches=matches,
            X_test=X_test,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            metadata=metadata,
            match_strategy=match_strategy,
            actual_col=actual_col,
            out_path=out_path,
            prefix=prefix,
        )

        row = build_selected_spectrum_row(
            row_number=row_number,
            idx=idx,
            searched_smiles=searched_smiles,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            metadata=metadata,
        )
        row["plot_png"] = f"{plot_base_name}.png"
        row["plot_pdf"] = f"{plot_base_name}.pdf"
        rows.append(row)

        for match_idx in matches:
            match_meta = metadata[match_idx]
            match_rows.append(
                {
                    "searched_smiles": searched_smiles,
                    "selected": int(match_idx == idx),
                    "test_index": int(match_idx),
                    "case_type": classify_case(y_true[match_idx], y_pred[match_idx]),
                    "true_labels": "; ".join(labels_from_binary(y_true[match_idx])),
                    "predicted_binary_labels": "; ".join(labels_from_binary(y_pred[match_idx])),
                    "top_probability_labels": "; ".join(
                        top_probability_labels(y_probs[match_idx] if y_probs is not None else None, 10)
                    ),
                    "source_file": match_meta.get("source_file"),
                    "source_row": match_meta.get("source_row"),
                    "global_index": match_meta.get("global_index"),
                    "formula": first_available_formula(match_meta),
                    "smiles": first_available_smiles(match_meta),
                }
            )

    selected_df = pd.DataFrame(rows)
    selected_df.to_csv(out_path / f"{prefix}_selected_spectra.csv", index=False)

    pd.DataFrame(match_rows).to_csv(
        out_path / f"{prefix}_selected_spectra_all_smiles_matches.csv",
        index=False,
    )

    return selected_df


def read_smiles_from_csv(csv_path: Path, smiles_column: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES CSV does not contain column '{smiles_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    return [smiles for smiles in (normalize_smiles(v) for v in df[smiles_column]) if smiles]


@click.command()
@click.option("--analytical_data", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--out_path",
    type=click.Path(path_type=Path),
    default="xgboost_error_analysis",
)
@click.option(
    "--column",
    type=click.Choice(list(COLUMN_MAPPING.keys()), case_sensitive=False),
    default="ir_spectra",
    help="Input modality/column whose trained XGBoost model should be analyzed.",
)
@click.option("--seed", type=int, default=42)
@click.option("--test_size", type=float, default=0.1)
@click.option("--threshold", type=float, default=0.5)
@click.option("--batch_size", type=int, default=4096)
@click.option("--max_files", type=int, default=None)
@click.option(
    "--sort_files/--preserve_file_order",
    default=False,
    help=(
        "Use --preserve_file_order to mirror your training script's list(glob(...)) order. "
        "Use --sort_files only if the model was trained with sorted parquet files."
    ),
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Optional XGBoost prediction device override, e.g. 'cpu' or 'cuda'.",
)
@click.option("--top_n_confusion", type=int, default=20)
@click.option(
    "--smiles",
    "smiles_targets",
    multiple=True,
    default=(),
    help="SMILES string to search in the test split. Can be passed multiple times.",
)
@click.option(
    "--smiles_csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional CSV containing SMILES to search.",
)
@click.option(
    "--smiles_column",
    type=str,
    default="smiles",
    help="Column name used in --smiles_csv.",
)
@click.option(
    "--match_strategy",
    type=click.Choice(["first", "mixed", "worst", "correct"], case_sensitive=False),
    default="mixed",
    help=(
        "How to choose if multiple test spectra have the same SMILES. "
        "'mixed' prefers a sample with both false positives and false negatives."
    ),
)
@click.option(
    "--save_all_predictions/--no_save_all_predictions",
    default=True,
    help="Save one CSV row per test spectrum.",
)
def main(
    analytical_data: Path,
    model_path: Path,
    out_path: Path,
    column: str,
    seed: int,
    test_size: float,
    threshold: float,
    batch_size: int,
    max_files: int | None,
    sort_files: bool,
    device: str | None,
    top_n_confusion: int,
    smiles_targets: tuple[str, ...],
    smiles_csv: Path | None,
    smiles_column: str,
    match_strategy: str,
    save_all_predictions: bool,
):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    column = column.lower()
    actual_col, output_dir = COLUMN_MAPPING[column]
    prefix = f"xgboost_{output_dir}"

    smiles_candidates = []
    smiles_candidates.extend(TARGET_SMILES)

    if smiles_csv is not None:
        smiles_candidates.extend(read_smiles_from_csv(smiles_csv, smiles_column))

    smiles_candidates.extend(smiles_targets)
    target_smiles = unique_smiles_preserve_order(smiles_candidates)

    print(f"Analyzing XGBoost model for column: {column} ({actual_col})")
    print(f"Model path: {model_path}")
    print(f"Output path: {out_path}")
    print(f"Received {len(target_smiles)} SMILES target(s) to search.")

    print("\nLoading XGBoost model...")
    model = load_pickle_model(model_path)
    model = set_xgboost_device(model, device)

    print("\nLoading analytical data and rebuilding labels...")
    X_data, y_data, metadata = load_data_for_column_with_metadata(
        analytical_data=analytical_data,
        actual_col=actual_col,
        max_files=max_files,
        sort_files=sort_files,
    )

    print(f"Input shape: {X_data.shape}")
    print(f"Target shape: {y_data.shape}")
    print(f"Metadata rows: {len(metadata)}")

    X_train_full, X_test, y_train_full, y_test, train_metadata, test_metadata = train_test_split(
        X_data,
        y_data,
        metadata,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    print(
        f"Split: Train-full={len(X_train_full)} ({1 - test_size:.0%}), "
        f"Test={len(X_test)} ({test_size:.0%})"
    )

    del X_train_full, y_train_full, train_metadata

    print("\nRunning inference...")
    y_true, y_pred, y_probs = run_inference(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        batch_size=batch_size,
    )

    if y_pred.shape[1] != len(FUNCTIONAL_GROUP_NAMES):
        raise ValueError(
            f"Prediction label dimension {y_pred.shape[1]} does not match "
            f"functional group count {len(FUNCTIONAL_GROUP_NAMES)}."
        )

    metrics, per_label_df = save_global_outputs(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        threshold=threshold,
        metadata=test_metadata,
        out_path=out_path,
        prefix=prefix,
        save_all_predictions=save_all_predictions,
    )

    print("\nTest metrics")
    print(f"Micro-F1:          {metrics['micro_f1']:.4f}")
    print(f"Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"Precision micro:   {metrics['precision_micro']:.4f}")
    print(f"Recall micro:      {metrics['recall_micro']:.4f}")
    print(f"Hamming loss:      {metrics['hamming_loss']:.4f}")
    print(f"Exact match ratio: {metrics['exact_match_ratio']:.4f}")

    print("\nCreating global plots...")
    plot_label_confusion(y_true, y_pred, out_path, prefix, top_n=top_n_confusion)
    plot_per_label_f1(per_label_df, out_path, prefix)
    plot_per_label_precision_recall(per_label_df, out_path, prefix)
    plot_error_type_counts(y_true, y_pred, out_path, prefix)
    plot_sample_error_histogram(y_true, y_pred, out_path, prefix)
    plot_support_vs_f1(per_label_df, out_path, prefix)
    plot_probability_distributions(y_true, y_probs, out_path, prefix)

    if len(target_smiles) == 0:
        print("\nNo SMILES targets provided. Global metrics and plots were saved only.")
        print(f"Saved outputs to: {out_path}")
        return

    print("\nSelecting spectra by SMILES...")
    selected, skipped = select_target_spectra_by_smiles(
        smiles_targets=target_smiles,
        metadata=test_metadata,
        y_true=y_true,
        y_pred=y_pred,
        match_strategy=match_strategy,
    )

    if len(skipped) > 0:
        skipped_df = pd.DataFrame(skipped)
        skipped_path = out_path / f"{prefix}_skipped_smiles.csv"
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Saved skipped SMILES CSV: {skipped_path}")

    if len(selected) == 0:
        print("No requested SMILES targets were found in the test split. Nothing to plot.")
        print(f"Saved available outputs to: {out_path}")
        return

    selected_df = save_selected_spectra(
        selected=selected,
        X_test=X_test,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        metadata=test_metadata,
        match_strategy=match_strategy,
        actual_col=actual_col,
        out_path=out_path,
        prefix=prefix,
    )

    print()
    print("=" * 80)
    print("Selected spectra")
    print("=" * 80)

    for _, row in selected_df.iterrows():
        print(
            f"{row['row_number']}. "
            f"searched_smiles={row['searched_smiles']} | "
            f"matched_smiles={row['matched_smiles']} | "
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
    print(f"Saved selected CSV: {out_path / f'{prefix}_selected_spectra.csv'}")
    print(
        "Saved all SMILES matches CSV: "
        f"{out_path / f'{prefix}_selected_spectra_all_smiles_matches.csv'}"
    )


if __name__ == "__main__":
    main()
