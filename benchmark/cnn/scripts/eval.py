import pickle
import textwrap
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


FUNCTIONAL_GROUP_NAMES = [
    "Acid anhydride", "Acyl halide", "Alcohol", "Aldehyde", "Alkane",
    "Alkene", "Alkyne", "Amide", "Amine", "Arene", "Azo compound",
    "Carbamate", "Carboxylic acid", "Enamine", "Enol", "Ester", "Ether",
    "Haloalkane", "Hydrazine", "Hydrazone", "Imide", "Imine", "Isocyanate",
    "Isothiocyanate", "Ketone", "Nitrile", "Phenol", "Phosphine", "Sulfide",
    "Sulfonamide", "Sulfonate", "Sulfone", "Sulfonic acid", "Sulfoxide",
    "Thial", "Thioamide", "Thiol"
]

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


def load_results(results_path):
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    if "test_predictions" in results:
        y_pred = results["test_predictions"]
        y_true = results["test_targets"]
    else:
        y_pred = results["pred"]
        y_true = results["tgt"]

    return np.asarray(y_true), np.asarray(y_pred)


def labels_from_vector(vector):
    idx = np.where(vector == 1)[0]
    return [FUNCTIONAL_GROUP_NAMES[i] for i in idx]


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


def classify_case(y_true_row, y_pred_row):
    fp = np.logical_and(y_pred_row == 1, y_true_row == 0).sum()
    fn = np.logical_and(y_pred_row == 0, y_true_row == 1).sum()

    if fp == 0 and fn == 0:
        return "correct"
    if fp > 0 and fn > 0:
        return "mixed error"
    if fp > 0:
        return "false positive"
    return "false negative"


def select_spectrum_indices(y_true, y_pred, preferred_case="mixed", n_spectra=500):
    """
    Select multiple spectra for analysis.

    The preferred case is used first. If fewer than n_spectra are available,
    the function fills the remaining slots using the fallback priority:
    mixed errors -> false positives -> false negatives -> correct spectra.

    Returns
    -------
    np.ndarray
        Array of selected test-set indices.
    """
    false_pos = np.logical_and(y_pred == 1, y_true == 0).sum(axis=1)
    false_neg = np.logical_and(y_pred == 0, y_true == 1).sum(axis=1)
    total_errors = false_pos + false_neg
    true_label_count = y_true.sum(axis=1)

    def rank_candidates(candidates, score):
        candidates = np.asarray(candidates, dtype=int)
        if len(candidates) == 0:
            return candidates

        order = np.argsort(-score[candidates], kind="mergesort")
        return candidates[order]

    mixed_candidates = rank_candidates(
        np.where((false_pos > 0) & (false_neg > 0))[0],
        total_errors,
    )

    false_positive_candidates = rank_candidates(
        np.where(false_pos > 0)[0],
        false_pos,
    )

    false_negative_candidates = rank_candidates(
        np.where(false_neg > 0)[0],
        false_neg,
    )

    correct_candidates = rank_candidates(
        np.where(total_errors == 0)[0],
        true_label_count,
    )

    preferred_map = {
        "mixed": mixed_candidates,
        "false_positive": false_positive_candidates,
        "false_negative": false_negative_candidates,
        "correct": correct_candidates,
    }

    preferred_case = preferred_case.lower()
    selected = []
    seen = set()

    def add_until_full(candidates):
        for idx in candidates:
            idx = int(idx)
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

            if len(selected) >= n_spectra:
                break

    add_until_full(preferred_map[preferred_case])

    fallback_order = [
        mixed_candidates,
        false_positive_candidates,
        false_negative_candidates,
        correct_candidates,
    ]

    for candidates in fallback_order:
        if len(selected) >= n_spectra:
            break
        add_until_full(candidates)

    if len(selected) == 0:
        raise ValueError("No spectra could be selected.")

    return np.asarray(selected, dtype=int)


def is_missing_value(value):
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def get_identifier_string(meta):
    parts = []

    for col in IDENTIFIER_COLUMNS:
        value = meta.get(col)
        if not is_missing_value(value):
            parts.append(f"{col}: {value}")

    parts.append(f"source_file: {meta.get('source_file')}")
    parts.append(f"source_row: {meta.get('source_row')}")
    parts.append(f"global_index: {meta.get('global_index')}")

    return " | ".join(parts)


def load_ir_test_spectra_with_metadata(analytical_data, seed, test_size):
    spectra = []
    metadata = []

    parquet_files = sorted(Path(analytical_data).glob("*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in analytical_data path: {analytical_data}")

    for parquet_file in parquet_files:
        data_full = pd.read_parquet(parquet_file)
        identifier_columns = [c for c in IDENTIFIER_COLUMNS if c in data_full.columns]

        required_columns = ["ir_spectra"] + identifier_columns
        data = data_full[required_columns]

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

    X = np.stack(spectra)
    indices = np.arange(len(X))

    _, X_test, _, test_indices = train_test_split(
        X,
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    test_metadata = [metadata[i] for i in test_indices]

    return X_test, test_metadata


def plot_label_confusion(y_true, y_pred, out_path, top_n=20):
    n_labels = y_true.shape[1]
    confusion = np.zeros((n_labels, n_labels), dtype=float)

    for t, p in zip(y_true, y_pred):
        true_idx = np.where(t == 1)[0]
        pred_idx = np.where(p == 1)[0]

        for i in true_idx:
            for j in pred_idx:
                if i != j and t[j] == 0:
                    confusion[i, j] += 1

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion),
        where=row_sums != 0,
    )

    label_support = y_true.sum(axis=0)
    active = np.where(label_support > 0)[0]

    row_error_mass = confusion.sum(axis=1)
    selected = active[np.argsort(row_error_mass[active])[-top_n:]]
    selected = selected[np.argsort(label_support[selected])]

    mat = confusion_norm[np.ix_(selected, selected)]
    names = [FUNCTIONAL_GROUP_NAMES[i] for i in selected]

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Row-normalized false-positive co-prediction")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("Predicted false-positive label")
    plt.ylabel("Ground-truth label")
    plt.title("IR functional-group confusion structure")
    plt.tight_layout()
    plt.savefig(out_path / "ir_confusion_heatmap.pdf")
    plt.savefig(out_path / "ir_confusion_heatmap.png", dpi=300)
    plt.close()


def plot_per_label_f1(y_true, y_pred, out_path):
    scores = []

    for i, name in enumerate(FUNCTIONAL_GROUP_NAMES):
        if y_true[:, i].sum() == 0:
            score = np.nan
        else:
            score = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

        scores.append((name, score, y_true[:, i].sum()))

    scores = sorted(scores, key=lambda x: 0 if np.isnan(x[1]) else x[1])

    names = [x[0] for x in scores]
    values = [x[1] for x in scores]

    plt.figure(figsize=(11, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.ylabel("F1")
    plt.title("IR per-label F1 scores")
    plt.tight_layout()
    plt.savefig(out_path / "ir_per_label_f1.pdf")
    plt.savefig(out_path / "ir_per_label_f1.png", dpi=300)
    plt.close()


def build_spectrum_metadata_row(selection_rank, idx, y_true, y_pred, metadata):
    diff = label_difference(y_true[idx], y_pred[idx])
    meta = metadata[idx]
    case = classify_case(y_true[idx], y_pred[idx])

    false_positive_count = len(diff["false_positive_labels"])
    false_negative_count = len(diff["false_negative_labels"])
    total_error_count = false_positive_count + false_negative_count

    row = {
        "selection_rank": int(selection_rank),
        "test_index": int(idx),
        "case_type": case,
        "true_labels": "; ".join(labels_from_vector(y_true[idx])),
        "predicted_labels": "; ".join(labels_from_vector(y_pred[idx])),
        "correct_labels": "; ".join(diff["correct_labels"]),
        "false_positive_labels": "; ".join(diff["false_positive_labels"]),
        "false_negative_labels": "; ".join(diff["false_negative_labels"]),
        "false_positive_count": int(false_positive_count),
        "false_negative_count": int(false_negative_count),
        "total_error_count": int(total_error_count),
        "source_file": meta.get("source_file"),
        "source_row": meta.get("source_row"),
        "global_index": meta.get("global_index"),
    }

    for col in IDENTIFIER_COLUMNS:
        if col in meta:
            row[col] = meta[col]

    return row


def save_selected_spectra_metadata(indices, y_true, y_pred, metadata, out_path):
    rows = []

    for selection_rank, idx in enumerate(indices, start=1):
        row = build_spectrum_metadata_row(
            selection_rank=selection_rank,
            idx=int(idx),
            y_true=y_true,
            y_pred=y_pred,
            metadata=metadata,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_path / "ir_selected_spectra_metadata.csv"
    df.to_csv(csv_path, index=False)

    return df


def plot_selected_spectra(
    ir_spectra,
    y_true,
    y_pred,
    metadata,
    indices,
    out_path,
    save_individual_pngs=False,
):
    x_axis = np.linspace(4000, 400, 600)

    pdf_path = out_path / "ir_selected_spectra.pdf"

    png_dir = out_path / "ir_selected_spectra_png"
    if save_individual_pngs:
        png_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for selection_rank, idx in enumerate(indices, start=1):
            idx = int(idx)

            case = classify_case(y_true[idx], y_pred[idx])
            diff = label_difference(y_true[idx], y_pred[idx])
            meta = metadata[idx]

            true_labels = ", ".join(labels_from_vector(y_true[idx])) or "none"
            pred_labels = ", ".join(labels_from_vector(y_pred[idx])) or "none"
            correct_labels = ", ".join(diff["correct_labels"]) or "none"
            fp_labels = ", ".join(diff["false_positive_labels"]) or "none"
            fn_labels = ", ".join(diff["false_negative_labels"]) or "none"

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(x_axis, ir_spectra[idx])
            ax.invert_xaxis()

            ax.set_xlabel("Wavenumber / cm$^{-1}$")
            ax.set_ylabel("Intensity")

            title = (
                f"{selection_rank}/{len(indices)} | {case.upper()} | test index: {idx}\n"
                f"true: {true_labels}\n"
                f"predicted: {pred_labels}\n"
                f"correct labels: {correct_labels}\n"
                f"false positives: {fp_labels}\n"
                f"false negatives: {fn_labels}"
            )

            wrapped_title = "\n".join(
                textwrap.wrap(title, width=140, replace_whitespace=False)
            )

            ax.set_title(wrapped_title, fontsize=8)

            identifier = get_identifier_string(meta)
            wrapped_identifier = "\n".join(
                textwrap.wrap(identifier, width=170, replace_whitespace=False)
            )

            fig.text(
                0.01,
                0.01,
                wrapped_identifier,
                fontsize=6,
                va="bottom",
                ha="left",
            )

            fig.tight_layout(rect=[0, 0.08, 1, 1])

            pdf.savefig(fig)

            if save_individual_pngs:
                png_path = png_dir / f"ir_spectrum_rank_{selection_rank:03d}_test_index_{idx}.png"
                fig.savefig(png_path, dpi=300)

            plt.close(fig)

    return pdf_path


def print_selection_summary(metadata_df):
    print()
    print("=" * 80)
    print("Selected spectra summary")
    print("=" * 80)

    print(f"Selected spectra: {len(metadata_df)}")

    print()
    print("Case counts:")
    print(metadata_df["case_type"].value_counts().to_string())

    print()
    print("Total label-error statistics:")
    print(metadata_df["total_error_count"].describe().to_string())

    print("=" * 80)


@click.command()
@click.option(
    "--results_path",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default="./benchmark/cnn/models/ir/original/results.pickle",
)
@click.option(
    "--out_path",
    type=click.Path(path_type=Path),
    required=False,
    default="./benchmark/cnn/models/ir/original",
)
@click.option(
    "--analytical_data",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default="data/raw",
)
@click.option("--seed", type=int, default=3245)
@click.option("--test_size", type=float, default=0.1)
@click.option(
    "--preferred_case",
    type=click.Choice(
        ["mixed", "false_positive", "false_negative", "correct"],
        case_sensitive=False,
    ),
    default="mixed",
)
@click.option(
    "--n_spectra",
    type=int,
    default=1000,
    show_default=True,
    help="Number of spectra to analyze.",
)
@click.option(
    "--save_individual_pngs",
    is_flag=True,
    help="Also save one PNG file per selected spectrum.",
)
def main(
    results_path,
    out_path,
    analytical_data,
    seed,
    test_size,
    preferred_case,
    n_spectra,
    save_individual_pngs,
):
    out_path.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = load_results(results_path)

    print("Loaded results")
    print(f"Samples: {len(y_true)}")
    print(f"Micro-F1: {f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    plot_label_confusion(y_true, y_pred, out_path)
    plot_per_label_f1(y_true, y_pred, out_path)

    ir_spectra, metadata = load_ir_test_spectra_with_metadata(
        analytical_data=analytical_data,
        seed=seed,
        test_size=test_size,
    )

    if len(ir_spectra) != len(y_true):
        raise ValueError(
            f"Mismatch: loaded {len(ir_spectra)} spectra but results contain {len(y_true)} labels. "
            "Check seed, test_size, parquet ordering, or whether you used k-fold/original mode."
        )

    selected_indices = select_spectrum_indices(
        y_true=y_true,
        y_pred=y_pred,
        preferred_case=preferred_case,
        n_spectra=n_spectra,
    )

    metadata_df = save_selected_spectra_metadata(
        indices=selected_indices,
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        out_path=out_path,
    )

    selected_pdf_path = plot_selected_spectra(
        ir_spectra=ir_spectra,
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        indices=selected_indices,
        out_path=out_path,
        save_individual_pngs=save_individual_pngs,
    )

    print_selection_summary(metadata_df)

    print()
    print(f"Saved figures to: {out_path}")
    print(f"Saved confusion heatmap: {out_path / 'ir_confusion_heatmap.png'}")
    print(f"Saved per-label F1 plot: {out_path / 'ir_per_label_f1.png'}")
    print(f"Saved selected spectra PDF: {selected_pdf_path}")
    print(f"Saved selected spectra metadata: {out_path / 'ir_selected_spectra_metadata.csv'}")

    if save_individual_pngs:
        print(f"Saved individual spectrum PNGs to: {out_path / 'ir_selected_spectra_png'}")


if __name__ == "__main__":
    main()