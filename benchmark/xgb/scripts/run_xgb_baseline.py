import pickle
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


XGB_N_ESTIMATORS = 300


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


def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) is Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)

    return 0 if n == 0 else 1


def get_functional_groups(smiles: str) -> list | None:
    RDLogger.DisableLog("rdApp.*")

    if smiles is None:
        return None

    smiles = str(smiles).strip().replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    func_groups = []

    for _, smarts in functional_groups.items():
        func_groups.append(match_group(mol, smarts))

    return func_groups


def make_msms_spectrum(spectrum):
    if spectrum is None:
        return None

    msms_spectrum = np.zeros(10000, dtype=np.float32)

    for peak in spectrum:
        if peak is None or len(peak) < 2:
            continue

        peak_pos = int(peak[0] * 10)

        if peak_pos >= 10000:
            peak_pos = 9999

        if peak_pos < 0:
            peak_pos = 0

        msms_spectrum[peak_pos] = peak[1]

    return msms_spectrum


def interpolate_to_600(spec):
    """
    Interpolates any 1D spectrum/vector to exactly 600 values.
    This mirrors the CNN input length.
    """
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

    interp = interp1d(old_x, spec, kind="linear")
    new_spec = interp(new_x)

    return new_spec.astype(np.float32)


def build_xgboost_model(seed: int, device: str, n_jobs: int):
    """
    MultiOutputClassifier trains one binary XGBoost classifier per functional group.
    This makes the model explicitly multi-label, matching the CNN sigmoid output setup.
    """
    base_classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=seed,
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=n_jobs,
        verbosity=1,
    )

    return MultiOutputClassifier(base_classifier, n_jobs=1)


def train_xgboost_model(
    X_train,
    y_train,
    seed: int,
    device: str,
    n_jobs: int,
):
    model = build_xgboost_model(seed=seed, device=device, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model


def append_training_log(logs: list[dict], **kwargs):
    """
    Appends one CSV-friendly training log row.
    Large arrays such as predictions and targets stay in the pickle files.
    """
    logs.append(
        {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            **kwargs,
        }
    )


def save_training_logs(
    logs: list[dict],
    out_path: Path,
    filename: str = "training_logs.csv",
) -> Path:
    log_path = out_path / filename
    pd.DataFrame(logs).to_csv(log_path, index=False)
    return log_path


def compute_f1_metrics(y_true, y_pred, prefix: str) -> dict:
    """
    Computes micro and macro F1 for multi-label predictions.
    """
    return {
        f"{prefix}_f1_micro": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        f"{prefix}_f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def predict_multioutput_at_step(model: MultiOutputClassifier, X, step: int) -> np.ndarray:
    """
    Predicts with a fitted MultiOutputClassifier using only the first `step`
    XGBoost boosting rounds.

    MultiOutputClassifier does not forward `iteration_range`, so each internal
    XGBClassifier is called manually.
    """
    predictions = []

    for estimator in model.estimators_:
        try:
            pred = estimator.predict(X, iteration_range=(0, step))
        except TypeError:
            # Compatibility fallback for older XGBoost versions.
            pred = estimator.predict(X, ntree_limit=step)

        predictions.append(pred.astype(int))

    return np.column_stack(predictions).astype(int)


def append_learning_curve_logs(
    logs: list[dict],
    model: MultiOutputClassifier,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    column: str,
    actual_column: str,
    output_dir: str,
    mode: str,
    fold,
    train_size: int,
    val_size: int,
    test_size: int,
    total_size: int,
    seed: int,
    n_folds,
    device: str,
    n_jobs: int,
    num_functional_groups: int,
    learning_curve_every: int = 1,
):
    """
    Stores one CSV row per XGBoost boosting step.

    Each row contains:
    - train_f1_micro
    - train_f1_macro
    - val_f1_micro
    - val_f1_macro

    Filter `event == "learning_curve"` when plotting.
    """
    n_estimators = int(model.estimators_[0].get_params()["n_estimators"])

    steps = list(range(learning_curve_every, n_estimators + 1, learning_curve_every))

    if not steps or steps[-1] != n_estimators:
        steps.append(n_estimators)

    for step in steps:
        train_pred = predict_multioutput_at_step(model, X_train, step)
        val_pred = predict_multioutput_at_step(model, X_val, step)

        train_metrics = compute_f1_metrics(y_train, train_pred, prefix="train")
        val_metrics = compute_f1_metrics(y_val, val_pred, prefix="val")

        append_training_log(
            logs,
            column=column,
            actual_column=actual_column,
            output_dir=output_dir,
            mode=mode,
            event="learning_curve",
            fold=fold,
            step=step,
            n_estimators=n_estimators,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            total_size=total_size,
            train_f1=train_metrics["train_f1_micro"],
            val_f1=val_metrics["val_f1_micro"],
            test_f1=None,
            train_f1_micro=train_metrics["train_f1_micro"],
            train_f1_macro=train_metrics["train_f1_macro"],
            val_f1_micro=val_metrics["val_f1_micro"],
            val_f1_macro=val_metrics["val_f1_macro"],
            test_f1_micro=None,
            test_f1_macro=None,
            mean_cv_f1=None,
            std_cv_f1=None,
            best_fold=None,
            load_seconds=None,
            fit_seconds=None,
            test_seconds=None,
            seed=seed,
            n_folds=n_folds,
            device=device,
            n_jobs=n_jobs,
            model_type="xgboost_multioutput",
            input_length=600,
            num_functional_groups=num_functional_groups,
        )


def load_data_for_column(analytical_data: Path, actual_col: str) -> pd.DataFrame:
    columns_to_load = ["smiles", actual_col]

    training_data = None
    parquet_files = list(analytical_data.glob("*.parquet"))

    print(f"Found {len(parquet_files)} parquet files")

    for i, parquet_file in enumerate(parquet_files):
        print(
            f"Loading file {i + 1}/{len(parquet_files)}: {parquet_file.name}...",
            end=" ",
            flush=True,
        )

        data = pd.read_parquet(parquet_file, columns=columns_to_load)
        print(f"[{len(data)} samples]")

        if actual_col in ["msms_positive_40ev", "msms_negative_40ev"]:
            data[actual_col] = [make_msms_spectrum(s) for s in data[actual_col]]

        data["func_group"] = [get_functional_groups(s) for s in data["smiles"]]
        data[actual_col] = [interpolate_to_600(s) for s in data[actual_col]]

        data = data.dropna(subset=[actual_col, "func_group"])

        if training_data is None:
            training_data = data
        else:
            training_data = pd.concat((training_data, data), ignore_index=True)

        del data

    if training_data is None:
        raise ValueError(f"No training data could be loaded for column: {actual_col}")

    return training_data


@click.command()
@click.option("--analytical_data", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--base_out_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--columns",
    type=str,
    required=False,
    help="Comma-separated list of columns to process",
)
@click.option("--seed", type=int, default=42)
@click.option("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
@click.option(
    "--use_kfold/--no_kfold",
    default=True,
    help="Use K-Fold cross-validation, matching the CNN script",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="XGBoost device, e.g. 'cuda' or 'cpu'",
)
@click.option(
    "--n_jobs",
    type=int,
    default=-1,
    help="Number of CPU threads for each XGBoost classifier",
)
@click.option(
    "--log_learning_curve/--no_log_learning_curve",
    default=True,
    help="Store train/validation F1 micro and macro for XGBoost boosting steps",
)
@click.option(
    "--learning_curve_every",
    type=int,
    default=20,
    help="Log every N boosting steps. Use 1 to store every step.",
)
def main(
    analytical_data: Path,
    base_out_path: Path,
    columns: str,
    seed: int,
    n_folds: int,
    use_kfold: bool,
    device: str,
    n_jobs: int,
    log_learning_curve: bool,
    learning_curve_every: int,
):
    if learning_curve_every <= 0:
        raise ValueError("--learning_curve_every must be greater than 0")

    columns_to_process = (
        [col.strip() for col in columns.split(",") if col.strip()]
        if columns
        else ["h_nmr_spectra", "c_nmr_spectra", "ir_spectra", "pos_msms", "neg_msms"]
    )

    column_mapping = {
        "h_nmr_spectra": ("h_nmr_spectra", "hnmr"),
        "c_nmr_spectra": ("c_nmr_spectra", "cnmr"),
        "ir_spectra": ("ir_spectra", "ir"),
        "pos_msms": ("msms_positive_40ev", "pos_msms"),
        "neg_msms": ("msms_negative_40ev", "neg_msms"),
    }

    all_training_logs = []

    for col_name in columns_to_process:
        if col_name not in column_mapping:
            raise ValueError(
                f"Unknown column '{col_name}'. "
                f"Valid columns are: {list(column_mapping.keys())}"
            )

        actual_col, output_dir = column_mapping[col_name]
        column_training_logs = []

        print(f"\n{'=' * 60}")
        print(f"Loading data for: {col_name}")
        print(f"{'=' * 60}")

        load_start = time.perf_counter()
        training_data = load_data_for_column(analytical_data, actual_col)
        load_seconds = time.perf_counter() - load_start

        print(f"Total samples loaded: {len(training_data)}")
        print(f"Data loading time: {load_seconds:.2f}s")

        print(f"\n{'=' * 60}")
        print(f"Training XGBoost model for: {col_name} column: {actual_col}")
        print(f"{'=' * 60}")

        X_data = np.stack(training_data[actual_col].to_list()).astype(np.float32)
        y_data = np.stack(training_data["func_group"].to_list()).astype(np.int32)

        num_fgs = y_data.shape[1]

        print(f"Input shape: {X_data.shape}")
        print(f"Target shape: {y_data.shape}")
        print(f"Number of functional groups: {num_fgs}")

        append_training_log(
            column_training_logs,
            column=col_name,
            actual_column=actual_col,
            output_dir=output_dir,
            mode="k_fold" if use_kfold else "original",
            event="data_loaded",
            fold=None,
            step=None,
            n_estimators=XGB_N_ESTIMATORS,
            train_size=None,
            val_size=None,
            test_size=None,
            total_size=len(training_data),
            train_f1=None,
            val_f1=None,
            test_f1=None,
            train_f1_micro=None,
            train_f1_macro=None,
            val_f1_micro=None,
            val_f1_macro=None,
            test_f1_micro=None,
            test_f1_macro=None,
            mean_cv_f1=None,
            std_cv_f1=None,
            best_fold=None,
            load_seconds=load_seconds,
            fit_seconds=None,
            test_seconds=None,
            seed=seed,
            n_folds=n_folds if use_kfold else None,
            device=device,
            n_jobs=n_jobs,
            model_type="xgboost_multioutput",
            input_length=600,
            num_functional_groups=num_fgs,
        )

        # Same first split as CNN script:
        # 90% train-full, 10% held-out test.
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_data,
            y_data,
            test_size=0.1,
            random_state=seed,
            shuffle=True,
        )

        print(
            f"Initial split: Train-full={len(X_train_full)} 90%, "
            f"Test={len(X_test)} 10%"
        )

        if use_kfold:
            print(f"Performing {n_folds}-fold CV on training set...")

            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

            fold_f1_micro_scores = []
            fold_f1_macro_scores = []
            all_predictions = []
            all_targets = []

            best_model = None
            best_fold_idx = None
            best_fold_f1 = -1.0

            for fold_idx, (train_idx, val_idx) in enumerate(
                kfold.split(X_train_full), start=1
            ):
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")

                X_train = X_train_full[train_idx]
                X_val = X_train_full[val_idx]
                y_train = y_train_full[train_idx]
                y_val = y_train_full[val_idx]

                print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

                fit_start = time.perf_counter()

                model = train_xgboost_model(
                    X_train=X_train,
                    y_train=y_train,
                    seed=seed,
                    device=device,
                    n_jobs=n_jobs,
                )

                fit_seconds = time.perf_counter() - fit_start

                if log_learning_curve:
                    print(
                        f"Logging learning curve every {learning_curve_every} "
                        f"boosting step(s)..."
                    )

                    append_learning_curve_logs(
                        column_training_logs,
                        model,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        column=col_name,
                        actual_column=actual_col,
                        output_dir=output_dir,
                        mode="k_fold",
                        fold=fold_idx,
                        train_size=len(X_train),
                        val_size=len(X_val),
                        test_size=len(X_test),
                        total_size=len(training_data),
                        seed=seed,
                        n_folds=n_folds,
                        device=device,
                        n_jobs=n_jobs,
                        num_functional_groups=num_fgs,
                        learning_curve_every=learning_curve_every,
                    )

                train_prediction = model.predict(X_train).astype(int)
                val_prediction = model.predict(X_val).astype(int)

                train_metrics = compute_f1_metrics(
                    y_train,
                    train_prediction,
                    prefix="train",
                )
                val_metrics = compute_f1_metrics(
                    y_val,
                    val_prediction,
                    prefix="val",
                )

                train_f1_micro = train_metrics["train_f1_micro"]
                val_f1_micro = val_metrics["val_f1_micro"]
                val_f1_macro = val_metrics["val_f1_macro"]

                fold_f1_micro_scores.append(val_f1_micro)
                fold_f1_macro_scores.append(val_f1_macro)
                all_predictions.append(val_prediction)
                all_targets.append(y_val)

                print(f"Fold {fold_idx} Train F1 Micro: {train_metrics['train_f1_micro']:.4f}")
                print(f"Fold {fold_idx} Train F1 Macro: {train_metrics['train_f1_macro']:.4f}")
                print(f"Fold {fold_idx} Validation F1 Micro: {val_metrics['val_f1_micro']:.4f}")
                print(f"Fold {fold_idx} Validation F1 Macro: {val_metrics['val_f1_macro']:.4f}")
                print(f"Fold {fold_idx} Fit time: {fit_seconds:.2f}s")

                append_training_log(
                    column_training_logs,
                    column=col_name,
                    actual_column=actual_col,
                    output_dir=output_dir,
                    mode="k_fold",
                    event="fold_finished",
                    fold=fold_idx,
                    step=XGB_N_ESTIMATORS,
                    n_estimators=XGB_N_ESTIMATORS,
                    train_size=len(X_train),
                    val_size=len(X_val),
                    test_size=len(X_test),
                    total_size=len(training_data),
                    train_f1=train_metrics["train_f1_micro"],
                    val_f1=val_metrics["val_f1_micro"],
                    test_f1=None,
                    train_f1_micro=train_metrics["train_f1_micro"],
                    train_f1_macro=train_metrics["train_f1_macro"],
                    val_f1_micro=val_metrics["val_f1_micro"],
                    val_f1_macro=val_metrics["val_f1_macro"],
                    test_f1_micro=None,
                    test_f1_macro=None,
                    mean_cv_f1=None,
                    std_cv_f1=None,
                    best_fold=None,
                    load_seconds=None,
                    fit_seconds=fit_seconds,
                    test_seconds=None,
                    seed=seed,
                    n_folds=n_folds,
                    device=device,
                    n_jobs=n_jobs,
                    model_type="xgboost_multioutput",
                    input_length=600,
                    num_functional_groups=num_fgs,
                )

                if val_f1_micro > best_fold_f1:
                    best_fold_f1 = val_f1_micro
                    best_fold_idx = fold_idx - 1
                    best_model = model

            mean_f1_micro = float(np.mean(fold_f1_micro_scores))
            std_f1_micro = float(np.std(fold_f1_micro_scores))

            mean_f1_macro = float(np.mean(fold_f1_macro_scores))
            std_f1_macro = float(np.std(fold_f1_macro_scores))

            print(f"\n{'=' * 60}")
            print(f"Cross-Validation Results for {col_name}:")
            print(f"Mean CV F1 Micro: {mean_f1_micro:.4f} ± {std_f1_micro:.4f}")
            print(f"Mean CV F1 Macro: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")
            print(
                "Individual Fold Micro Scores: "
                f"{[f'{score:.4f}' for score in fold_f1_micro_scores]}"
            )
            print(
                "Individual Fold Macro Scores: "
                f"{[f'{score:.4f}' for score in fold_f1_macro_scores]}"
            )
            print(f"{'=' * 60}")

            print(
                f"\nBest model: Fold {best_fold_idx + 1} "
                f"CV F1 Micro: {fold_f1_micro_scores[best_fold_idx]:.4f}"
            )

            print(f"\nEvaluating on test set {len(X_test)} samples...")

            test_start = time.perf_counter()
            test_predictions = best_model.predict(X_test).astype(int)
            test_seconds = time.perf_counter() - test_start

            test_metrics = compute_f1_metrics(
                y_test,
                test_predictions,
                prefix="test",
            )
            test_f1 = test_metrics["test_f1_micro"]

            print(f"\n{'=' * 60}")
            print(f"FINAL TEST RESULTS for {col_name}:")
            print(f"Test F1 Micro: {test_metrics['test_f1_micro']:.4f}")
            print(f"Test F1 Macro: {test_metrics['test_f1_macro']:.4f}")
            print(f"Test prediction time: {test_seconds:.2f}s")
            print(f"{'=' * 60}")

            append_training_log(
                column_training_logs,
                column=col_name,
                actual_column=actual_col,
                output_dir=output_dir,
                mode="k_fold",
                event="final_test",
                fold=best_fold_idx + 1,
                step=XGB_N_ESTIMATORS,
                n_estimators=XGB_N_ESTIMATORS,
                train_size=len(X_train_full),
                val_size=None,
                test_size=len(X_test),
                total_size=len(training_data),
                train_f1=None,
                val_f1=fold_f1_micro_scores[best_fold_idx],
                test_f1=test_metrics["test_f1_micro"],
                train_f1_micro=None,
                train_f1_macro=None,
                val_f1_micro=fold_f1_micro_scores[best_fold_idx],
                val_f1_macro=fold_f1_macro_scores[best_fold_idx],
                test_f1_micro=test_metrics["test_f1_micro"],
                test_f1_macro=test_metrics["test_f1_macro"],
                mean_cv_f1=mean_f1_micro,
                std_cv_f1=std_f1_micro,
                mean_cv_f1_micro=mean_f1_micro,
                std_cv_f1_micro=std_f1_micro,
                mean_cv_f1_macro=mean_f1_macro,
                std_cv_f1_macro=std_f1_macro,
                best_fold=best_fold_idx + 1,
                load_seconds=None,
                fit_seconds=None,
                test_seconds=test_seconds,
                seed=seed,
                n_folds=n_folds,
                device=device,
                n_jobs=n_jobs,
                model_type="xgboost_multioutput",
                input_length=600,
                num_functional_groups=num_fgs,
            )

            out_path = base_out_path / output_dir / "k_fold"
            out_path.mkdir(parents=True, exist_ok=True)

            training_log_path = save_training_logs(column_training_logs, out_path)
            all_training_logs.extend(column_training_logs)

            cv_results = {
                "fold_f1_micro_scores": fold_f1_micro_scores,
                "fold_f1_macro_scores": fold_f1_macro_scores,
                "mean_cv_f1_micro": mean_f1_micro,
                "std_cv_f1_micro": std_f1_micro,
                "mean_cv_f1_macro": mean_f1_macro,
                "std_cv_f1_macro": std_f1_macro,
                "best_fold_idx": best_fold_idx,
                "test_f1_micro": test_metrics["test_f1_micro"],
                "test_f1_macro": test_metrics["test_f1_macro"],
                "test_predictions": test_predictions,
                "test_targets": y_test,
                "all_cv_predictions": all_predictions,
                "all_cv_targets": all_targets,
                "n_folds": n_folds,
                "train_size": len(X_train_full),
                "test_size": len(X_test),
                "seed": seed,
                "model_type": "xgboost_multioutput",
                "column": col_name,
                "actual_column": actual_col,
                "input_length": 600,
                "num_functional_groups": num_fgs,
                "n_estimators": XGB_N_ESTIMATORS,
                "learning_curve_logged": log_learning_curve,
                "learning_curve_every": learning_curve_every,
            }

            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(cv_results, file)

            with open(out_path / f"{output_dir}_xgboost_model.pickle", "wb") as file:
                pickle.dump(best_model, file)

            print(f"\nResults saved to: {out_path / 'results.pickle'}")
            print(
                "Best XGBoost model saved to: "
                f"{out_path / f'{output_dir}_xgboost_model.pickle'}"
            )
            print(f"Training logs saved to: {training_log_path}")

        else:
            print("Training without K-Fold, matching CNN original mode...")

            # Same as CNN no-kfold mode:
            # X_train_full is 90% of total.
            # Validation is 1/9 of train-full = 10% of total.
            # Final split = 80% train, 10% validation, 10% test.
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=1 / 9,
                random_state=seed,
                shuffle=True,
            )

            print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

            fit_start = time.perf_counter()

            model = train_xgboost_model(
                X_train=X_train,
                y_train=y_train,
                seed=seed,
                device=device,
                n_jobs=n_jobs,
            )

            fit_seconds = time.perf_counter() - fit_start

            if log_learning_curve:
                print(
                    f"Logging learning curve every {learning_curve_every} "
                    f"boosting step(s)..."
                )

                append_learning_curve_logs(
                    column_training_logs,
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    column=col_name,
                    actual_column=actual_col,
                    output_dir=output_dir,
                    mode="original",
                    fold=None,
                    train_size=len(X_train),
                    val_size=len(X_val),
                    test_size=len(X_test),
                    total_size=len(training_data),
                    seed=seed,
                    n_folds=None,
                    device=device,
                    n_jobs=n_jobs,
                    num_functional_groups=num_fgs,
                    learning_curve_every=learning_curve_every,
                )

            train_predictions = model.predict(X_train).astype(int)
            val_predictions = model.predict(X_val).astype(int)

            test_start = time.perf_counter()
            test_predictions = model.predict(X_test).astype(int)
            test_seconds = time.perf_counter() - test_start

            train_metrics = compute_f1_metrics(
                y_train,
                train_predictions,
                prefix="train",
            )
            val_metrics = compute_f1_metrics(
                y_val,
                val_predictions,
                prefix="val",
            )
            test_metrics = compute_f1_metrics(
                y_test,
                test_predictions,
                prefix="test",
            )

            train_f1 = train_metrics["train_f1_micro"]
            val_f1 = val_metrics["val_f1_micro"]
            test_f1 = test_metrics["test_f1_micro"]

            append_training_log(
                column_training_logs,
                column=col_name,
                actual_column=actual_col,
                output_dir=output_dir,
                mode="original",
                event="training_finished",
                fold=None,
                step=XGB_N_ESTIMATORS,
                n_estimators=XGB_N_ESTIMATORS,
                train_size=len(X_train),
                val_size=len(X_val),
                test_size=len(X_test),
                total_size=len(training_data),
                train_f1=train_f1,
                val_f1=val_f1,
                test_f1=test_f1,
                train_f1_micro=train_metrics["train_f1_micro"],
                train_f1_macro=train_metrics["train_f1_macro"],
                val_f1_micro=val_metrics["val_f1_micro"],
                val_f1_macro=val_metrics["val_f1_macro"],
                test_f1_micro=test_metrics["test_f1_micro"],
                test_f1_macro=test_metrics["test_f1_macro"],
                mean_cv_f1=None,
                std_cv_f1=None,
                best_fold=None,
                load_seconds=None,
                fit_seconds=fit_seconds,
                test_seconds=test_seconds,
                seed=seed,
                n_folds=None,
                device=device,
                n_jobs=n_jobs,
                model_type="xgboost_multioutput",
                input_length=600,
                num_functional_groups=num_fgs,
            )

            print(f"\n{'=' * 60}")
            print(f"FINAL RESULTS for {col_name}:")
            print(f"Train F1 Micro: {train_metrics['train_f1_micro']:.4f}")
            print(f"Train F1 Macro: {train_metrics['train_f1_macro']:.4f}")
            print(f"Validation F1 Micro: {val_metrics['val_f1_micro']:.4f}")
            print(f"Validation F1 Macro: {val_metrics['val_f1_macro']:.4f}")
            print(f"Test F1 Micro: {test_metrics['test_f1_micro']:.4f}")
            print(f"Test F1 Macro: {test_metrics['test_f1_macro']:.4f}")
            print(f"Fit time: {fit_seconds:.2f}s")
            print(f"Test prediction time: {test_seconds:.2f}s")
            print(f"{'=' * 60}")

            out_path = base_out_path / output_dir / "original"
            out_path.mkdir(parents=True, exist_ok=True)

            training_log_path = save_training_logs(column_training_logs, out_path)
            all_training_logs.extend(column_training_logs)

            results = {
                "train_f1_micro": train_metrics["train_f1_micro"],
                "train_f1_macro": train_metrics["train_f1_macro"],
                "val_f1_micro": val_metrics["val_f1_micro"],
                "val_f1_macro": val_metrics["val_f1_macro"],
                "test_f1_micro": test_metrics["test_f1_micro"],
                "test_f1_macro": test_metrics["test_f1_macro"],
                "train_pred": train_predictions,
                "train_tgt": y_train,
                "val_pred": val_predictions,
                "val_tgt": y_val,
                "pred": test_predictions,
                "tgt": y_test,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "seed": seed,
                "model_type": "xgboost_multioutput",
                "column": col_name,
                "actual_column": actual_col,
                "input_length": 600,
                "num_functional_groups": num_fgs,
                "n_estimators": XGB_N_ESTIMATORS,
                "learning_curve_logged": log_learning_curve,
                "learning_curve_every": learning_curve_every,
            }

            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(results, file)

            with open(out_path / f"{output_dir}_xgboost_model.pickle", "wb") as file:
                pickle.dump(model, file)

            print(f"\nResults saved to: {out_path / 'results.pickle'}")
            print(f"XGBoost model saved to: {out_path / f'{output_dir}_xgboost_model.pickle'}")
            print(f"Training logs saved to: {training_log_path}")

        del training_data, X_data, y_data

    if all_training_logs:
        all_logs_path = base_out_path / "training_logs_all_columns.csv"
        pd.DataFrame(all_training_logs).to_csv(all_logs_path, index=False)
        print(f"\nCombined training logs saved to: {all_logs_path}")


if __name__ == "__main__":
    main()