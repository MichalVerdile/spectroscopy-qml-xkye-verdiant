# Adapted from Guwon Jung: https://github.com/gj475/irchracterizationcnn

import os
import pickle
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from keras import backend as K
from keras import ops
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from keras.models import Model
from keras.optimizers import Adam
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.config.optimizer.set_jit(False)

# GPU Configuration
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {len(gpus)} device(s)")
        print(f"  {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected - running on CPU")


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


class F1ScoreCallback(Callback):
    """
    Computes multilabel F1 scores at the end of every epoch and stores them
    inside Keras' logs/history so they are automatically written to CSV later.
    """

    def __init__(self, X_train, y_train, X_val=None, y_val=None):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        train_pred = (self.model.predict(self.X_train, verbose=0) > 0.5).astype(int)

        logs["train_f1_micro"] = f1_score(
            self.y_train,
            train_pred,
            average="micro",
            zero_division=0,
        )
        logs["train_f1_macro"] = f1_score(
            self.y_train,
            train_pred,
            average="macro",
            zero_division=0,
        )

        if self.X_val is not None and self.y_val is not None:
            val_pred = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)

            logs["val_f1_micro"] = f1_score(
                self.y_val,
                val_pred,
                average="micro",
                zero_division=0,
            )
            logs["val_f1_macro"] = f1_score(
                self.y_val,
                val_pred,
                average="macro",
                zero_division=0,
            )


class LearningRateLogger(Callback):
    """
    Stores the current optimizer learning rate in Keras' logs/history so it is
    exported to the CSV training logs together with the other epoch metrics.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        learning_rate = self.model.optimizer.learning_rate

        # If learning_rate is a schedule, evaluate it at the current optimizer step.
        if callable(learning_rate):
            learning_rate = learning_rate(self.model.optimizer.iterations)

        learning_rate = float(ops.convert_to_numpy(learning_rate))
        logs["learning_rate"] = learning_rate


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


def append_training_log(logs: list[dict], **kwargs):
    """
    Appends one CSV-friendly training log row.
    Predictions and target arrays are intentionally stored only in pickle files.
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


def append_keras_history_logs(
    logs: list[dict],
    history,
    column: str,
    actual_column: str,
    output_dir: str,
    mode: str,
    fold: int | None,
    train_size: int,
    val_size: int | None,
    test_size: int,
    total_size: int,
    seed: int,
    n_folds: int | None,
    num_functional_groups: int,
):
    """
    Converts Keras History.history into CSV-friendly epoch rows.
    Includes loss, val_loss, learning_rate, train_f1_micro, train_f1_macro,
    val_f1_micro, and val_f1_macro when available.
    """
    if history is None:
        return

    history_dict = history.history

    if not history_dict:
        return

    num_epochs = max(len(values) for values in history_dict.values())

    for epoch_idx in range(num_epochs):
        row = {
            "column": column,
            "actual_column": actual_column,
            "output_dir": output_dir,
            "mode": mode,
            "event": "epoch_finished",
            "fold": fold,
            "epoch": epoch_idx + 1,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "total_size": total_size,
            "train_f1": None,
            "val_f1": None,
            "test_f1": None,
            "train_f1_micro": None,
            "train_f1_macro": None,
            "val_f1_micro": None,
            "val_f1_macro": None,
            "test_f1_micro": None,
            "test_f1_macro": None,
            "mean_cv_f1": None,
            "std_cv_f1": None,
            "best_fold": None,
            "load_seconds": None,
            "fit_seconds": None,
            "predict_seconds": None,
            "seed": seed,
            "n_folds": n_folds,
            "model_type": "cnn",
            "input_length": 600,
            "num_functional_groups": num_functional_groups,
        }

        for metric_name, values in history_dict.items():
            if epoch_idx < len(values):
                value = values[epoch_idx]

                try:
                    value = float(value)
                except TypeError:
                    pass

                row[metric_name] = value

        append_training_log(logs, **row)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    num_fgs,
    aug,
    num,
    weighted,
):
    """Trains final model with the best hyper-parameters."""

    # Input
    X_train = X_train.reshape(X_train.shape[0], 600, 1)

    if X_val is not None:
        X_val = X_val.reshape(X_val.shape[0], 600, 1)

    # Shape of input data.
    input_shape = X_train.shape[1:]
    input_tensor = Input(shape=input_shape)

    # 1st CNN layer.
    x = Conv1D(filters=31, kernel_size=11, strides=1, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 2nd CNN layer.
    x = Conv1D(filters=62, kernel_size=11, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Flatten layer.
    x = Flatten()(x)

    # 1st dense layer.
    x = Dense(4927, activation="relu")(x)
    x = Dropout(0.48599073736368)(x)

    # 2nd dense layer.
    x = Dense(2785, activation="relu")(x)
    x = Dropout(0.48599073736368)(x)

    # 3rd dense layer.
    x = Dense(1574, activation="relu")(x)
    x = Dropout(0.48599073736368)(x)

    output_tensor = Dense(num_fgs, activation="sigmoid")(x)

    print("Model Construction")
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()

    optimizer = Adam(learning_rate=2.5e-4)

    if weighted == 1:

        def calculate_class_weights(y_true):
            number_dim = np.shape(y_true)[1]
            weights = np.zeros((2, number_dim))

            # Calculates weights for each label in a for loop.
            for i in range(number_dim):
                negative_count = (y_train[:, i] == 0).sum()
                positive_count = (y_train[:, i] == 1).sum()

                # Avoid division by zero for rare labels.
                weights_n = y_train.shape[0] / (2 * negative_count) if negative_count > 0 else 1.0
                weights_p = y_train.shape[0] / (2 * positive_count) if positive_count > 0 else 1.0

                weights[1, i], weights[0, i] = weights_p, weights_n

            return weights.T

        def get_weighted_loss(weights):
            def weighted_loss(y_true, y_pred):
                return K.mean(
                    (weights[:, 0] ** (1.0 - y_true))
                    * (weights[:, 1] ** y_true)
                    * K.binary_crossentropy(y_true, y_pred),
                    axis=-1,
                )

            return weighted_loss

        model.compile(
            optimizer=optimizer,
            loss=get_weighted_loss(calculate_class_weights(y_train)),
        )

    else:
        model.compile(optimizer=optimizer, loss="binary_crossentropy")

    print("Start training")

    X_test = X_test.reshape(X_test.shape[0], 600, 1)

    monitor_metric = "val_loss" if X_val is not None and y_val is not None else "loss"

    lr_scheduler = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.7,
        patience=10,
        mode="min",
        min_lr=1e-8,
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=20,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": 200,
        "batch_size": 1024,
        "verbose": 1,
        "callbacks": [
            F1ScoreCallback(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            ),
            lr_scheduler,
            early_stopping,
            LearningRateLogger(),
        ],
    }

    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)

    fit_start = time.perf_counter()
    history = model.fit(**fit_kwargs)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    prediction = model.predict(X_test)
    predict_seconds = time.perf_counter() - predict_start

    return (prediction > 0.5).astype(int), model, history, fit_seconds, predict_seconds


def interpolate_to_600(spec):
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

    interp = interp1d(old_x, spec)
    new_spec = interp(new_x)

    return new_spec.astype(np.float32)


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
@click.option("--seed", type=int, default=3245)
@click.option("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
@click.option(
    "--use_kfold/--no_kfold",
    default=True,
    help="Use K-Fold cross-validation (default: True)",
)
def main(
    analytical_data: Path,
    base_out_path: Path,
    columns: str,
    seed: int,
    n_folds: int,
    use_kfold: bool,
):
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
        print(f"Training model for: {col_name} column: {actual_col}")
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
            epoch=None,
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
            predict_seconds=None,
            seed=seed,
            n_folds=n_folds if use_kfold else None,
            model_type="cnn",
            input_length=600,
            num_functional_groups=num_fgs,
        )

        # Same first split as the XGBoost script:
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

            fold_f1_scores = []
            fold_f1_macro_scores = []
            all_predictions = []
            all_targets = []
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(
                kfold.split(X_train_full), start=1
            ):
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")

                X_train = X_train_full[train_idx]
                X_val = X_train_full[val_idx]
                y_train = y_train_full[train_idx]
                y_val = y_train_full[val_idx]

                print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

                prediction, model, history, fit_seconds, predict_seconds = train_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_val,
                    num_fgs,
                    "e",
                    0,
                    0,
                )

                fold_f1_micro = f1_score(
                    y_val,
                    prediction,
                    average="micro",
                    zero_division=0,
                )
                fold_f1_macro = f1_score(
                    y_val,
                    prediction,
                    average="macro",
                    zero_division=0,
                )

                fold_f1_scores.append(fold_f1_micro)
                fold_f1_macro_scores.append(fold_f1_macro)

                print(f"Fold {fold_idx} F1 Micro Score: {fold_f1_micro:.4f}")
                print(f"Fold {fold_idx} F1 Macro Score: {fold_f1_macro:.4f}")
                print(f"Fold {fold_idx} Fit time: {fit_seconds:.2f}s")
                print(f"Fold {fold_idx} Validation prediction time: {predict_seconds:.2f}s")

                append_keras_history_logs(
                    column_training_logs,
                    history=history,
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
                    num_functional_groups=num_fgs,
                )

                append_training_log(
                    column_training_logs,
                    column=col_name,
                    actual_column=actual_col,
                    output_dir=output_dir,
                    mode="k_fold",
                    event="fold_finished",
                    fold=fold_idx,
                    epoch=None,
                    train_size=len(X_train),
                    val_size=len(X_val),
                    test_size=len(X_test),
                    total_size=len(training_data),
                    train_f1=None,
                    val_f1=fold_f1_micro,
                    test_f1=None,
                    train_f1_micro=None,
                    train_f1_macro=None,
                    val_f1_micro=fold_f1_micro,
                    val_f1_macro=fold_f1_macro,
                    test_f1_micro=None,
                    test_f1_macro=None,
                    mean_cv_f1=None,
                    std_cv_f1=None,
                    best_fold=None,
                    load_seconds=None,
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                    seed=seed,
                    n_folds=n_folds,
                    model_type="cnn",
                    input_length=600,
                    num_functional_groups=num_fgs,
                )

                all_predictions.append(prediction)
                all_targets.append(y_val)
                fold_models.append(model)

            mean_f1 = float(np.mean(fold_f1_scores))
            std_f1 = float(np.std(fold_f1_scores))
            mean_f1_macro = float(np.mean(fold_f1_macro_scores))
            std_f1_macro = float(np.std(fold_f1_macro_scores))

            print(f"\n{'=' * 60}")
            print(f"Cross-Validation Results for {col_name}:")
            print(f"Mean CV F1 Micro Score: {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"Mean CV F1 Macro Score: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")
            print(f"Individual Fold Micro Scores: {[f'{score:.4f}' for score in fold_f1_scores]}")
            print(f"Individual Fold Macro Scores: {[f'{score:.4f}' for score in fold_f1_macro_scores]}")
            print(f"{'=' * 60}")

            best_fold_idx = int(np.argmax(fold_f1_scores))
            best_model = fold_models[best_fold_idx]

            print(
                f"\nBest model: Fold {best_fold_idx + 1} "
                f"CV F1 Micro: {fold_f1_scores[best_fold_idx]:.4f}"
            )

            print(f"\nEvaluating on test set {len(X_test)} samples...")

            X_test_reshaped = X_test.reshape(X_test.shape[0], 600, 1)

            test_predict_start = time.perf_counter()
            test_predictions = best_model.predict(X_test_reshaped)
            test_predict_seconds = time.perf_counter() - test_predict_start

            test_predictions_binary = (test_predictions > 0.5).astype(int)
            test_f1_micro = f1_score(
                y_test,
                test_predictions_binary,
                average="micro",
                zero_division=0,
            )
            test_f1_macro = f1_score(
                y_test,
                test_predictions_binary,
                average="macro",
                zero_division=0,
            )

            print(f"\n{'=' * 60}")
            print(f"FINAL TEST RESULTS for {col_name}:")
            print(f"Test F1 Micro Score: {test_f1_micro:.4f}")
            print(f"Test F1 Macro Score: {test_f1_macro:.4f}")
            print(f"Test prediction time: {test_predict_seconds:.2f}s")
            print(f"{'=' * 60}")

            append_training_log(
                column_training_logs,
                column=col_name,
                actual_column=actual_col,
                output_dir=output_dir,
                mode="k_fold",
                event="final_test",
                fold=best_fold_idx + 1,
                epoch=None,
                train_size=len(X_train_full),
                val_size=None,
                test_size=len(X_test),
                total_size=len(training_data),
                train_f1=None,
                val_f1=fold_f1_scores[best_fold_idx],
                test_f1=test_f1_micro,
                train_f1_micro=None,
                train_f1_macro=None,
                val_f1_micro=fold_f1_scores[best_fold_idx],
                val_f1_macro=fold_f1_macro_scores[best_fold_idx],
                test_f1_micro=test_f1_micro,
                test_f1_macro=test_f1_macro,
                mean_cv_f1=mean_f1,
                std_cv_f1=std_f1,
                mean_cv_f1_macro=mean_f1_macro,
                std_cv_f1_macro=std_f1_macro,
                best_fold=best_fold_idx + 1,
                load_seconds=None,
                fit_seconds=None,
                predict_seconds=test_predict_seconds,
                seed=seed,
                n_folds=n_folds,
                model_type="cnn",
                input_length=600,
                num_functional_groups=num_fgs,
            )

            out_path = base_out_path / output_dir / "k_fold"
            out_path.mkdir(parents=True, exist_ok=True)

            training_log_path = save_training_logs(column_training_logs, out_path)
            all_training_logs.extend(column_training_logs)

            cv_results = {
                "fold_scores": fold_f1_scores,
                "fold_f1_micro_scores": fold_f1_scores,
                "fold_f1_macro_scores": fold_f1_macro_scores,
                "mean_cv_f1": mean_f1,
                "std_cv_f1": std_f1,
                "mean_cv_f1_micro": mean_f1,
                "std_cv_f1_micro": std_f1,
                "mean_cv_f1_macro": mean_f1_macro,
                "std_cv_f1_macro": std_f1_macro,
                "best_fold_idx": best_fold_idx,
                "test_f1": test_f1_micro,
                "test_f1_micro": test_f1_micro,
                "test_f1_macro": test_f1_macro,
                "test_predictions": test_predictions_binary,
                "test_targets": y_test,
                "all_cv_predictions": all_predictions,
                "all_cv_targets": all_targets,
                "n_folds": n_folds,
                "train_size": len(X_train_full),
                "test_size": len(X_test),
                "seed": seed,
                "model_type": "cnn",
                "column": col_name,
                "actual_column": actual_col,
                "input_length": 600,
                "num_functional_groups": num_fgs,
            }

            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(cv_results, file)

            print(f"\nResults saved to: {out_path / 'results.pickle'}")

            best_model.save(str(out_path / f"{output_dir}_model.keras"))
            print(f"Best model saved to: {out_path / f'{output_dir}_model.keras'}")
            print(f"Training logs saved to: {training_log_path}")

        else:
            print("Training without K-Fold, matching original mode...")

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

            test_predictions_binary, model, history, fit_seconds, predict_seconds = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                num_fgs,
                "e",
                0,
                0,
            )

            val_predict_start = time.perf_counter()
            X_val_reshaped = X_val.reshape(X_val.shape[0], 600, 1)
            val_predictions = model.predict(X_val_reshaped)
            val_predict_seconds = time.perf_counter() - val_predict_start
            val_predictions_binary = (val_predictions > 0.5).astype(int)

            val_f1_micro = f1_score(
                y_val,
                val_predictions_binary,
                average="micro",
                zero_division=0,
            )
            val_f1_macro = f1_score(
                y_val,
                val_predictions_binary,
                average="macro",
                zero_division=0,
            )
            test_f1_micro = f1_score(
                y_test,
                test_predictions_binary,
                average="micro",
                zero_division=0,
            )
            test_f1_macro = f1_score(
                y_test,
                test_predictions_binary,
                average="macro",
                zero_division=0,
            )

            append_keras_history_logs(
                column_training_logs,
                history=history,
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
                num_functional_groups=num_fgs,
            )

            append_training_log(
                column_training_logs,
                column=col_name,
                actual_column=actual_col,
                output_dir=output_dir,
                mode="original",
                event="training_finished",
                fold=None,
                epoch=None,
                train_size=len(X_train),
                val_size=len(X_val),
                test_size=len(X_test),
                total_size=len(training_data),
                train_f1=None,
                val_f1=val_f1_micro,
                test_f1=test_f1_micro,
                train_f1_micro=None,
                train_f1_macro=None,
                val_f1_micro=val_f1_micro,
                val_f1_macro=val_f1_macro,
                test_f1_micro=test_f1_micro,
                test_f1_macro=test_f1_macro,
                mean_cv_f1=None,
                std_cv_f1=None,
                best_fold=None,
                load_seconds=None,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                val_predict_seconds=val_predict_seconds,
                seed=seed,
                n_folds=None,
                model_type="cnn",
                input_length=600,
                num_functional_groups=num_fgs,
            )

            print(f"\n{'=' * 60}")
            print(f"FINAL TEST RESULTS for {col_name}:")
            print(f"Validation F1 Micro Score: {val_f1_micro:.4f}")
            print(f"Validation F1 Macro Score: {val_f1_macro:.4f}")
            print(f"Test F1 Micro Score: {test_f1_micro:.4f}")
            print(f"Test F1 Macro Score: {test_f1_macro:.4f}")
            print(f"Fit time: {fit_seconds:.2f}s")
            print(f"Test prediction time: {predict_seconds:.2f}s")
            print(f"{'=' * 60}")

            out_path = base_out_path / output_dir / "original"
            out_path.mkdir(parents=True, exist_ok=True)

            training_log_path = save_training_logs(column_training_logs, out_path)
            all_training_logs.extend(column_training_logs)

            results = {
                "val_f1": val_f1_micro,
                "val_f1_micro": val_f1_micro,
                "val_f1_macro": val_f1_macro,
                "test_f1": test_f1_micro,
                "test_f1_micro": test_f1_micro,
                "test_f1_macro": test_f1_macro,
                "val_pred": val_predictions_binary,
                "val_tgt": y_val,
                "pred": test_predictions_binary,
                "tgt": y_test,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "seed": seed,
                "model_type": "cnn",
                "column": col_name,
                "actual_column": actual_col,
                "input_length": 600,
                "num_functional_groups": num_fgs,
            }

            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(results, file)

            print(f"\nResults saved to: {out_path / 'results.pickle'}")

            model.save(str(out_path / f"{output_dir}_model.keras"))
            print(f"Model saved to: {out_path / f'{output_dir}_model.keras'}")
            print(f"Training logs saved to: {training_log_path}")

        del training_data, X_data, y_data

    if all_training_logs:
        all_logs_path = base_out_path / "training_logs_all_columns.csv"
        pd.DataFrame(all_training_logs).to_csv(all_logs_path, index=False)
        print(f"\nCombined training logs saved to: {all_logs_path}")


if __name__ == "__main__":
    main()
