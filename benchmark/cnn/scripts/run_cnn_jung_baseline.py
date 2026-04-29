# Adapted from Guwon Jung: https://github.com/gj475/irchracterizationcnn

import os
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from keras import backend as K
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
    "Imine": Chem.MolFromSmarts("[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"),
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


def get_functional_groups(smiles: str) -> dict:
    RDLogger.DisableLog("rdApp.*")
    smiles = smiles.strip().replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    func_groups = list()
    for func_group_name, smarts in functional_groups.items():
        func_groups.append(match_group(mol, smarts))

    return func_groups


def train_model(X_train, y_train, X_val, y_val, X_test, num_fgs, aug, num, weighted):
    """Trains final model with the best hyper-parameters."""
    # Input
    X_train = X_train.reshape(X_train.shape[0], 600, 1)
    if X_val is not None:
        X_val = X_val.reshape(X_val.shape[0], 600, 1)

    # Shape of input data.
    input_shape = X_train.shape[1:]
    input_tensor = Input(shape=input_shape)

    # 1st CNN layer.
    x = Conv1D(filters=31, kernel_size=(11), strides=1, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 2nd CNN layer.
    x = Conv1D(filters=62, kernel_size=(11), strides=1, padding="same")(x)
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
    optimizer = Adam()

    if weighted == 1:

        def calculate_class_weights(y_true):
            number_dim = np.shape(y_true)[1]
            weights = np.zeros((2, number_dim))
            # Calculates weights for each label in a for loop.
            for i in range(number_dim):
                weights_n, weights_p = (
                    (y_train.shape[0] / (2 * (y_train[:, i] == 0).sum())),
                    (y_train.shape[0] / (2 * (y_train[:, i] == 1).sum())),
                )
                # Weights could be log-dampened to avoid extreme weights for extremly unbalanced data.
                weights[1, i], weights[0, i] = weights_p, weights_n

            return weights.T

        def get_weighted_loss(weights):
            def weighted_loss(y_true, y_pred):
                return K.mean(
                    (weights[:, 0] ** (1.0 - y_true))
                    * (weights[:, 1] ** (y_true))
                    * K.binary_crossentropy(y_true, y_pred),
                    axis=-1,
                )

            return weighted_loss

        model.compile(optimizer=optimizer, loss=get_weighted_loss(calculate_class_weights(y_train)))

    else:
        model.compile(optimizer=optimizer, loss="binary_crossentropy")

    def custom_learning_rate_schedular(epoch):
        if epoch < 31:
            return 2.5e-4
        elif 31 <= epoch < 37:
            return 2.5000001187436283e-05
        elif 37 <= epoch < 42:
            return 2.5000001187436284e-06

    print("Start training")
    X_test = X_test.reshape(X_test.shape[0], 600, 1)

    from keras.callbacks import LearningRateScheduler

    lrs = LearningRateScheduler(custom_learning_rate_schedular)

    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": 42,
        "batch_size": 1024,
        "verbose": 1,
        "callbacks": [lrs],
    }
    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)

    model.fit(**fit_kwargs)

    prediction = model.predict(X_test)
    return (prediction > 0.5).astype(int), model


def interpolate_to_600(spec):
    old_x = np.arange(len(spec))
    new_x = np.linspace(min(old_x), max(old_x), 600)

    interp = interp1d(old_x, spec)
    new_spec = interp(new_x)
    return new_spec


def make_msms_spectrum(spectrum):
    msms_spectrum = np.zeros(10000)
    for peak in spectrum:
        peak_pos = int(peak[0] * 10)
        if peak_pos >= 10000:
            peak_pos = 9999

        msms_spectrum[peak_pos] = peak[1]

    return msms_spectrum


@click.command()
@click.option("--analytical_data", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--base_out_path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--columns", type=str, required=False, help="Comma-separated list of columns to process"
)
@click.option("--seed", type=int, default=3245)
@click.option("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
@click.option(
    "--use_kfold/--no_kfold", default=True, help="Use K-Fold cross-validation (default: True)"
)
def main(analytical_data, base_out_path, columns, seed, n_folds, use_kfold):
    # Parse columns to process
    columns_to_process = (
        columns.split(",")
        if columns
        else ["h_nmr_spectra", "c_nmr_spectra", "ir_spectra", "pos_msms", "neg_msms"]
    )

    # Map column names to actual parquet column names and output paths
    column_mapping = {
        "h_nmr_spectra": ("h_nmr_spectra", "hnmr"),
        "c_nmr_spectra": ("c_nmr_spectra", "cnmr"),
        "ir_spectra": ("ir_spectra", "ir"),
        "pos_msms": ("msms_positive_40ev", "pos_msms"),
        "neg_msms": ("msms_negative_40ev", "neg_msms"),
    }

    # Process each column SEPARATELY to save memory and time
    for col_name in columns_to_process:
        actual_col, output_dir = column_mapping[col_name]

        print(f"\n{'='*60}")
        print(f"Loading data for: {col_name}")
        print(f"{'='*60}")

        # Only load the columns we need for THIS spectrum type
        columns_to_load = ["smiles", actual_col]

        training_data = None
        parquet_files = list(analytical_data.glob("*.parquet"))
        print(f"Found {len(parquet_files)} parquet files")

        for i, parquet_file in enumerate(parquet_files):
            print(
                f"Loading file {i+1}/{len(parquet_files)}: {parquet_file.name}...",
                end=" ",
                flush=True,
            )

            # Load only necessary columns
            data = pd.read_parquet(parquet_file, columns=columns_to_load)
            print(f"[{len(data)} samples]")

            # Process MSMS columns if present
            if actual_col == "msms_positive_40ev" or actual_col == "msms_negative_40ev":
                data[actual_col] = [make_msms_spectrum(s) for s in data[actual_col]]

            # Compute functional groups
            data["func_group"] = [get_functional_groups(s) for s in data["smiles"]]

            # Interpolate spectrum
            data[actual_col] = [interpolate_to_600(s) for s in data[actual_col]]

            # Concatenate data
            if training_data is None:
                training_data = data
            else:
                training_data = pd.concat((training_data, data))

            del data

        print(f"Total samples loaded: {len(training_data)}")
        print(f"\n{'='*60}")
        print(f"Training model for: {col_name} (column: {actual_col})")
        print(f"{'='*60}")

        # Prepare data
        X_data = np.stack(training_data[actual_col].to_list())
        y_data = np.stack(training_data["func_group"].to_list())

        # First split: 80% train, 20% test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=seed, shuffle=True
        )

        print(f"Initial split: Train={len(X_train_full)} (80%), Test={len(X_test)} (20%)")

        if use_kfold:
            # ============ K-FOLD CROSS VALIDATION ============
            print(f"Performing {n_folds}-fold CV on training set...")

            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_f1_scores = []
            all_predictions = []
            all_targets = []
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_full), 1):
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")

                # Split data for this fold
                X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
                y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

                print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

                # Train model for this fold
                prediction, model = train_model(
                    X_train, y_train, X_val, y_val, X_val, 37, "e", 0, 0
                )

                # Calculate F1 score for this fold
                fold_f1 = f1_score(y_val, prediction, average="micro")
                fold_f1_scores.append(fold_f1)
                print(f"Fold {fold_idx} F1 Score: {fold_f1:.4f}")

                # Store predictions and targets
                all_predictions.append(prediction)
                all_targets.append(y_val)
                fold_models.append(model)

            # Calculate and display cross-validation results
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)
            print(f"\n{'='*60}")
            print(f"Cross-Validation Results for {col_name}:")
            print(f"Mean CV F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"Individual Fold Scores: {[f'{score:.4f}' for score in fold_f1_scores]}")
            print(f"{'='*60}")

            # Select best model and evaluate on held-out test set
            best_fold_idx = np.argmax(fold_f1_scores)
            best_model = fold_models[best_fold_idx]
            print(
                f"\nBest model: Fold {best_fold_idx + 1} (CV F1: {fold_f1_scores[best_fold_idx]:.4f})"
            )

            # Evaluate on held-out test set
            print(f"\nEvaluating on test set ({len(X_test)} samples)...")
            X_test_reshaped = X_test.reshape(X_test.shape[0], 600, 1)
            test_predictions = best_model.predict(X_test_reshaped)
            test_predictions_binary = (test_predictions > 0.5).astype(int)
            test_f1 = f1_score(y_test, test_predictions_binary, average="micro")

            print(f"\n{'='*60}")
            print(f"FINAL TEST RESULTS for {col_name}:")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"{'='*60}")

            # Save results
            out_path = base_out_path / output_dir / "k_fold"
            out_path.mkdir(parents=True, exist_ok=True)

            # Save cross-validation and test results
            cv_results = {
                "fold_scores": fold_f1_scores,
                "mean_cv_f1": mean_f1,
                "std_cv_f1": std_f1,
                "best_fold_idx": best_fold_idx,
                "test_f1": test_f1,
                "test_predictions": test_predictions_binary,
                "test_targets": y_test,
                "all_cv_predictions": all_predictions,
                "all_cv_targets": all_targets,
                "n_folds": n_folds,
                "train_size": len(X_train_full),
                "test_size": len(X_test),
                "seed": seed,
            }
            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(cv_results, file)
            print(f"\nResults saved to: {out_path / 'results.pickle'}")

            # Save the best model
            best_model.save(str(out_path / f"{output_dir}_model.keras"))
            print(f"Best model saved to: {out_path}")

        else:
            # ============ ORIGINAL TRAINING (NO K-FOLD) ============
            print("Training without K-Fold (original mode)...")

            # Split train into train/val (80/20 of train set = 64/16 of total)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.5, random_state=seed, shuffle=True
            )

            print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

            # Train single model
            test_predictions_binary, model = train_model(
                X_train, y_train, X_val, y_val, X_test, 37, "e", 0, 0
            )

            # Calculate F1 score on test set
            test_f1 = f1_score(y_test, test_predictions_binary, average="micro")

            print(f"\n{'='*60}")
            print(f"FINAL TEST RESULTS for {col_name}:")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"{'='*60}")

            # Save results
            out_path = base_out_path / output_dir / "original"
            out_path.mkdir(parents=True, exist_ok=True)

            results = {
                "test_f1": test_f1,
                "pred": test_predictions_binary,
                "tgt": y_test,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "seed": seed,
            }
            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(results, file)
            print(f"\nResults saved to: {out_path / 'results.pickle'}")

            # Save model
            model.save(str(out_path / f"{output_dir}_model.keras"))
            print(f"Model saved to: {out_path}")


if __name__ == "__main__":
    main()
