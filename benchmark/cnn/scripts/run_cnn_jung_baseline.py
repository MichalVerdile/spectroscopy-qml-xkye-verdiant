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
from sklearn.model_selection import train_test_split

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
        print(f"✓ GPU(s) detected: {len(gpus)} device(s)")
        print(f"  {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU detected - running on CPU")

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

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=42,
        batch_size=1024,
        verbose=1,
        callbacks=[lrs],
    )

    prediction = model.predict(X_test)
    return (prediction > 0.5).astype(int)


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
@click.option("--seed", type=int, default=42)
def main(analytical_data, base_out_path, columns, seed):
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

    # Get all actual column names needed
    actual_columns = set(["smiles"])
    for col in columns_to_process:
        actual_col, _ = column_mapping[col]
        actual_columns.add(actual_col)

    print(f"Loading data for columns: {columns_to_process}")
    print(f"Reading columns from parquet: {actual_columns}")
    training_data = None

    for i, parquet_file in enumerate(analytical_data.glob("*.parquet")):
        data = pd.read_parquet(parquet_file, columns=list(actual_columns))

        # Process MSMS columns if present
        if "msms_positive_40ev" in data.columns:
            data["msms_positive_40ev"] = data["msms_positive_40ev"].map(make_msms_spectrum)
        if "msms_negative_40ev" in data.columns:
            data["msms_negative_40ev"] = data["msms_negative_40ev"].map(make_msms_spectrum)

        # Compute functional groups once
        data["func_group"] = data.smiles.map(get_functional_groups)

        # Interpolate all spectrum columns
        for col in actual_columns:
            if col != "smiles" and col in data.columns:
                data[col] = data[col].map(interpolate_to_600)

        if training_data is None:
            training_data = data
        else:
            training_data = pd.concat((training_data, data))
        del data

        print(f"Loaded parquet file {i+1}")

    print(f"Total samples loaded: {len(training_data)}")

    # Split data: 80% train, 10% val, 10% test
    train_val, test = train_test_split(training_data, test_size=0.1, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.111, random_state=seed
    )  # 0.111 of 0.9 ≈ 0.1

    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    # Process each column
    for col_name in columns_to_process:
        actual_col, output_dir = column_mapping[col_name]
        print(f"\n{'='*60}")
        print(f"Training model for: {col_name} (column: {actual_col})")
        print(f"{'='*60}")

        X_train = np.stack(train[actual_col].to_list())
        y_train = np.stack(train["func_group"].to_list())
        X_val = np.stack(val[actual_col].to_list())
        y_val = np.stack(val["func_group"].to_list())
        X_test = np.stack(test[actual_col].to_list())
        y_test = np.stack(test["func_group"].to_list())

        # Train model
        prediction = train_model(X_train, y_train, X_val, y_val, X_test, 37, "e", 0, 0)

        f1 = f1_score(y_test, prediction, average="micro")
        print(f"F1 Score for {col_name}: {f1}")

        # Save results
        out_path = base_out_path / output_dir
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "results.pickle", "wb") as file:
            pickle.dump({"pred": prediction, "tgt": y_test}, file)

        print(f"Results saved to: {out_path / 'results.pickle'}")


if __name__ == "__main__":
    main()
