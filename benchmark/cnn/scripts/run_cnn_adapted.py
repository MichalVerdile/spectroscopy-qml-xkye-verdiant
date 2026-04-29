# Adapted from Guwon Jung: https://github.com/gj475/irchracterizationcnn
# PyTorch version preserving the original script logic, architecture, hyperparameters,
# splits, threshold tuning, K-Fold/original modes, and output structure as closely as possible.

import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU(s) detected: {torch.cuda.device_count()} device(s)")
    print(f"  {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    torch.cuda.set_device(3)
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


def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) is Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)

    return 0 if n == 0 else 1


def get_functional_groups(smiles: str):
    RDLogger.DisableLog("rdApp.*")

    smiles = smiles.strip().replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    func_groups = []

    for _, smarts in functional_groups.items():
        func_groups.append(match_group(mol, smarts))

    return func_groups


def tune_per_label_thresholds(y_true, y_prob):
    thresholds = np.zeros(y_true.shape[1])

    for label_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1

        for threshold in np.linspace(0.05, 0.95, 91):
            y_pred_label = (y_prob[:, label_idx] >= threshold).astype(int)

            score = f1_score(
                y_true[:, label_idx],
                y_pred_label,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds[label_idx] = best_threshold

    return thresholds


def apply_per_label_thresholds(y_prob, thresholds):
    return (y_prob >= thresholds.reshape(1, -1)).astype(int)


class FunctionalGroupCNN(nn.Module):
    def __init__(self, input_length, num_fgs):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=31,
                kernel_size=11,
                stride=1,
                padding=5,
            ),
            nn.BatchNorm1d(31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=31,
                out_channels=62,
                kernel_size=11,
                stride=1,
                padding=5,
            ),
            nn.BatchNorm1d(62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 4927),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(4927, 2785),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(2785, 1574),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(1574, num_fgs),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def predict_torch(model, X):
    model.eval()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    predictions = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb)
            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def train_model(X_train, y_train, X_val, y_val, X_test, num_fgs, aug, num, weighted):
    """Trains model and tunes one threshold per functional-group label."""

    input_length = X_train.shape[1]

    X_train = X_train.reshape(X_train.shape[0], 1, input_length)

    if X_val is not None:
        X_val = X_val.reshape(X_val.shape[0], 1, input_length)

    model = FunctionalGroupCNN(input_length=input_length, num_fgs=num_fgs).to(device)

    print("Model Construction")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters())

    if weighted == 1:
        number_dim = np.shape(y_train)[1]
        weights = np.zeros((2, number_dim))

        for i in range(number_dim):
            neg_count = (y_train[:, i] == 0).sum()
            pos_count = (y_train[:, i] == 1).sum()

            weights_n = y_train.shape[0] / (2 * neg_count) if neg_count > 0 else 1.0
            weights_p = y_train.shape[0] / (2 * pos_count) if pos_count > 0 else 1.0

            weights[1, i], weights[0, i] = weights_p, weights_n

        weights = torch.tensor(weights.T, dtype=torch.float32, device=device)

        def criterion(y_pred, y_true):
            bce = nn.functional.binary_cross_entropy(
                y_pred,
                y_true,
                reduction="none",
            )

            sample_weights = (
                (weights[:, 0] ** (1.0 - y_true))
                * (weights[:, 1] ** y_true)
            )

            return torch.mean(sample_weights * bce)

    else:
        criterion = nn.BCELoss()

    def custom_learning_rate_schedular(epoch):
        if epoch < 31:
            return 2.5e-4
        elif 31 <= epoch < 37:
            return 2.5000001187436283e-5
        else:
            return 2.5000001187436284e-6

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
    )

    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1024,
            shuffle=False,
        )
    else:
        val_loader = None

    print("Start training")

    for epoch in range(42):
        lr = custom_learning_rate_schedular(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    y_pred = model(xb)
                    loss = criterion(y_pred, yb)

                    batch_size = xb.size(0)
                    val_loss_sum += loss.item() * batch_size
                    val_samples += batch_size

            val_loss = val_loss_sum / val_samples

            print(
                f"Epoch {epoch + 1}/42 - "
                f"lr: {lr:.10f} - "
                f"loss: {train_loss:.6f} - "
                f"val_loss: {val_loss:.6f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/42 - "
                f"lr: {lr:.10f} - "
                f"loss: {train_loss:.6f}"
            )

    if X_val is not None and y_val is not None:
        val_prob = predict_torch(model, X_val)
        thresholds = tune_per_label_thresholds(y_val, val_prob)
    else:
        thresholds = np.full(num_fgs, 0.5)

    X_test = X_test.reshape(X_test.shape[0], 1, input_length)
    test_prob = predict_torch(model, X_test)
    test_pred = apply_per_label_thresholds(test_prob, thresholds)

    return test_pred, model, thresholds


def interpolate_to_length(spec, target_length):
    old_x = np.arange(len(spec))
    new_x = np.linspace(old_x.min(), old_x.max(), target_length)

    interp = interp1d(old_x, spec, bounds_error=False, fill_value=0)
    return interp(new_x)


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
    "--columns",
    type=str,
    required=False,
    help="Comma-separated list of columns to process",
)
@click.option("--seed", type=int, default=42)
@click.option("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
@click.option(
    "--use_kfold/--no_kfold",
    default=False,
    help="Use K-Fold cross-validation (default: True)",
)
def main(analytical_data, base_out_path, columns, seed, n_folds, use_kfold):
    columns_to_process = (
        columns.split(",")
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

    for col_name in columns_to_process:
        actual_col, output_dir = column_mapping[col_name]

        target_length = 1800 if col_name == "ir_spectra" else 10000

        print(f"\n{'=' * 60}")
        print(f"Loading data for: {col_name}")
        print(f"Target interpolation length: {target_length}")
        print(f"{'=' * 60}")

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

            data = data.dropna(subset=["func_group", actual_col])

            data[actual_col] = [
                interpolate_to_length(s, target_length) for s in data[actual_col]
            ]

            if training_data is None:
                training_data = data
            else:
                training_data = pd.concat((training_data, data), ignore_index=True)

            del data

        print(f"Total samples loaded: {len(training_data)}")

        print(f"\n{'=' * 60}")
        print(f"Training model for: {col_name} column: {actual_col}")
        print(f"{'=' * 60}")

        X_data = np.stack(training_data[actual_col].to_list())
        y_data = np.stack(training_data["func_group"].to_list())

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_data,
            y_data,
            test_size=0.1,
            random_state=seed,
            shuffle=True,
        )

        print(f"Initial split: Train={len(X_train_full)} 80%, Test={len(X_test)} 20%")

        if use_kfold:
            print(f"Performing {n_folds}-fold CV on training set...")

            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

            fold_f1_scores = []
            fold_models = []
            fold_thresholds = []

            all_predictions = []
            all_targets = []

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_full), 1):
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")

                X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
                y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

                print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

                prediction, model, thresholds = train_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_val,
                    37,
                    "e",
                    0,
                    0,
                )

                fold_f1 = f1_score(y_val, prediction, average="micro")
                fold_f1_scores.append(fold_f1)

                print(f"Fold {fold_idx} F1 Score: {fold_f1:.4f}")

                all_predictions.append(prediction)
                all_targets.append(y_val)

                fold_models.append(model)
                fold_thresholds.append(thresholds)

            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)

            print(f"\n{'=' * 60}")
            print(f"Cross-Validation Results for {col_name}:")
            print(f"Mean CV F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"Individual Fold Scores: {[f'{score:.4f}' for score in fold_f1_scores]}")
            print(f"{'=' * 60}")

            best_fold_idx = np.argmax(fold_f1_scores)
            best_model = fold_models[best_fold_idx]
            best_thresholds = fold_thresholds[best_fold_idx]

            print(
                f"\nBest model: Fold {best_fold_idx + 1} "
                f"CV F1: {fold_f1_scores[best_fold_idx]:.4f}"
            )

            print(f"\nEvaluating on test set {len(X_test)} samples...")

            X_test_reshaped = X_test.reshape(X_test.shape[0], 1, target_length)
            test_predictions_prob = predict_torch(best_model, X_test_reshaped)
            test_predictions_binary = apply_per_label_thresholds(
                test_predictions_prob,
                best_thresholds,
            )

            test_f1 = f1_score(y_test, test_predictions_binary, average="micro")

            print(f"\n{'=' * 60}")
            print(f"FINAL TEST RESULTS for {col_name}:")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"{'=' * 60}")

            out_path = base_out_path / output_dir / "k_fold"
            out_path.mkdir(parents=True, exist_ok=True)

            cv_results = {
                "fold_scores": fold_f1_scores,
                "mean_cv_f1": mean_f1,
                "std_cv_f1": std_f1,
                "best_fold_idx": best_fold_idx,
                "test_f1": test_f1,
                "test_predictions": test_predictions_binary,
                "test_targets": y_test,
                "test_probabilities": test_predictions_prob,
                "all_cv_predictions": all_predictions,
                "all_cv_targets": all_targets,
                "fold_thresholds": fold_thresholds,
                "best_thresholds": best_thresholds,
                "target_length": target_length,
                "n_folds": n_folds,
                "train_size": len(X_train_full),
                "test_size": len(X_test),
                "seed": seed,
            }

            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(cv_results, file)

            print(f"\nResults saved to: {out_path / 'results.pickle'}")

            torch.save(
                {
                    "model_state_dict": best_model.state_dict(),
                    "target_length": target_length,
                    "num_fgs": 37,
                },
                out_path / f"{output_dir}_model.pt",
            )
            print(f"Best model saved to: {out_path}")

        else:
            print("Training without K-Fold original mode...")

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=0.11,
                random_state=seed,
                shuffle=True,
            )

            print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

            test_predictions_binary, model, thresholds = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                37,
                "e",
                0,
                0,
            )

            test_f1 = f1_score(y_test, test_predictions_binary, average="micro")

            print(f"\n{'=' * 60}")
            print(f"FINAL TEST RESULTS for {col_name}:")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"{'=' * 60}")

            out_path = base_out_path / output_dir / "original"
            out_path.mkdir(parents=True, exist_ok=True)

            results = {
                "test_f1": test_f1,
                "pred": test_predictions_binary,
                "tgt": y_test,
                "thresholds": thresholds,
                "target_length": target_length,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "seed": seed,
            }

            with open(out_path / "results.pickle", "wb") as file:
                pickle.dump(results, file)

            print(f"\nResults saved to: {out_path / 'results.pickle'}")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_length": target_length,
                    "num_fgs": 37,
                },
                out_path / f"{output_dir}_model.pt",
            )
            print(f"Model saved to: {out_path}")


if __name__ == "__main__":
    main()
