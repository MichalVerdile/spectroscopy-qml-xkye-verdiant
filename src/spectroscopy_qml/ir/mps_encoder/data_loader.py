from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Functional groups definitions (same as Jung et al.)
FUNCTIONAL_GROUPS = {
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
    """Check if molecule contains functional group."""
    if type(func_group) is Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1


def get_functional_groups(smiles: str) -> list | None:
    """Extract functional group labels from SMILES string."""
    RDLogger.DisableLog("rdApp.*")
    smiles = smiles.strip().replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    func_groups = []
    for func_group_name, smarts in FUNCTIONAL_GROUPS.items():
        func_groups.append(match_group(mol, smarts))

    return func_groups


def interpolate_spectrum(spectrum: np.ndarray, target_length: int = 1800) -> np.ndarray:
    """
    Interpolate spectrum to target length.

    Args:
        spectrum: Input spectrum array
        target_length: Desired output length

    Returns:
        Interpolated spectrum of specified length
    """
    if len(spectrum) == target_length:
        return spectrum

    old_x = np.arange(len(spectrum))
    new_x = np.linspace(0, len(spectrum) - 1, target_length)

    interp_func = interp1d(old_x, spectrum, kind="linear")
    new_spectrum = interp_func(new_x)

    return new_spectrum


def apply_snv_normalization(spectrum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply Standard Normal Variate (SNV) normalization to a spectrum.

    SNV removes scatter effects by normalizing each spectrum to zero mean
    and unit variance. This is a common preprocessing technique for
    spectroscopic data.

    Args:
        spectrum: Input spectrum array
        eps: Small value to prevent division by zero (default: 1e-8)

    Returns:
        SNV-normalized spectrum
    """
    mean = np.mean(spectrum)
    std = np.std(spectrum)

    # Prevent division by zero
    if std < eps:
        return spectrum - mean

    return (spectrum - mean) / std


def apply_savgol_smoothing(
    spectrum: np.ndarray, window_length: int = 11, polyorder: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter to a spectrum.

    Smooths the spectrum to reduce noise before derivative computation.

    Args:
        spectrum: Input spectrum array
        window_length: Length of the filter window (must be odd and > polyorder)
        polyorder: Order of the polynomial used to fit the samples

    Returns:
        Smoothed spectrum
    """
    if len(spectrum) < window_length:
        return spectrum
    return savgol_filter(spectrum, window_length=window_length, polyorder=polyorder)


class IRSpectraDataset(Dataset):
    """PyTorch Dataset for IR spectra."""

    def __init__(self, spectra: np.ndarray, labels: np.ndarray):
        """
        Args:
            spectra: Array of shape (n_samples, spectrum_length)
            labels: Array of shape (n_samples, num_classes)
        """
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spectra[idx], self.labels[idx]


def load_ir_data(
    data_dir: Path,
    target_length: int = 1800,
    max_files: int | None = None,
    apply_snv: bool = False,
    apply_savgol: bool = False,
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load IR spectra data from parquet files.

    Args:
        data_dir: Directory containing parquet files
        target_length: Target length for spectrum interpolation
        max_files: Maximum number of files to load (for testing)
        apply_snv: Whether to apply SNV normalization (default: False)
        apply_savgol: Whether to apply Savitzky-Golay smoothing (default: False)
        savgol_window_length: Window length for Savitzky-Golay filter
        savgol_polyorder: Polynomial order for Savitzky-Golay filter

    Returns:
        Tuple of (spectra, labels) as numpy arrays
    """
    print(f"Loading IR spectra from {data_dir}")

    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("*.parquet"))

    if max_files:
        parquet_files = parquet_files[:max_files]

    all_spectra = []
    all_labels = []

    for i, parquet_file in enumerate(parquet_files):
        # Load only necessary columns
        df = pd.read_parquet(parquet_file, columns=["ir_spectra", "smiles"])

        # Extract functional groups
        df["func_groups"] = df["smiles"].map(get_functional_groups)

        # Filter out invalid entries
        df = df[df["func_groups"].notna()]
        df = df[df["ir_spectra"].notna()]

        # Interpolate spectra
        spectra = np.stack(
            [interpolate_spectrum(spec, target_length) for spec in df["ir_spectra"].values]
        )

        # Apply Savitzky-Golay smoothing before derivatives (cleans noise)
        if apply_savgol:
            spectra = np.stack(
                [apply_savgol_smoothing(spec, savgol_window_length, savgol_polyorder) for spec in spectra]
            )

        # Apply SNV normalization if requested
        if apply_snv:
            spectra = np.stack([apply_snv_normalization(spec) for spec in spectra])

        labels = np.stack(df["func_groups"].values)

        all_spectra.append(spectra)
        all_labels.append(labels)

        if (i + 1) % 10 == 0:
            print(f"Loaded {i + 1}/{len(parquet_files)} files")

    # Concatenate all data
    X = np.vstack(all_spectra)
    y = np.vstack(all_labels)

    print(f"Total samples loaded: {len(X)}")
    print(f"Spectra shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    label_dist = {name: int(count) for name, count in zip(FUNCTIONAL_GROUPS.keys(), y.sum(axis=0))}
    print("Label distribution:")
    for name, count in label_dist.items():
        print(f"  {name}: {count}")
    if apply_savgol:
        print(f"Savitzky-Golay smoothing applied (window={savgol_window_length}, order={savgol_polyorder})")
    if apply_snv:
        print("SNV normalization applied")

    return X, y


def prepare_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        X: Spectra array
        y: Labels array
        batch_size: Batch size for dataloaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading (0 = single process)
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, shuffle=True
    )

    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_seed, shuffle=True
    )

    print("\nData split:")
    print(f"  Train: {len(X_train)} samples ({train_ratio:.1%})")
    print(f"  Val:   {len(X_val)} samples ({val_ratio:.1%})")
    print(f"  Test:  {len(X_test)} samples ({test_ratio:.1%})")

    # Create datasets
    train_dataset = IRSpectraDataset(X_train, y_train)
    val_dataset = IRSpectraDataset(X_val, y_val)
    test_dataset = IRSpectraDataset(X_test, y_test)

    # Create dataloaders with optimization settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader
