"""
Data loading and preprocessing utilities for spectroscopy data.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
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


class SpectraDataset(Dataset):
    """PyTorch Dataset for spectroscopy tensors and labels."""

    def __init__(self, spectra: np.ndarray, labels: np.ndarray):
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spectra[idx], self.labels[idx]


# Backward-compatible alias
IRSpectraDataset = SpectraDataset


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
    for _, smarts in FUNCTIONAL_GROUPS.items():
        func_groups.append(match_group(mol, smarts))

    return func_groups


def interpolate_spectrum(spectrum: np.ndarray, target_length: int = 1800) -> np.ndarray:
    """Interpolate spectrum to target length."""
    if len(spectrum) == target_length:
        return spectrum

    old_x = np.arange(len(spectrum), dtype=np.float32)
    new_x = np.linspace(0, len(spectrum) - 1, target_length, dtype=np.float32)
    interp_func = interp1d(old_x, spectrum, kind="linear")
    return interp_func(new_x)


def apply_snv_normalization(spectrum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply Standard Normal Variate normalization to one spectrum."""
    mean = np.mean(spectrum)
    std = np.std(spectrum)

    if std < eps:
        return spectrum - mean

    return (spectrum - mean) / std


def _load_ir_spectra(
    data_dir: Path,
    target_length: int,
    max_files: int | None,
    batch_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load IR spectra from parquet files.

    Args:
        data_dir: Directory containing parquet files
        target_length: Target length for interpolation
        max_files: Maximum number of files to load
        batch_transform: Transform applied to batch of spectra (per file)

    Returns:
        (X, y) tuple of spectra and labels
    """
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("*.parquet"))

    if max_files:
        parquet_files = parquet_files[:max_files]

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    all_spectra: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for i, parquet_file in enumerate(parquet_files):
        df = pd.read_parquet(parquet_file, columns=["ir_spectra", "smiles"])
        df["func_groups"] = df["smiles"].map(get_functional_groups)

        df = df[df["func_groups"].notna()]
        df = df[df["ir_spectra"].notna()]

        processed_spectra: list[np.ndarray] = []
        labels: list[list[int]] = []

        for raw_spec, label in zip(df["ir_spectra"].values, df["func_groups"].values):
            spec = np.asarray(raw_spec, dtype=np.float32)
            if spec.ndim != 1:
                continue

            spec = interpolate_spectrum(spec, target_length)
            processed_spectra.append(spec)
            labels.append(label)

        if not processed_spectra:
            continue

        spectra = np.stack(processed_spectra).astype(np.float32)
        if batch_transform is not None:
            spectra = batch_transform(spectra)

        label_array = np.stack(labels).astype(np.float32)

        all_spectra.append(spectra)
        all_labels.append(label_array)

        if (i + 1) % 10 == 0:
            print(f"Loaded {i + 1}/{len(parquet_files)} files")

    if not all_spectra:
        raise ValueError(f"No valid IR spectra loaded from directory {data_dir}")

    X = np.vstack(all_spectra)
    y = np.vstack(all_labels)

    return X, y


def load_ir_data(
    data_dir: Path,
    target_length: int = 1800,
    max_files: int | None = None,
    apply_snv: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load IR spectra with optional SNV normalization.

    Args:
        data_dir: Directory with parquet files
        target_length: Target spectrum length after interpolation
        max_files: Maximum number of files to load
        apply_snv: Whether to apply SNV normalization

    Returns:
        (X, y) tuple of spectra and labels
    """
    print(f"Loading IR spectra from {data_dir}")

    batch_transform = None
    if apply_snv:
        batch_transform = lambda arr: np.stack([apply_snv_normalization(spec) for spec in arr])

    X, y = _load_ir_spectra(
        data_dir=data_dir,
        target_length=target_length,
        max_files=max_files,
        batch_transform=batch_transform,
    )

    print(f"Total samples loaded: {len(X)}")
    print(f"Spectra shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {y.sum(axis=0)}")
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
    """Create train/validation/test dataloaders."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, shuffle=True
    )

    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_seed, shuffle=True
    )

    print("\nData split:")
    print(f"  Train: {len(X_train)} samples ({train_ratio:.1%})")
    print(f"  Val:   {len(X_val)} samples ({val_ratio:.1%})")
    print(f"  Test:  {len(X_test)} samples ({test_ratio:.1%})")

    train_dataset = SpectraDataset(X_train, y_train)
    val_dataset = SpectraDataset(X_val, y_val)
    test_dataset = SpectraDataset(X_test, y_test)

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
