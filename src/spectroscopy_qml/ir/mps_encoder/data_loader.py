"""
Data loading and preprocessing utilities for IR spectra.
"""

from pathlib import Path
from typing import Optional

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


def apply_quantile_normalization_single_group(spectra: np.ndarray) -> np.ndarray:
    """
    Apply quantile normalization to a single group of spectra.

    Quantile normalization ensures that all spectra have the same distribution
    of intensity values, which is useful for removing technical variations.

    Args:
        spectra: 2D numpy array (n_samples, n_features)

    Returns:
        Quantile-normalized spectra as float32
    """
    sorted_values = np.sort(spectra, axis=1)
    mean_ranks = np.mean(sorted_values, axis=0)

    ranks = np.argsort(np.argsort(spectra, axis=1), axis=1)
    normalized = mean_ranks[ranks]
    return normalized.astype(np.float32)


def apply_quantile_normalization(spectra: np.ndarray) -> np.ndarray:
    """
    Apply quantile normalization across all spectra.

    Args:
        spectra: 2D numpy array (n_samples, n_features)

    Returns:
        Quantile-normalized spectra
    """
    if spectra.ndim != 2:
        raise ValueError("Quantile normalization expects a 2D array (n_samples, n_features).")

    spectra = np.asarray(spectra, dtype=np.float32)
    return apply_quantile_normalization_single_group(spectra)


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


class CNMRSpectraDataset(Dataset):
    """PyTorch Dataset for C-NMR spectra."""

    def __init__(self, spectra: np.ndarray, labels: np.ndarray):
        """
        Args:
            spectra: Array of shape (n_samples, spectrum_length)
            labels: Array of shape (n_samples, num_classes)
        """
        self.spectra = torch.HalfTensor(spectra)  # float16 — halves RAM (~15GB vs ~30GB)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spectra[idx].float(), self.labels[idx]


def load_ir_data(
    data_dir: Path,
    target_length: int = 1800,
    max_files: int | None = None,
    apply_snv: bool = False,
    cache_path: Path | None = None,
    overwrite_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load IR spectra data from parquet files.

    Args:
        data_dir: Directory containing parquet files
        target_length: Target length for spectrum interpolation
        max_files: Maximum number of files to load (for testing)
        apply_snv: Whether to apply SNV normalization (default: False)
        cache_path: Optional path to a preprocessed dataset cache (.npz)
        overwrite_cache: Rebuild cache even if a matching cache already exists

    Returns:
        Tuple of (spectra, labels) as numpy arrays
    """
    print(f"Loading IR spectra from {data_dir}")

    data_dir = Path(data_dir)
    all_parquet_files = sorted(data_dir.glob("*.parquet"))
    if not all_parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    if max_files is not None:
        if int(max_files) <= 0:
            raise ValueError("max_files must be positive when provided.")
        parquet_files = all_parquet_files[: int(max_files)]
    else:
        parquet_files = all_parquet_files

    print(f"Using {len(parquet_files)}/{len(all_parquet_files)} parquet files")

    cache_path = Path(cache_path) if cache_path is not None else None
    source_paths = np.asarray([str(path.resolve()) for path in parquet_files], dtype=str)
    source_mtimes = np.asarray([path.stat().st_mtime_ns for path in parquet_files], dtype=np.int64)

    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        cache_payload = np.load(cache_path, allow_pickle=False)
        cached_source_paths = cache_payload["source_paths"].astype(str, copy=False)
        cached_source_mtimes = cache_payload["source_mtimes"].astype(np.int64, copy=False)
        cache_matches = (
            int(cache_payload["target_length"]) == int(target_length)
            and bool(cache_payload["apply_snv"]) == bool(apply_snv)
            and int(cache_payload["num_files"]) == len(parquet_files)
            and np.array_equal(cached_source_paths, source_paths)
            and np.array_equal(cached_source_mtimes, source_mtimes)
        )
        if cache_matches:
            print(f"Loading cached preprocessed spectra from {cache_path}")
            X = cache_payload["X"]
            y = cache_payload["y"]
            print(f"Total samples loaded: {len(X)}")
            print(f"Spectra shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Label distribution: {y.sum(axis=0)}")
            if apply_snv:
                print("SNV normalization applied")
            return X, y
        print(f"Ignoring stale cache at {cache_path}")

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
    print(f"Label distribution: {y.sum(axis=0)}")
    if apply_snv:
        print("SNV normalization applied")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            target_length=np.asarray(target_length, dtype=np.int64),
            apply_snv=np.asarray(apply_snv, dtype=bool),
            num_files=np.asarray(len(parquet_files), dtype=np.int64),
            source_paths=source_paths,
            source_mtimes=source_mtimes,
        )
        print(f"Saved preprocessed cache to {cache_path}")

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


def load_cnmr_data(
    data_dir: Path,
    target_length: int = 10000,
    max_files: int | None = None,
    apply_snv: bool = False,
    apply_quantile_norm: bool = True,
    cache_path: Path | None = None,
    overwrite_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load C-NMR spectra data from parquet files.

    Args:
        data_dir: Directory containing parquet files
        target_length: Target length for spectrum interpolation (default: 10000 for C-NMR)
        max_files: Maximum number of files to load (for testing)
        apply_snv: Whether to apply SNV normalization (default: False)
        apply_quantile_norm: Whether to apply quantile normalization (default: True for C-NMR)
        cache_path: Optional path to a preprocessed dataset cache (.npz)
        overwrite_cache: Rebuild cache even if a matching cache already exists

    Returns:
        Tuple of (spectra, labels) as numpy arrays
    """
    print(f"Loading C-NMR spectra from {data_dir}")

    data_dir = Path(data_dir)
    all_parquet_files = sorted(data_dir.glob("*.parquet"))
    if not all_parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    if max_files is not None:
        if int(max_files) <= 0:
            raise ValueError("max_files must be positive when provided.")
        parquet_files = all_parquet_files[: int(max_files)]
    else:
        parquet_files = all_parquet_files

    print(f"Using {len(parquet_files)}/{len(all_parquet_files)} parquet files")

    cache_path = Path(cache_path) if cache_path is not None else None
    source_paths = np.asarray([str(path.resolve()) for path in parquet_files], dtype=str)
    source_mtimes = np.asarray([path.stat().st_mtime_ns for path in parquet_files], dtype=np.int64)

    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        cache_payload = np.load(cache_path, allow_pickle=False)
        cached_source_paths = cache_payload["source_paths"].astype(str, copy=False)
        cached_source_mtimes = cache_payload["source_mtimes"].astype(np.int64, copy=False)
        # Check if cache has QN metadata; if not, invalidate cache (assume old cache without QN)
        cached_apply_qn = bool(cache_payload.get("apply_quantile_norm", False))
        cache_matches = (
            int(cache_payload["target_length"]) == int(target_length)
            and bool(cache_payload["apply_snv"]) == bool(apply_snv)
            and cached_apply_qn == bool(apply_quantile_norm)
            and int(cache_payload["num_files"]) == len(parquet_files)
            and np.array_equal(cached_source_paths, source_paths)
            and np.array_equal(cached_source_mtimes, source_mtimes)
        )
        if cache_matches:
            print(f"Loading cached preprocessed spectra from {cache_path}")
            X = cache_payload["X"]
            y = cache_payload["y"]
            print(f"Total samples loaded: {len(X)}")
            print(f"Spectra shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Label distribution: {y.sum(axis=0)}")
            if apply_snv:
                print("SNV normalization applied")
            if apply_quantile_norm:
                print("Quantile normalization applied")
            return X, y
        print(f"Ignoring stale cache at {cache_path} (mismatch in preprocessing options)")

    all_spectra = []
    all_labels = []

    for i, parquet_file in enumerate(parquet_files):
        # Load only necessary columns for C-NMR
        df = pd.read_parquet(parquet_file, columns=["c_nmr_spectra", "smiles"])

        # Extract functional groups
        df["func_groups"] = df["smiles"].map(get_functional_groups)

        # Filter out invalid entries
        df = df[df["func_groups"].notna()]
        df = df[df["c_nmr_spectra"].notna()]

        # Interpolate spectra
        spectra = np.stack(
            [interpolate_spectrum(spec, target_length) for spec in df["c_nmr_spectra"].values]
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
    print(f"Label distribution: {y.sum(axis=0)}")
    if apply_snv:
        print("SNV normalization applied")
    
    # Apply quantile normalization if requested
    if apply_quantile_norm:
        print(f"[DEBUG] Starting quantile normalization on {X.shape} array...")
        import sys
        sys.stdout.flush()
        import time
        start = time.time()
        X = apply_quantile_normalization(X)
        elapsed = time.time() - start
        print(f"[DEBUG] Quantile normalization completed in {elapsed:.1f}s")
        print("Quantile normalization applied")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Starting cache save to {cache_path}...")
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            target_length=np.asarray(target_length, dtype=np.int64),
            apply_snv=np.asarray(apply_snv, dtype=bool),
            apply_quantile_norm=np.asarray(apply_quantile_norm, dtype=bool),
            num_files=np.asarray(len(parquet_files), dtype=np.int64),
            source_paths=source_paths,
            source_mtimes=source_mtimes,
        )
        print(f"Saved preprocessed cache to {cache_path}")

    return X, y
