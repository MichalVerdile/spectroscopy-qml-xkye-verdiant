"""
Dataset module for IR spectra functional group prediction.

Handles:
- Loading IR spectra from parquet files => (benchmark/data/raw/ (aligned_chunk_*.parquet))
- Extracting functional group labels from SMILES using RDKit
- Resampling to fixed grid
- Intensity normalization (z-score)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

# Functional groups defined by SMARTS patterns (37 classes, matching benchmark)
FUNCTIONAL_GROUPS: dict[str, str] = {
    "Acid anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "Acyl halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "Alcohol": "[#6][OX2H]",
    "Aldehyde": "[CX3H1](=O)[#6,H]",
    "Alkane": "[CX4;H3,H2]",
    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#[CX2]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "Arene": "[cX3]1[cX3][cX3][cX3][cX3][cX3]1",
    "Azo compound": "[#6][NX2]=[NX2][#6]",
    "Carbamate": "[NX3][CX3](=[OX1])[OX2H0]",
    "Carboxylic acid": "[CX3](=O)[OX2H]",
    "Enamine": "[NX3][CX3]=[CX3]",
    "Enol": "[OX2H][#6X3]=[#6]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "Ether": "[OD2]([#6])[#6]",
    "Haloalkane": "[#6][F,Cl,Br,I]",
    "Hydrazine": "[NX3][NX3]",
    "Hydrazone": "[NX3][NX2]=[#6]",
    "Imide": "[CX3](=[OX1])[NX3][CX3](=[OX1])",
    "Imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Isocyanate": "[NX2]=[C]=[O]",
    "Isothiocyanate": "[NX2]=[C]=[S]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Nitrile": "[NX1]#[CX2]",
    "Phenol": "[OX2H][cX3]:[c]",
    "Phosphine": "[PX3]",
    "Sulfide": "[#16X2H0]",
    "Sulfonamide": "[#16X4]([NX3])(=[OX1])(=[OX1])[#6]",
    "Sulfonate": "[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]",
    "Sulfone": "[#16X4](=[OX1])(=[OX1])([#6])[#6]",
    "Sulfonic acid": "[#16X4](=[OX1])(=[OX1])([#6])[OX2H]",
    "Sulfoxide": "[#16X3]=[OX1]",
    "Thial": "[CX3H1](=S)[#6,H]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    "Thiol": "[#16X2H]",
}

# Precompile SMARTS patterns for efficiency
_COMPILED_PATTERNS: dict[str, Chem.Mol | None] = {}


# Compile SMARTS once and then reuse it.
def _get_compiled_pattern(name: str, smarts: str) -> Chem.Mol | None:
    """Get or create compiled SMARTS pattern."""
    if name not in _COMPILED_PATTERNS:
        _COMPILED_PATTERNS[name] = Chem.MolFromSmarts(smarts)
    return _COMPILED_PATTERNS[name]


def extract_functional_groups(smiles: str, groups: dict[str, str] | None = None) -> dict[str, bool]:
    """
    Extract functional group presence from SMILES string.

    Args:
        smiles: SMILES string of the molecule
        groups: Dictionary of {name: SMARTS} patterns (defaults to FUNCTIONAL_GROUPS)

    Returns:
        Dictionary of {group_name: is_present}
    """
    if groups is None:
        groups = FUNCTIONAL_GROUPS

    mol = Chem.MolFromSmiles(smiles)
    result: dict[str, bool] = {}
    """
    The SMILES string is first converted by RDKit into a molecule graph object (MolFromSmiles),
    and then HasSubstructMatch(pattern) checks whether the substructure defined by the SMARTS pattern
    is present in this molecule (True/False).
    """
    if mol is None:
        return dict.fromkeys(groups, False)

    for name, smarts in groups.items():
        pattern = _get_compiled_pattern(name, smarts)
        if pattern is not None:
            result[name] = mol.HasSubstructMatch(pattern)
        else:
            result[name] = False

    return result


def resample_spectrum(
    spectrum: np.ndarray,
    target_length: int,
    method: Literal["linear", "cubic"] = "linear",
) -> np.ndarray:
    """
    Resample spectrum to fixed grid length.

    Args:
        spectrum: 1D array of spectral intensities
        target_length: Desired output length
        method: Interpolation method

    Returns:
        Resampled spectrum of shape (target_length,)
    """
    if len(spectrum) == target_length:
        return spectrum.astype(np.float32)

    x_old = np.linspace(0, 1, len(spectrum))
    x_new = np.linspace(0, 1, target_length)

    interpolator = interp1d(x_old, spectrum, kind=method, fill_value="extrapolate")
    result: np.ndarray = np.asarray(interpolator(x_new), dtype=np.float32)
    return result


def normalize_spectrum(
    spectrum: np.ndarray,
    method: Literal["zscore"] = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize spectrum intensity.

    Args:
        spectrum: 1D array of spectral intensities
        method: Normalization method ("zscore")
        eps: Small constant for numerical stability

    Returns:
        Normalized spectrum
    """
    spectrum = spectrum.astype(np.float32)

    if method != "zscore":
        raise ValueError(f"Unknown normalization method: {method}")
    mean = float(np.mean(spectrum))
    std = float(np.std(spectrum))
    result = (spectrum - mean) / (std + eps)
    return result


class IRFunctionalGroupDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for IR spectra with functional group labels.

    Loads IR spectra from parquet files, extracts functional group labels
    from SMILES strings, and applies preprocessing.
    """

    def __init__(
        self,
        data_dir: str | Path,
        target_length: int = 600,
        normalization: Literal["zscore"] = "zscore",
        functional_groups: dict[str, str] | None = None,
        max_chunks: int | None = None,
        cache_labels: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to directory containing parquet files
            target_length: Resample spectra to this length
            normalization: Normalization method ("zscore")
            functional_groups: Custom functional groups dict (defaults to FUNCTIONAL_GROUPS)
            max_chunks: Maximum number of parquet chunks to load (for testing)
            cache_labels: Whether to precompute and cache labels
        """
        self.data_dir = Path(data_dir)
        self.target_length = target_length
        self.normalization = normalization
        self.functional_groups = functional_groups or FUNCTIONAL_GROUPS
        self.group_names = list(self.functional_groups.keys())
        self.num_classes = len(self.group_names)

        self._load_data(max_chunks)

        # Cache labels if requested
        self._labels_cache: np.ndarray | None = None
        if cache_labels:
            self._precompute_labels()

    def _load_data(self, max_chunks: int | None = None) -> None:
        parquet_files = sorted(self.data_dir.glob("aligned_chunk_*.parquet"))

        if max_chunks is not None:
            parquet_files = parquet_files[:max_chunks]

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f, columns=["smiles", "ir_spectra"])
            dfs.append(df)

        self.data = pd.concat(dfs, ignore_index=True)

        # Filter out rows with missing IR spectra
        valid_mask = self.data["ir_spectra"].apply(lambda x: x is not None and len(x) > 0)
        self.data = self.data[valid_mask].reset_index(drop=True)

        print(f"Loaded {len(self.data)} samples from {len(parquet_files)} parquet files")

    def _precompute_labels(self) -> None:
        """Precompute all functional group labels."""
        labels = []
        for smiles in self.data["smiles"]:
            fg = extract_functional_groups(smiles, self.functional_groups)
            labels.append([int(fg[name]) for name in self.group_names])
        self._labels_cache = np.array(labels, dtype=np.float32)
        print(f"Label distribution: {self._labels_cache.sum(axis=0).astype(int)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (spectrum, labels) tensors
        """
        row = self.data.iloc[idx]

        # Get and preprocess spectrum
        spectrum = np.array(row["ir_spectra"], dtype=np.float32)
        spectrum = resample_spectrum(spectrum, self.target_length)
        spectrum = normalize_spectrum(spectrum, self.normalization)

        # Get labels
        if self._labels_cache is not None:
            labels = self._labels_cache[idx]
        else:
            fg = extract_functional_groups(row["smiles"], self.functional_groups)
            labels = np.array([float(fg[name]) for name in self.group_names], dtype=np.float32)

        return torch.from_numpy(spectrum), torch.from_numpy(labels)

    def get_label_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced labels.

        Returns:
            Tensor of shape (num_classes,) with positive class weights
        """
        if self._labels_cache is None:
            self._precompute_labels()

        assert self._labels_cache is not None  # for type checker
        pos_counts = self._labels_cache.sum(axis=0)
        neg_counts = len(self._labels_cache) - pos_counts

        # Avoid division by zero
        pos_counts = np.maximum(pos_counts, 1)

        weights = neg_counts / pos_counts
        return torch.from_numpy(weights.astype(np.float32))

    def get_active_class_mask(self) -> torch.Tensor:
        """
        Return a boolean mask for classes that occur at least once.

        Classes with zero positives in the loaded subset are excluded from
        loss/metrics to avoid unstable optimization and misleading macro scores.
        """
        if self._labels_cache is None:
            self._precompute_labels()

        assert self._labels_cache is not None
        pos_counts = self._labels_cache.sum(axis=0)
        mask = pos_counts > 0
        return torch.from_numpy(mask.astype(np.bool_))


def create_data_splits(
    dataset: IRFunctionalGroupDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[
    torch.utils.data.Subset[tuple[torch.Tensor, torch.Tensor]],
    torch.utils.data.Subset[tuple[torch.Tensor, torch.Tensor]],
    torch.utils.data.Subset[tuple[torch.Tensor, torch.Tensor]],
]:
    """
    Create train/val/test splits.

    Args:
        dataset: The full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()

    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
        torch.utils.data.Subset(dataset, test_indices),
    )
