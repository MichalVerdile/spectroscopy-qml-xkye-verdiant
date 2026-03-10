"""
Data loading and preprocessing utilities for spectroscopy data.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from scipy.interpolate import interp1d
from sklearn.cluster import AgglomerativeClustering
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

NMR_COLUMNS = {"h_nmr_spectra", "c_nmr_spectra"}
MSMS_COLUMNS = {"msms_positive_40ev", "msms_negative_40ev"}


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


def make_msms_spectrum(spectrum, max_mz: int = 10000) -> np.ndarray:
    """Convert sparse MS/MS peaks to a dense binned spectrum."""
    dense = np.zeros(max_mz, dtype=np.float32)

    if spectrum is None:
        return dense

    for peak in spectrum:
        if peak is None or len(peak) < 2:
            continue
        mz, intensity = peak[0], peak[1]
        if mz is None or intensity is None:
            continue

        mz_idx = int(np.round(float(mz)))
        if 0 <= mz_idx < max_mz:
            dense[mz_idx] = max(dense[mz_idx], float(intensity))

    return dense


def apply_snv_normalization(spectrum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply Standard Normal Variate normalization to one spectrum."""
    mean = np.mean(spectrum)
    std = np.std(spectrum)

    if std < eps:
        return spectrum - mean

    return (spectrum - mean) / std


def apply_quantile_normalization(spectra: np.ndarray) -> np.ndarray:
    """Apply quantile normalization across spectra (sample-wise rows).

    Standard quantile normalization - use apply_grouped_quantile_normalization
    for NMR data with Tanimoto-based grouping.
    """
    if spectra.ndim != 2:
        raise ValueError("Quantile normalization expects a 2D array (n_samples, n_features).")

    sorted_values = np.sort(spectra, axis=1)
    mean_ranks = np.mean(sorted_values, axis=0)

    ranks = np.argsort(np.argsort(spectra, axis=1), axis=1)
    normalized = mean_ranks[ranks]
    return normalized.astype(np.float32)


def compute_fingerprints(smiles_list: list[str]) -> list:
    """Compute Morgan fingerprints for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of RDKit fingerprint objects (None for invalid SMILES)
    """
    RDLogger.DisableLog("rdApp.*")
    fingerprints: list[Any] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles.strip().replace(" ", ""))
        if mol is None:
            fingerprints.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append(fp)
    return fingerprints


def compute_tanimoto_distance_matrix(fingerprints: list) -> np.ndarray:
    """Compute pairwise Tanimoto distance matrix from fingerprints.

    Args:
        fingerprints: List of RDKit fingerprint objects

    Returns:
        Distance matrix (1 - Tanimoto similarity)
    """
    n = len(fingerprints)
    dist_matrix = np.ones((n, n), dtype=np.float32)

    for i in range(n):
        if fingerprints[i] is None:
            continue
        for j in range(i + 1, n):
            if fingerprints[j] is None:
                continue
            sim = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            dist = 1.0 - sim
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
        dist_matrix[i, i] = 0.0

    return dist_matrix


def cluster_by_tanimoto(
    smiles_list: list[str],
    n_clusters: int | None = None,
    distance_threshold: float = 0.3,
) -> np.ndarray:
    """Cluster molecules by Tanimoto similarity using hierarchical clustering.

    Args:
        smiles_list: List of SMILES strings
        n_clusters: Number of clusters (if None, uses distance_threshold)
        distance_threshold: Tanimoto distance threshold for clustering

    Returns:
        Array of cluster labels
    """
    fingerprints = compute_fingerprints(smiles_list)
    dist_matrix = compute_tanimoto_distance_matrix(fingerprints)

    # Use agglomerative clustering with precomputed distance
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold if n_clusters is None else None,
    )
    labels = clustering.fit_predict(dist_matrix)
    return labels


def apply_grouped_quantile_normalization(
    spectra: np.ndarray,
    smiles_list: list[str],
    distance_threshold: float = 0.3,
    min_group_size: int = 5,
) -> np.ndarray:
    """Apply quantile normalization within Tanimoto-similarity groups.

    NMR spectra from similar molecules (by Tanimoto distance) should have
    similar signal patterns. This normalizes within groups of similar molecules
    to correct for dilution differences while preserving chemical differences.

    Args:
        spectra: NMR spectra array (n_samples, n_features)
        smiles_list: List of SMILES strings corresponding to spectra
        distance_threshold: Tanimoto distance threshold for grouping
        min_group_size: Minimum samples for group-wise normalization;
                        smaller groups use global normalization

    Returns:
        Normalized spectra array
    """
    if spectra.ndim != 2:
        raise ValueError("Grouped quantile normalization expects 2D array.")

    if len(smiles_list) != spectra.shape[0]:
        raise ValueError("SMILES list length must match number of spectra.")

    # Cluster molecules by Tanimoto similarity
    print(f"Clustering {len(smiles_list)} molecules by Tanimoto similarity...")
    cluster_labels = cluster_by_tanimoto(smiles_list, distance_threshold=distance_threshold)
    n_clusters = len(set(cluster_labels))
    print(f"Found {n_clusters} clusters")

    # Apply quantile normalization within each cluster
    normalized = np.zeros_like(spectra)
    cluster_sizes = []

    for cluster_id in set(cluster_labels):
        mask = cluster_labels == cluster_id
        cluster_size = mask.sum()
        cluster_sizes.append(cluster_size)

        if cluster_size >= min_group_size:
            # Apply quantile normalization to this cluster
            cluster_spectra = spectra[mask]
            normalized[mask] = apply_quantile_normalization(cluster_spectra)
        else:
            # Small cluster: just copy (will be normalized globally at end)
            normalized[mask] = spectra[mask]

    # For very small clusters, apply global normalization
    small_cluster_mask = np.array(
        [
            cluster_labels[i]
            in {c for c in set(cluster_labels) if (cluster_labels == c).sum() < min_group_size}
            for i in range(len(cluster_labels))
        ]
    )

    if small_cluster_mask.sum() > 0:
        normalized[small_cluster_mask] = apply_quantile_normalization(spectra[small_cluster_mask])

    print(
        f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
        f"median={np.median(cluster_sizes):.0f}"
    )

    return normalized.astype(np.float32)


def apply_pqn_normalization(spectra: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply Probabilistic Quotient Normalization across MS/MS spectra.

    For sparse MS/MS data, the median quotient is computed only over bins
    where both sample and reference have non-zero signal to avoid bias
    from empty bins.
    """
    if spectra.ndim != 2:
        raise ValueError("PQN expects a 2D array (n_samples, n_features).")

    spectra = np.asarray(spectra, dtype=np.float32)

    # Step 1: normalize by total ion current (row sum)
    row_sums = spectra.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums > eps, row_sums, 1.0)
    tic_normalized = spectra / safe_row_sums

    # Step 2: reference spectrum as median profile
    reference = np.median(tic_normalized, axis=0)

    # Step 3: sample-wise dilution factors via median quotients
    # Only compute quotients where both sample and reference have signal
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = (tic_normalized > eps) & (reference > eps)
        quotients = np.where(mask, tic_normalized / reference, np.nan)
        dilution_factors = np.nanmedian(quotients, axis=1)
    # Fallback for samples with no valid quotients
    dilution_factors = np.where(np.isnan(dilution_factors), 1.0, dilution_factors)
    safe_dilution = np.where(dilution_factors > eps, dilution_factors, 1.0)

    return (tic_normalized / safe_dilution[:, None]).astype(np.float32)


def apply_grouped_pqn_normalization(
    spectra: np.ndarray,
    smiles_list: list[str],
    distance_threshold: float = 0.3,
    min_group_size: int = 5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply PQN normalization within Tanimoto-similarity groups.

    MS/MS spectra from similar molecules (by Tanimoto distance) should have
    similar fragmentation patterns. This normalizes within groups of similar
    molecules to correct for intensity scaling differences while preserving
    chemical differences.

    Args:
        spectra: MS/MS spectra array (n_samples, n_features)
        smiles_list: List of SMILES strings corresponding to spectra
        distance_threshold: Tanimoto distance threshold for grouping
        min_group_size: Minimum samples for group-wise normalization;
                        smaller groups use global normalization
        eps: Small value for numerical stability

    Returns:
        Normalized spectra array
    """
    if spectra.ndim != 2:
        raise ValueError("Grouped PQN normalization expects 2D array.")

    if len(smiles_list) != spectra.shape[0]:
        raise ValueError("SMILES list length must match number of spectra.")

    # Cluster molecules by Tanimoto similarity
    print(f"Clustering {len(smiles_list)} molecules by Tanimoto similarity...")
    cluster_labels = cluster_by_tanimoto(smiles_list, distance_threshold=distance_threshold)
    n_clusters = len(set(cluster_labels))
    print(f"Found {n_clusters} clusters")

    # Apply PQN normalization within each cluster
    normalized = np.zeros_like(spectra)
    cluster_sizes = []

    for cluster_id in set(cluster_labels):
        mask = cluster_labels == cluster_id
        cluster_size = mask.sum()
        cluster_sizes.append(cluster_size)

        if cluster_size >= min_group_size:
            # Apply PQN normalization to this cluster
            cluster_spectra = spectra[mask]
            normalized[mask] = apply_pqn_normalization(cluster_spectra, eps=eps)
        else:
            # Small cluster: just copy (will be normalized globally at end)
            normalized[mask] = spectra[mask]

    # For very small clusters, apply global PQN normalization
    small_cluster_mask = np.array(
        [
            cluster_labels[i]
            in {c for c in set(cluster_labels) if (cluster_labels == c).sum() < min_group_size}
            for i in range(len(cluster_labels))
        ]
    )

    if small_cluster_mask.sum() > 0:
        normalized[small_cluster_mask] = apply_pqn_normalization(
            spectra[small_cluster_mask], eps=eps
        )

    print(
        f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
        f"median={np.median(cluster_sizes):.0f}"
    )

    return normalized.astype(np.float32)


def _load_spectra_data(
    data_dir: Path,
    spectra_column: str,
    target_length: int,
    max_files: int | None,
    sample_transform: Callable | None = None,
    batch_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    return_smiles: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]:
    """Shared data loading pipeline for all spectroscopy modalities.

    Args:
        data_dir: Directory containing parquet files
        spectra_column: Column name for spectra data
        target_length: Target length for interpolation
        max_files: Maximum number of files to load
        sample_transform: Transform applied to each spectrum
        batch_transform: Transform applied to batch of spectra (per file)
        return_smiles: If True, also return list of SMILES strings

    Returns:
        (X, y) or (X, y, smiles_list) if return_smiles=True
    """
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("*.parquet"))

    if max_files:
        parquet_files = parquet_files[:max_files]

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    all_spectra: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_smiles: list[str] = []

    for i, parquet_file in enumerate(parquet_files):
        df = pd.read_parquet(parquet_file, columns=[spectra_column, "smiles"])
        df["func_groups"] = df["smiles"].map(get_functional_groups)

        df = df[df["func_groups"].notna()]
        df = df[df[spectra_column].notna()]

        processed_spectra: list[np.ndarray] = []
        labels: list[Any] = []
        smiles_batch: list[str] = []

        for raw_spec, label, smiles in zip(
            df[spectra_column].values, df["func_groups"].values, df["smiles"].values
        ):
            spec = sample_transform(raw_spec) if sample_transform else raw_spec
            if spec is None:
                continue

            spec = np.asarray(spec, dtype=np.float32)
            if spec.ndim != 1:
                continue

            spec = interpolate_spectrum(spec, target_length)
            processed_spectra.append(spec)
            labels.append(label)
            smiles_batch.append(smiles)

        if not processed_spectra:
            continue

        spectra = np.stack(processed_spectra).astype(np.float32)
        if batch_transform is not None:
            spectra = batch_transform(spectra)

        label_array = np.stack(labels).astype(np.float32)

        all_spectra.append(spectra)
        all_labels.append(label_array)
        all_smiles.extend(smiles_batch)

        if (i + 1) % 10 == 0:
            print(f"Loaded {i + 1}/{len(parquet_files)} files")

    if not all_spectra:
        raise ValueError(
            f"No valid spectra loaded for column '{spectra_column}' from directory {data_dir}"
        )

    X = np.vstack(all_spectra)
    y = np.vstack(all_labels)

    if return_smiles:
        return X, y, all_smiles
    return X, y


def load_ir_data(
    data_dir: Path,
    target_length: int = 1800,
    max_files: int | None = None,
    apply_snv: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load IR spectra with optional SNV normalization."""
    print(f"Loading IR spectra from {data_dir}")

    batch_transform = None
    if apply_snv:
        batch_transform = lambda arr: np.stack([apply_snv_normalization(spec) for spec in arr])

    X, y = _load_spectra_data(
        data_dir=data_dir,
        spectra_column="ir_spectra",
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


def load_nmr_data(
    data_dir: Path,
    input_column: str,
    target_length: int = 1800,
    max_files: int | None = None,
    apply_quantile: bool = True,
    use_grouped_normalization: bool = True,
    tanimoto_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Load NMR spectra with optional Tanimoto-grouped quantile normalization.

    NMR spectra from similar molecules should have similar signal patterns.
    Grouped quantile normalization normalizes within clusters of structurally
    similar molecules (by Tanimoto similarity) to correct for dilution
    differences while preserving chemical differences.

    Args:
        data_dir: Directory with parquet files
        input_column: NMR column name ('h_nmr_spectra' or 'c_nmr_spectra')
        target_length: Target spectrum length after interpolation
        max_files: Maximum number of files to load
        apply_quantile: Whether to apply quantile normalization
        use_grouped_normalization: If True, group by Tanimoto similarity first
        tanimoto_threshold: Tanimoto distance threshold for grouping

    Returns:
        (X, y) tuple of spectra and labels
    """
    if input_column not in NMR_COLUMNS:
        raise ValueError(f"NMR column must be one of {sorted(NMR_COLUMNS)}, got '{input_column}'")

    print(f"Loading NMR spectra from {data_dir} (column: {input_column})")

    if apply_quantile and use_grouped_normalization:
        # Load without normalization, but with SMILES for clustering
        X, y, smiles_list = _load_spectra_data(
            data_dir=data_dir,
            spectra_column=input_column,
            target_length=target_length,
            max_files=max_files,
            batch_transform=None,
            return_smiles=True,
        )

        # Apply grouped quantile normalization
        print("Applying Tanimoto-grouped quantile normalization...")
        X = apply_grouped_quantile_normalization(
            X, smiles_list, distance_threshold=tanimoto_threshold
        )
        norm_method = "Tanimoto-grouped quantile"
    elif apply_quantile:
        # Standard (non-grouped) quantile normalization
        X, y = _load_spectra_data(
            data_dir=data_dir,
            spectra_column=input_column,
            target_length=target_length,
            max_files=max_files,
            batch_transform=apply_quantile_normalization,
        )
        norm_method = "Standard quantile"
    else:
        # No normalization
        X, y = _load_spectra_data(
            data_dir=data_dir,
            spectra_column=input_column,
            target_length=target_length,
            max_files=max_files,
        )
        norm_method = None

    print(f"Total samples loaded: {len(X)}")
    print(f"Spectra shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {y.sum(axis=0)}")
    if norm_method:
        print(f"{norm_method} normalization applied")

    return X, y


def load_msms_data(
    data_dir: Path,
    input_column: str,
    target_length: int = 10000,
    max_files: int | None = None,
    apply_pqn: bool = True,
    use_grouped_normalization: bool = True,
    tanimoto_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MS/MS spectra with Tanimoto-grouped PQN normalization.

    MS/MS spectra from similar molecules should have similar fragmentation
    patterns. Grouped PQN normalization normalizes within clusters of
    structurally similar molecules (by Tanimoto similarity) to correct for
    intensity scaling differences while preserving chemical differences.

    Args:
        data_dir: Directory with parquet files
        input_column: MS/MS column name ('msms_positive_40ev' or 'msms_negative_40ev')
        target_length: Target spectrum length (m/z bins)
        max_files: Maximum number of files to load
        apply_pqn: Whether to apply PQN normalization
        use_grouped_normalization: If True, group by Tanimoto similarity first
        tanimoto_threshold: Tanimoto distance threshold for grouping

    Returns:
        (X, y) tuple of spectra and labels
    """
    if input_column not in MSMS_COLUMNS:
        raise ValueError(
            f"MS/MS column must be one of {sorted(MSMS_COLUMNS)}, got '{input_column}'"
        )

    print(f"Loading MS/MS spectra from {data_dir} (column: {input_column})")

    if apply_pqn and use_grouped_normalization:
        # Load without normalization, but with SMILES for clustering
        X, y, smiles_list = _load_spectra_data(
            data_dir=data_dir,
            spectra_column=input_column,
            target_length=target_length,
            max_files=max_files,
            sample_transform=make_msms_spectrum,
            batch_transform=None,
            return_smiles=True,
        )

        # Apply grouped PQN normalization
        print("Applying Tanimoto-grouped PQN normalization...")
        X = apply_grouped_pqn_normalization(X, smiles_list, distance_threshold=tanimoto_threshold)
        norm_method = "Tanimoto-grouped PQN"
    elif apply_pqn:
        # Standard (non-grouped) PQN normalization
        X, y = _load_spectra_data(
            data_dir=data_dir,
            spectra_column=input_column,
            target_length=target_length,
            max_files=max_files,
            sample_transform=make_msms_spectrum,
            batch_transform=apply_pqn_normalization,
        )
        norm_method = "Standard PQN"
    else:
        # No normalization
        X, y = _load_spectra_data(
            data_dir=data_dir,
            spectra_column=input_column,
            target_length=target_length,
            max_files=max_files,
            sample_transform=make_msms_spectrum,
        )
        norm_method = None

    print(f"Total samples loaded: {len(X)}")
    print(f"Spectra shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {y.sum(axis=0)}")
    if norm_method:
        print(f"{norm_method} normalization applied")

    return X, y


def load_spectra_data(
    data_dir: Path,
    modality: str = "ir",
    input_column: str | None = None,
    target_length: int = 1800,
    max_files: int | None = None,
    normalization_method: str | None = None,
    apply_snv: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch data loading by modality.

    Args:
        data_dir: Directory with parquet files
        modality: One of "ir", "nmr", "msms"
        input_column: Column name for nmr/msms
        target_length: Output spectrum length after interpolation
        max_files: Optional file cap
        normalization_method: One of "snv", "quantile", "pqn", "none"
        apply_snv: Legacy flag for IR compatibility
    """
    modality = modality.lower()
    norm = normalization_method.lower() if normalization_method else None

    if modality == "ir":
        use_snv = apply_snv if norm is None else norm == "snv"
        if norm not in {None, "snv", "none"}:
            raise ValueError("IR normalization must be 'snv' or 'none'.")
        return load_ir_data(
            data_dir=data_dir,
            target_length=target_length,
            max_files=max_files,
            apply_snv=use_snv,
        )

    if modality == "nmr":
        if input_column is None:
            input_column = "h_nmr_spectra"
        if norm not in {None, "quantile", "none"}:
            raise ValueError("NMR normalization must be 'quantile' or 'none'.")
        apply_quantile = True if norm is None else norm == "quantile"
        return load_nmr_data(
            data_dir=data_dir,
            input_column=input_column,
            target_length=target_length,
            max_files=max_files,
            apply_quantile=apply_quantile,
        )

    if modality == "msms":
        if input_column is None:
            input_column = "msms_positive_40ev"
        if norm not in {None, "pqn", "none"}:
            raise ValueError("MS/MS normalization must be 'pqn' or 'none'.")
        apply_pqn = True if norm is None else norm == "pqn"
        return load_msms_data(
            data_dir=data_dir,
            input_column=input_column,
            target_length=target_length,
            max_files=max_files,
            apply_pqn=apply_pqn,
        )

    raise ValueError("modality must be one of: 'ir', 'nmr', 'msms'")


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
