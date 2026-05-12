"""Detailed error analysis for MPSFunctionalGroupClassifier models.

This script mirrors the TTN/CNN error-analysis scripts, adapted to your dual-MPS
IR classifier. It loads the saved `.pt` checkpoint, rebuilds the architecture
from the checkpoint config, reloads/preprocesses the IR data, recreates the same
train/val/test split, runs inference, and writes the same analysis artifacts.

Outputs:
  - per_class_metrics.csv
  - per_class_diagnosis.csv
  - global_label_distribution.csv
  - split_label_distribution.csv
  - switch_matrix_false_negative_vs_false_positive.csv
  - top_confusions.csv
  - threshold_sensitivity.csv
  - sample_errors.csv
  - probability_summary.csv
  - analysis_summary.json
  - plots/*.png

Example:

python scripts/analyse_mps_errors.py \
  --data-dir data/raw \
  --checkpoint src/spectroscopy_qml/ir/mps_classifier/models_per_label/mps_model_best.pt \
  --output-dir src/spectroscopy_qml/ir/mps_classifier/results_per_label/error_analysis \
  --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Optional project import path setup
# -----------------------------------------------------------------------------


# Add src directory to path so spectroscopy_qml can be imported
src_dir = Path(__file__).parents[3]
sys.path.insert(0, str(src_dir))


# -----------------------------------------------------------------------------
# Functional groups, identical order to your MPS data loader.
# -----------------------------------------------------------------------------

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

LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())


# -----------------------------------------------------------------------------
# Model architecture, copied from your MPS classifier.
# -----------------------------------------------------------------------------

class LocalFeatureMap(nn.Module):
    def __init__(self, site_dim: int = 50, physical_dim: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.site_dim = site_dim
        self.physical_dim = physical_dim
        self.hidden_dim = max(site_dim, physical_dim * 2)
        self.layer_norm = nn.LayerNorm(site_dim)
        self.fc1 = nn.Linear(site_dim, self.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, physical_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.tanh(x)


class MPSEncoder(nn.Module):
    def __init__(
        self,
        num_sites: int = 36,
        physical_dim: int = 8,
        bond_dim: int = 16,
        output_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_sites = num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.output_dim = output_dim
        self.eps = eps

        self.forward_cores = nn.ParameterList()
        first_core = torch.empty(1, physical_dim, bond_dim)
        nn.init.xavier_uniform_(first_core)
        self.forward_cores.append(nn.Parameter(first_core))
        for _ in range(num_sites - 1):
            core = torch.empty(bond_dim, physical_dim, bond_dim)
            nn.init.xavier_uniform_(core)
            self.forward_cores.append(nn.Parameter(core))

        self.backward_cores = nn.ParameterList()
        first_core_back = torch.empty(1, physical_dim, bond_dim)
        nn.init.xavier_uniform_(first_core_back)
        self.backward_cores.append(nn.Parameter(first_core_back))
        for _ in range(num_sites - 1):
            core = torch.empty(bond_dim, physical_dim, bond_dim)
            nn.init.xavier_uniform_(core)
            self.backward_cores.append(nn.Parameter(core))

        total_features = 2 * num_sites * bond_dim
        self.output_norm = nn.LayerNorm(total_features)
        self.output_proj = nn.Linear(total_features, output_dim)

    def _contract_forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        states = []
        state = torch.einsum("bp,ipj->bj", features[:, 0], self.forward_cores[0])
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
        states.append(state)
        for i in range(1, self.num_sites):
            state = torch.einsum("bi,ipj,bp->bj", state, self.forward_cores[i], features[:, i])
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
            states.append(state)
        return states

    def _contract_backward(self, features: torch.Tensor) -> list[torch.Tensor]:
        states = []
        features_rev = torch.flip(features, dims=[1])
        state = torch.einsum("bp,ipj->bj", features_rev[:, 0], self.backward_cores[0])
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
        states.append(state)
        for i in range(1, self.num_sites):
            state = torch.einsum("bi,ipj,bp->bj", state, self.backward_cores[i], features_rev[:, i])
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
            states.append(state)
        return states

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        forward_states = self._contract_forward(features)
        backward_states = self._contract_backward(features)
        combined = torch.cat(forward_states + backward_states, dim=1)
        embedding = self.output_norm(combined)
        return self.output_proj(embedding)


class MPSFunctionalGroupClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1800,
        num_sites: int = 36,
        physical_dim: int = 8,
        bond_dim: int = 16,
        num_classes: int = 37,
        dropout_rate: float = 0.2,
        classifier_head: str = "mps",
        num_sites_2: int = 1800,
        physical_dim_2: int = 8,
        bond_dim_2: int = 16,
    ):
        super().__init__()
        assert input_dim % num_sites == 0, f"input_dim ({input_dim}) must be divisible by num_sites ({num_sites})"
        assert input_dim % num_sites_2 == 0, f"input_dim ({input_dim}) must be divisible by num_sites_2 ({num_sites_2})"
        assert classifier_head in ("cnn", "mps")

        self.input_dim = input_dim
        self.num_sites = num_sites
        self.site_dim = input_dim // num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.classifier_head = classifier_head
        self.num_sites_2 = num_sites_2
        self.site_dim_2 = input_dim // num_sites_2
        self.physical_dim_2 = physical_dim_2
        self.bond_dim_2 = bond_dim_2

        self.feature_map = LocalFeatureMap(self.site_dim, physical_dim, dropout_rate)
        self.feature_map_2 = LocalFeatureMap(self.site_dim_2, physical_dim_2, dropout_rate)

        intermediate_dim = 128
        if classifier_head == "cnn":
            self.mps_encoder = MPSEncoder(num_sites, physical_dim, bond_dim, output_dim=256)
            self.mps_encoder_2 = MPSEncoder(num_sites_2, physical_dim_2, bond_dim_2, output_dim=256)
            cnn_in_channels = 2 * bond_dim + self.site_dim
            self.conv1 = nn.Conv1d(cnn_in_channels, 31, kernel_size=11, stride=1, padding="same")
            self.bn1 = nn.BatchNorm1d(31)
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv1d(31, 62, kernel_size=11, stride=1, padding="same")
            self.bn2 = nn.BatchNorm1d(62)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            conv_out_length = num_sites // 2 // 2
            flat_size = 62 * conv_out_length
            self.fc1 = nn.Linear(flat_size + 256, 4927)
            self.fc2 = nn.Linear(4927, 2785)
            self.fc3 = nn.Linear(2785, 1574)
            self.fc_out = nn.Linear(1574, num_classes)
            self.cnn_dropout = nn.Dropout(0.48599073736368)
            self.relu = nn.ReLU()
        else:
            self.mps_encoder = MPSEncoder(num_sites, physical_dim, bond_dim, output_dim=intermediate_dim)
            self.mps_encoder_2 = MPSEncoder(num_sites_2, physical_dim_2, bond_dim_2, output_dim=intermediate_dim)
            combined_dim = 2 * intermediate_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Linear(combined_dim, 256),
                nn.GELU(),
                nn.Dropout(0.49),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_sites = x.view(batch_size, self.num_sites, self.site_dim)
        x_flat = x_sites.view(batch_size * self.num_sites, self.site_dim)
        features_flat = self.feature_map(x_flat)
        features = features_flat.view(batch_size, self.num_sites, self.physical_dim)

        x_sites_2 = x.view(batch_size, self.num_sites_2, self.site_dim_2)
        x_flat_2 = x_sites_2.view(batch_size * self.num_sites_2, self.site_dim_2)
        features_flat_2 = self.feature_map_2(x_flat_2)
        features_2 = features_flat_2.view(batch_size, self.num_sites_2, self.physical_dim_2)

        if self.classifier_head == "mps":
            amp_device = "cuda" if x.device.type == "cuda" else "cpu"
            with torch.amp.autocast(amp_device, enabled=False):
                emb1 = self.mps_encoder(features.float())
                emb2 = self.mps_encoder_2(features_2.float())
            return self.classifier(torch.cat([emb1, emb2], dim=1))

        amp_device = "cuda" if x.device.type == "cuda" else "cpu"
        with torch.amp.autocast(amp_device, enabled=False):
            features_f32 = features.float()
            forward_states = self.mps_encoder._contract_forward(features_f32)
            backward_states = self.mps_encoder._contract_backward(features_f32)

        fw = torch.stack(forward_states, dim=1)
        bw = torch.stack(backward_states, dim=1)
        per_site_mps = torch.cat([fw, bw], dim=2)
        per_site = torch.cat([per_site_mps, x_sites], dim=2)
        x = per_site.transpose(1, 2)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.view(batch_size, -1)

        with torch.amp.autocast(amp_device, enabled=False):
            emb2 = self.mps_encoder_2(features_2.float())
        x = torch.cat([x, emb2], dim=1)
        x = self.cnn_dropout(self.relu(self.fc1(x)))
        x = self.cnn_dropout(self.relu(self.fc2(x)))
        x = self.cnn_dropout(self.relu(self.fc3(x)))
        return self.fc_out(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Data loading/preprocessing compatible with your MPS data loader.
# -----------------------------------------------------------------------------

def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) is Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1


def get_functional_groups(smiles: str) -> list[int] | None:
    RDLogger.DisableLog("rdApp.*")
    smiles = smiles.strip().replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [match_group(mol, smarts) for smarts in FUNCTIONAL_GROUPS.values()]


def interpolate_spectrum(spectrum: np.ndarray, target_length: int = 1800) -> np.ndarray:
    if len(spectrum) == target_length:
        return np.asarray(spectrum, dtype=np.float32)
    old_x = np.arange(len(spectrum))
    new_x = np.linspace(0, len(spectrum) - 1, target_length)
    interp_func = interp1d(old_x, spectrum, kind="linear")
    return interp_func(new_x).astype(np.float32)


def apply_snv_normalization(spectrum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    if std < eps:
        return spectrum - mean
    return (spectrum - mean) / std


def apply_savgol_smoothing(spectrum: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    if len(spectrum) < window_length:
        return spectrum
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2 + ((polyorder + 2) % 2 == 0)
    return savgol_filter(spectrum, window_length=window_length, polyorder=polyorder)


def load_ir_data_with_ids(
    data_dir: Path,
    target_length: int,
    max_files: int | None,
    apply_snv: bool,
    apply_savgol: bool,
    savgol_window_length: int,
    savgol_polyorder: int,
    cache_path: Path | None,
    overwrite_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        payload = np.load(cache_path, allow_pickle=False)
        return payload["X"], payload["y"], payload["row_ids"]

    parquet_files = sorted(Path(data_dir).glob("*.parquet"))
    if max_files is not None:
        parquet_files = parquet_files[:max_files]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    all_spectra: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_row_ids: list[np.ndarray] = []
    row_offset = 0

    for i, parquet_file in enumerate(parquet_files, start=1):
        print(f"Loading {i}/{len(parquet_files)}: {parquet_file.name}")
        df = pd.read_parquet(parquet_file, columns=["ir_spectra", "smiles"])
        df["_row_id"] = np.arange(row_offset, row_offset + len(df), dtype=np.int64)
        row_offset += len(df)

        df["func_groups"] = df["smiles"].map(get_functional_groups)
        df = df[df["func_groups"].notna()]
        df = df[df["ir_spectra"].notna()]

        spectra = np.stack([interpolate_spectrum(spec, target_length) for spec in df["ir_spectra"].values])
        if apply_savgol:
            spectra = np.stack([
                apply_savgol_smoothing(spec, savgol_window_length, savgol_polyorder)
                for spec in spectra
            ])
        if apply_snv:
            spectra = np.stack([apply_snv_normalization(spec) for spec in spectra])

        labels = np.stack(df["func_groups"].values).astype(np.int32)
        all_spectra.append(spectra.astype(np.float32))
        all_labels.append(labels)
        all_row_ids.append(df["_row_id"].to_numpy(dtype=np.int64))

    X = np.vstack(all_spectra).astype(np.float32)
    y = np.vstack(all_labels).astype(np.int32)
    row_ids = np.concatenate(all_row_ids).astype(np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            row_ids=row_ids,
            target_length=np.asarray(target_length),
            apply_snv=np.asarray(apply_snv),
            apply_savgol=np.asarray(apply_savgol),
            savgol_window_length=np.asarray(savgol_window_length),
            savgol_polyorder=np.asarray(savgol_polyorder),
        )
        print(f"Saved cache to {cache_path}")

    return X, y, row_ids


# -----------------------------------------------------------------------------
# Split/checkpoint helpers
# -----------------------------------------------------------------------------

def normalize_model_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    # Some checkpoints pickle dataclass-like objects from the project.
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with key 'model_state_dict'.")
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict[str, Any], cli_overrides: argparse.Namespace, device: torch.device) -> MPSFunctionalGroupClassifier:
    config = normalize_model_config(checkpoint.get("config"))
    defaults = {
        "input_dim": 1800,
        "num_sites": 5,
        "physical_dim": 450,
        "bond_dim": 128,
        "num_classes": 37,
        "dropout_rate": 0.49,
        "classifier_head": "mps",
        "num_sites_2": 10,
        "physical_dim_2": 450,
        "bond_dim_2": 128,
    }
    defaults.update(config)

    for key in defaults:
        value = getattr(cli_overrides, key, None)
        if value is not None:
            defaults[key] = value

    model = MPSFunctionalGroupClassifier(**defaults).to(device)
    model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    return model


def rebuild_splits(
    X: np.ndarray,
    y: np.ndarray,
    row_ids: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    train_ratio: float,
    random_seed: int,
    num_folds: int,
    best_fold: int | None,
) -> dict[str, dict[str, np.ndarray]]:
    X_trainval, X_test, y_trainval, y_test, rows_trainval, rows_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    splits: dict[str, dict[str, np.ndarray]] = {
        "trainval": {"X": X_trainval, "y": y_trainval, "row_ids": rows_trainval},
        "test": {"X": X_test, "y": y_test, "row_ids": rows_test},
    }

    indices = np.arange(len(X_trainval))
    if num_folds <= 1:
        val_fraction = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=random_seed,
            shuffle=True,
        )
        splits["train"] = {"X": X_trainval[train_idx], "y": y_trainval[train_idx], "row_ids": rows_trainval[train_idx]}
        splits["val"] = {"X": X_trainval[val_idx], "y": y_trainval[val_idx], "row_ids": rows_trainval[val_idx]}
        return splits

    fold_to_use = best_fold or 1
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_trainval, y_trainval), start=1):
        if fold_idx == fold_to_use:
            splits["train"] = {"X": X_trainval[train_idx], "y": y_trainval[train_idx], "row_ids": rows_trainval[train_idx]}
            splits["val"] = {"X": X_trainval[val_idx], "y": y_trainval[val_idx], "row_ids": rows_trainval[val_idx]}
            return splits
    raise ValueError(f"Could not reconstruct fold {fold_to_use} from {num_folds} folds.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device, use_amp: bool) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    parts = []
    model.eval()
    for (spectra,) in loader:
        spectra = spectra.to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(spectra)
        else:
            logits = model(spectra)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        parts.append(probs)
    probs_np = np.concatenate(parts, axis=0)
    probs_np = np.nan_to_num(probs_np, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs_np, 0.0, 1.0).astype(np.float32)


def tune_thresholds(y_true: np.ndarray, y_probs: np.ndarray, grid_step: float = 0.05) -> np.ndarray:
    n_classes = y_true.shape[1]
    thresholds = np.zeros(n_classes, dtype=np.float32)
    grid = np.arange(0.1, 0.9 + 1e-12, grid_step)
    for i in range(n_classes):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in grid:
            y_pred_i = (y_probs[:, i] >= threshold).astype(int)
            score = f1_score(y_true[:, i], y_pred_i, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[i] = best_threshold
    return thresholds


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(path: Path, matrix: np.ndarray, row_names: list[str], col_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["false_negative_true_label__vs__false_positive_predicted_label"] + col_names)
        for name, row in zip(row_names, matrix, strict=False):
            writer.writerow([name] + [int(x) for x in row])


def class_metrics(y_true, y_pred, y_prob, thresholds, label_names, train_labels=None):
    rows = []
    n_samples, n_classes = y_true.shape
    train_prevalence = train_labels.mean(axis=0) if train_labels is not None else None
    for c in range(n_classes):
        yt = y_true[:, c].astype(int)
        yp = y_pred[:, c].astype(int)
        pr = y_prob[:, c]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        support = int(yt.sum())
        pred_pos = int(yp.sum())
        pos_probs = pr[yt == 1]
        neg_probs = pr[yt == 0]
        rows.append({
            "class_index": c,
            "label": label_names[c],
            "support_true": support,
            "support_true_pct": support / n_samples,
            "predicted_positive": pred_pos,
            "predicted_positive_pct": pred_pos / n_samples,
            "prediction_minus_truth_pct": (pred_pos - support) / n_samples,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            "threshold": float(thresholds[c]),
            "mean_prob_all": float(pr.mean()),
            "mean_prob_true_positive_samples": float(pos_probs.mean()) if pos_probs.size else None,
            "mean_prob_true_negative_samples": float(neg_probs.mean()) if neg_probs.size else None,
            "prob_gap_pos_minus_neg": float(pos_probs.mean() - neg_probs.mean()) if pos_probs.size and neg_probs.size else None,
            "roc_auc": safe_float(roc_auc_score(yt, pr)) if len(np.unique(yt)) == 2 else None,
            "average_precision": safe_float(average_precision_score(yt, pr)) if len(np.unique(yt)) == 2 else None,
            "train_prevalence": float(train_prevalence[c]) if train_prevalence is not None else None,
            "eval_prevalence": support / n_samples,
            "eval_minus_train_prevalence": float((support / n_samples) - train_prevalence[c]) if train_prevalence is not None else None,
        })
    return rows


def global_distribution_rows(y_true, y_pred, label_names):
    n = y_true.shape[0]
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    return [
        {
            "class_index": i,
            "label": name,
            "true_count": int(true_counts[i]),
            "true_pct": float(true_counts[i] / n),
            "pred_count": int(pred_counts[i]),
            "pred_pct": float(pred_counts[i] / n),
            "pred_minus_true_count": int(pred_counts[i] - true_counts[i]),
            "pred_minus_true_pct": float((pred_counts[i] - true_counts[i]) / n),
            "over_prediction_ratio": float(pred_counts[i] / true_counts[i]) if true_counts[i] > 0 else None,
        }
        for i, name in enumerate(label_names)
    ]


def split_distribution_rows(splits, label_names):
    rows = []
    for split_name, payload in splits.items():
        labels = payload["y"]
        n = labels.shape[0]
        counts = labels.sum(axis=0)
        for c, name in enumerate(label_names):
            rows.append({
                "split": split_name,
                "class_index": c,
                "label": name,
                "count": int(counts[c]),
                "pct": float(counts[c] / n) if n else 0.0,
                "num_samples": int(n),
            })
    return rows


def switch_matrix(y_true, y_pred):
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    return fn.astype(np.int64).T @ fp.astype(np.int64)


def top_confusion_rows(matrix, label_names, limit):
    rows = []
    for true_i in range(matrix.shape[0]):
        for pred_j in range(matrix.shape[1]):
            if true_i == pred_j:
                continue
            count = int(matrix[true_i, pred_j])
            if count > 0:
                rows.append({
                    "missed_true_label_index": true_i,
                    "missed_true_label": label_names[true_i],
                    "wrong_predicted_label_index": pred_j,
                    "wrong_predicted_label": label_names[pred_j],
                    "count": count,
                })
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:limit]


def probability_summary_rows(y_true, y_prob, thresholds, label_names):
    rows = []
    for c, name in enumerate(label_names):
        probs = y_prob[:, c]
        true_probs = probs[y_true[:, c] == 1]
        false_probs = probs[y_true[:, c] == 0]
        rows.append({
            "class_index": c,
            "label": name,
            "threshold": float(thresholds[c]),
            "prob_p01": float(np.quantile(probs, 0.01)),
            "prob_p05": float(np.quantile(probs, 0.05)),
            "prob_p25": float(np.quantile(probs, 0.25)),
            "prob_p50": float(np.quantile(probs, 0.50)),
            "prob_p75": float(np.quantile(probs, 0.75)),
            "prob_p95": float(np.quantile(probs, 0.95)),
            "prob_p99": float(np.quantile(probs, 0.99)),
            "true_sample_prob_p50": float(np.quantile(true_probs, 0.50)) if true_probs.size else None,
            "negative_sample_prob_p50": float(np.quantile(false_probs, 0.50)) if false_probs.size else None,
        })
    return rows


def threshold_sensitivity_rows(y_true, y_prob, label_names, grid_step):
    rows = []
    grid = np.arange(0.05, 0.95 + 1e-12, grid_step)
    for c, name in enumerate(label_names):
        yt = y_true[:, c]
        if yt.sum() == 0:
            continue
        for threshold in grid:
            yp = (y_prob[:, c] >= threshold).astype(int)
            rows.append({
                "class_index": c,
                "label": name,
                "threshold": float(threshold),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
            })
    return rows


def diagnosis_rows(per_class, switch, label_names):
    rows = []
    f1_values = np.asarray([row["f1"] for row in per_class], dtype=float)
    support_values = np.asarray([row["support_true"] for row in per_class], dtype=float)
    median_f1 = float(np.median(f1_values))
    median_support = float(np.median(support_values))
    for row in per_class:
        c = int(row["class_index"])
        reasons = []
        if row["support_true"] < median_support:
            reasons.append("low support / rare functional group")
        if row["recall"] < 0.5 and row["false_negative_rate"] > row["false_positive_rate"]:
            reasons.append("mostly under-detected: many false negatives")
        if row["precision"] < 0.5 and row["false_positive_rate"] > 0:
            reasons.append("over-predicted: many false positives")
        if row["prob_gap_pos_minus_neg"] is not None and row["prob_gap_pos_minus_neg"] < 0.15:
            reasons.append("weak probability separation between positive and negative samples")
        if row["threshold"] > 0.65 and row["recall"] < 0.6:
            reasons.append("high threshold may suppress recall")
        if row["threshold"] < 0.25 and row["precision"] < 0.6:
            reasons.append("low threshold may inflate false positives")
        if row["eval_minus_train_prevalence"] is not None and abs(row["eval_minus_train_prevalence"]) > 0.02:
            reasons.append("eval prevalence differs from train prevalence")

        top_switched_to = []
        if switch[c].sum() > 0:
            top_indices = np.argsort(-switch[c])[:3]
            top_switched_to = [
                f"{label_names[j]} ({int(switch[c, j])})" for j in top_indices if switch[c, j] > 0 and j != c
            ]
            if top_switched_to:
                reasons.append("often missed while another label is predicted")
        if row["f1"] >= max(0.75, median_f1):
            reasons.append("strong class: good precision/recall balance")
        if not reasons:
            reasons.append("no single dominant error pattern; inspect sample_errors and probability histograms")
        rows.append({
            "class_index": c,
            "label": row["label"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "support_true": row["support_true"],
            "predicted_positive": row["predicted_positive"],
            "main_diagnosis": "; ".join(reasons),
            "top_wrong_predicted_when_this_was_missed": " | ".join(top_switched_to),
        })
    rows.sort(key=lambda row: row["f1"])
    return rows


def sample_error_rows(y_true, y_pred, y_prob, row_ids, label_names, max_samples):
    rows = []
    for i in range(y_true.shape[0]):
        fn_idx = np.flatnonzero((y_true[i] == 1) & (y_pred[i] == 0))
        fp_idx = np.flatnonzero((y_true[i] == 0) & (y_pred[i] == 1))
        if fn_idx.size == 0 and fp_idx.size == 0:
            continue
        true_idx = np.flatnonzero(y_true[i] == 1)
        pred_idx = np.flatnonzero(y_pred[i] == 1)
        rows.append({
            "sample_index": int(row_ids[i]),
            "num_false_negatives": int(fn_idx.size),
            "num_false_positives": int(fp_idx.size),
            "error_count": int(fn_idx.size + fp_idx.size),
            "true_labels": " | ".join(label_names[j] for j in true_idx),
            "predicted_labels": " | ".join(label_names[j] for j in pred_idx),
            "false_negative_labels": " | ".join(label_names[j] for j in fn_idx),
            "false_positive_labels": " | ".join(label_names[j] for j in fp_idx),
            "false_negative_probs": " | ".join(f"{label_names[j]}={y_prob[i, j]:.4f}" for j in fn_idx),
            "false_positive_probs": " | ".join(f"{label_names[j]}={y_prob[i, j]:.4f}" for j in fp_idx),
        })
    rows.sort(key=lambda row: row["error_count"], reverse=True)
    return rows[:max_samples]


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_bar(path, names, values, title, ylabel):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(names) * 0.35), 6))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap(path, matrix, label_names, title, top_n=25):
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = matrix.sum(axis=1) + matrix.sum(axis=0)
    if matrix.shape[0] > top_n:
        selected = np.argsort(-totals)[:top_n]
        matrix = matrix[np.ix_(selected, selected)]
        label_names = [label_names[i] for i in selected]
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="count")
    plt.xticks(range(len(label_names)), label_names, rotation=90)
    plt.yticks(range(len(label_names)), label_names)
    plt.xlabel("wrong predicted label / false positive")
    plt.ylabel("missed true label / false negative")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_probability_histograms(out_dir, y_true, y_prob, label_names, worst_class_indices):
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in worst_class_indices:
        true_probs = y_prob[y_true[:, c] == 1, c]
        negative_probs = y_prob[y_true[:, c] == 0, c]
        plt.figure(figsize=(8, 5))
        if negative_probs.size:
            plt.hist(negative_probs, bins=40, alpha=0.65, label="true negative samples")
        if true_probs.size:
            plt.hist(true_probs, bins=40, alpha=0.65, label="true positive samples")
        plt.title(f"Probability separation: {label_names[c]}")
        plt.xlabel("predicted probability")
        plt.ylabel("sample count")
        plt.legend()
        plt.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label_names[c])
        plt.savefig(out_dir / f"prob_hist_{c:02d}_{safe_name}.png", dpi=180)
        plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detailed error analysis for MPSFunctionalGroupClassifier.")
    parser.add_argument("--data-dir", type=Path, default="data/raw")
    parser.add_argument("--checkpoint", type=Path, default="src/spectroscopy_qml/ir/mps_classifier/models_per_label/mps_model_best.pt")
    parser.add_argument("--output-dir", type=Path, default="src/spectroscopy_qml/ir/mps_classifier/results_per_label/error_analysis")
    parser.add_argument("--split", choices=["train", "val", "test", "trainval", "all"], default="test")

    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-savgol", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--savgol-window-length", type=int, default=25)
    parser.add_argument("--savgol-polyorder", type=int, default=4)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=None)
    parser.add_argument("--best-fold", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retune-thresholds-on", choices=["none", "train", "val", "test", "trainval", "selected"], default="none")
    parser.add_argument("--threshold-grid-step", type=float, default=0.05)

    parser.add_argument("--top-confusions", type=int, default=100)
    parser.add_argument("--max-sample-errors", type=int, default=500)
    parser.add_argument("--num-worst-probability-plots", type=int, default=12)

    # Optional architecture overrides, normally not needed because checkpoint has config.
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--num-sites", type=int, default=None)
    parser.add_argument("--physical-dim", type=int, default=None)
    parser.add_argument("--bond-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--classifier-head", choices=["mps", "cnn"], default=None)
    parser.add_argument("--num-sites-2", type=int, default=None)
    parser.add_argument("--physical-dim-2", type=int, default=None)
    parser.add_argument("--bond-dim-2", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(checkpoint, args, device)
    label_names = LABEL_NAMES[: model.num_classes]

    target_length = args.target_length or model.input_dim
    num_folds = args.num_folds if args.num_folds is not None else int(checkpoint.get("cv_num_folds", 1))
    best_fold = args.best_fold if args.best_fold is not None else checkpoint.get("best_fold")

    print("Loading and preprocessing data...")
    X, y, row_ids = load_ir_data_with_ids(
        data_dir=args.data_dir,
        target_length=target_length,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
        apply_savgol=args.apply_savgol,
        savgol_window_length=args.savgol_window_length,
        savgol_polyorder=args.savgol_polyorder,
        cache_path=args.cache_path,
        overwrite_cache=args.overwrite_cache,
    )
    y = y[:, : model.num_classes]

    splits = rebuild_splits(
        X=X,
        y=y,
        row_ids=row_ids,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        num_folds=num_folds,
        best_fold=int(best_fold) if best_fold is not None else None,
    )
    if args.split == "all":
        selected = {"X": X, "y": y, "row_ids": row_ids}
    else:
        selected = splits[args.split]

    thresholds = checkpoint.get("thresholds")
    if thresholds is None:
        thresholds = np.full(model.num_classes, 0.5, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)

    if args.retune_thresholds_on != "none":
        tune_split = args.split if args.retune_thresholds_on == "selected" else args.retune_thresholds_on
        tune_payload = {"X": X, "y": y, "row_ids": row_ids} if tune_split == "all" else splits[tune_split]
        print(f"Retuning thresholds on {tune_split} split...")
        tune_probs = predict_probs(model, tune_payload["X"], args.batch_size, device, args.use_amp)
        thresholds = tune_thresholds(tune_payload["y"], tune_probs, grid_step=args.threshold_grid_step)

    print(f"Running inference on split={args.split} ({len(selected['X'])} samples)...")
    probs = predict_probs(model, selected["X"], args.batch_size, device, args.use_amp)
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    labels = selected["y"].astype(np.int32)

    train_labels = splits["train"]["y"] if "train" in splits else splits["trainval"]["y"]

    print("Computing analysis tables...")
    per_class = class_metrics(labels, preds, probs, thresholds, label_names, train_labels=train_labels)
    distribution = global_distribution_rows(labels, preds, label_names)
    split_distribution = split_distribution_rows(splits, label_names)
    switch = switch_matrix(labels, preds)
    top_confusions = top_confusion_rows(switch, label_names, args.top_confusions)
    diagnosis = diagnosis_rows(per_class, switch, label_names)
    prob_summary = probability_summary_rows(labels, probs, thresholds, label_names)
    threshold_sensitivity = threshold_sensitivity_rows(labels, probs, label_names, grid_step=args.threshold_grid_step)
    sample_errors = sample_error_rows(labels, preds, probs, selected["row_ids"], label_names, args.max_sample_errors)

    save_csv(args.output_dir / "per_class_metrics.csv", per_class)
    save_csv(args.output_dir / "per_class_diagnosis.csv", diagnosis)
    save_csv(args.output_dir / "global_label_distribution.csv", distribution)
    save_csv(args.output_dir / "split_label_distribution.csv", split_distribution)
    save_matrix_csv(args.output_dir / "switch_matrix_false_negative_vs_false_positive.csv", switch, label_names, label_names)
    save_csv(args.output_dir / "top_confusions.csv", top_confusions)
    save_csv(args.output_dir / "probability_summary.csv", prob_summary)
    save_csv(args.output_dir / "threshold_sensitivity.csv", threshold_sensitivity)
    save_csv(args.output_dir / "sample_errors.csv", sample_errors)

    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision_micro = precision_score(labels, preds, average="micro", zero_division=0)
    recall_micro = recall_score(labels, preds, average="micro", zero_division=0)
    accuracy = accuracy_score(labels, preds)

    print("Creating plots...")
    plot_bar(plots_dir / "per_class_f1.png", label_names, [float(row["f1"]) for row in per_class], "Per-class F1", "F1")
    plot_bar(plots_dir / "support_true.png", label_names, [float(row["support_true"]) for row in per_class], "True support per functional group", "count")
    plot_bar(plots_dir / "pred_minus_true_pct.png", label_names, [float(row["pred_minus_true_pct"]) for row in distribution], "Prediction distribution bias: predicted % - true %", "percentage points as fraction")
    plot_heatmap(plots_dir / "switch_matrix_heatmap_top25.png", switch, label_names, "Multilabel switch proxy: FN true label vs FP predicted label", top_n=25)

    worst_by_f1 = sorted(per_class, key=lambda row: row["f1"])[: args.num_worst_probability_plots]
    worst_indices = [int(row["class_index"]) for row in worst_by_f1]
    plot_probability_histograms(plots_dir / "probability_histograms_worst_classes", labels, probs, label_names, worst_indices)

    model_config = normalize_model_config(checkpoint.get("config"))
    summary = {
        "split": args.split,
        "num_samples": int(labels.shape[0]),
        "num_labels": int(labels.shape[1]),
        "accuracy": float(accuracy),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "mean_threshold": float(np.mean(thresholds)),
        "std_threshold": float(np.std(thresholds)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_fold": int(checkpoint.get("best_fold", -1)) if checkpoint.get("best_fold") is not None else None,
        "checkpoint_cv_num_folds": int(checkpoint.get("cv_num_folds", num_folds)),
        "model_config": model_config,
        "preprocessing": {
            "target_length": target_length,
            "apply_snv": args.apply_snv,
            "apply_savgol": args.apply_savgol,
            "savgol_window_length": args.savgol_window_length,
            "savgol_polyorder": args.savgol_polyorder,
        },
        "worst_classes_by_f1": [
            {
                "label": row["label"],
                "f1": float(row["f1"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "support_true": int(row["support_true"]),
            }
            for row in worst_by_f1
        ],
        "most_over_predicted": sorted(distribution, key=lambda row: row["pred_minus_true_pct"], reverse=True)[:10],
        "most_under_predicted": sorted(distribution, key=lambda row: row["pred_minus_true_pct"])[:10],
        "top_confusions": top_confusions[:20],
        "artifacts": {
            "per_class_metrics": str(args.output_dir / "per_class_metrics.csv"),
            "per_class_diagnosis": str(args.output_dir / "per_class_diagnosis.csv"),
            "global_label_distribution": str(args.output_dir / "global_label_distribution.csv"),
            "split_label_distribution": str(args.output_dir / "split_label_distribution.csv"),
            "switch_matrix": str(args.output_dir / "switch_matrix_false_negative_vs_false_positive.csv"),
            "top_confusions": str(args.output_dir / "top_confusions.csv"),
            "sample_errors": str(args.output_dir / "sample_errors.csv"),
            "plots": str(plots_dir),
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print("\nDone.")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Micro F1:  {f1_micro:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")
    print(f"Precision: {precision_micro:.4f}")
    print(f"Recall:    {recall_micro:.4f}")
    print(f"Outputs:   {args.output_dir}")


if __name__ == "__main__":
    main()
