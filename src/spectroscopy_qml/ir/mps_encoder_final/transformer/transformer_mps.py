from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset

from spectroscopy_qml.ir.mps_encoder.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data
from spectroscopy_qml.ir.mps_encoder.model import MPSFunctionalGroupClassifier
from spectroscopy_qml.ir.mps_encoder.train import compute_pos_weight, tune_thresholds, validate
from .generate_input import process_ir, tokenize_formula

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class TransformerMPSConfig:
    max_seq_len: int = 256
    vocab_min_freq: int = 2
    vocab_max_size: int = 12000
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ffn_dim: int = 512
    dropout: float = 0.1
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    num_epochs: int = 15
    label_smoothing: float = 0.0
    use_formula: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    device: str = "cuda"
    random_seed: int = 42


class SimpleVocab:
    def __init__(self, tokens: list[str], min_freq: int = 1, max_size: int | None = None):
        token_counts = Counter(tokens)
        filtered = [tok for tok, count in token_counts.items() if count >= min_freq]
        filtered.sort(key=lambda tok: (-token_counts[tok], tok))

        if max_size is not None:
            filtered = filtered[: max_size - 2]

        self.idx_to_token = [PAD_TOKEN, UNK_TOKEN] + filtered
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        self.pad_idx = self.token_to_idx[PAD_TOKEN]
        self.unk_idx = self.token_to_idx[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_token)

    def encode(self, text: str, max_length: int) -> np.ndarray:
        tokens = text.strip().split()
        token_ids = [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
        if len(token_ids) >= max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (max_length - len(token_ids))
        return np.array(token_ids, dtype=np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx_to_token": self.idx_to_token,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleVocab:
        vocab = cls([])
        vocab.idx_to_token = data["idx_to_token"]
        vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
        vocab.pad_idx = vocab.token_to_idx[PAD_TOKEN]
        vocab.unk_idx = vocab.token_to_idx[UNK_TOKEN]
        return vocab


def build_vocab(texts: Iterable[str], min_freq: int = 2, max_size: int | None = None) -> SimpleVocab:
    tokens = []
    for text in texts:
        tokens.extend(text.strip().split())
    return SimpleVocab(tokens, min_freq=min_freq, max_size=max_size)


def create_source_text(row: pd.Series, use_formula: bool = True) -> str:
    source_parts: list[str] = []
    if use_formula and "molecular_formula" in row and pd.notna(row["molecular_formula"]):
        source_parts.append(tokenize_formula(str(row["molecular_formula"])))

    if "ir_spectra" in row:
        ir_value = row["ir_spectra"]
        if ir_value is not None and not (
            isinstance(ir_value, float) and np.isnan(ir_value)
        ):
            source_parts.append(process_ir(np.asarray(ir_value)))

    return " ".join(part.strip() for part in source_parts if part).strip()


def load_transformer_text_data(
    data_dir: Path,
    max_files: int | None = None,
    apply_snv: bool = DATA_CONFIG.apply_snv,
    apply_savgol: bool = DATA_CONFIG.apply_savgol,
    savgol_window_length: int = DATA_CONFIG.savgol_window_length,
    savgol_polyorder: int = DATA_CONFIG.savgol_polyorder,
) -> tuple[list[str], np.ndarray]:
    data_dir = Path(data_dir)
    print(f"Loading IR data for transformer text generation from {data_dir}")

    spectra, labels = load_ir_data(
        data_dir,
        target_length=DATA_CONFIG.target_length,
        max_files=max_files,
        apply_snv=apply_snv,
        apply_savgol=apply_savgol,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    source_texts = [process_ir(spec) for spec in spectra]
    if len(source_texts) == 0:
        raise ValueError(f"No valid transformer text samples found in {data_dir}")

    return source_texts, labels


class SpectraTextDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.BoolTensor(attention_mask)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        pad_idx: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_scale = math.sqrt(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_seq_len)

        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(input_ids) * self.embed_scale
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return self.layer_norm(x)


class TransformerMPSClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        config: TransformerMPSConfig,
        num_classes: int = 37,
        classifier_head: str = "mps",
    ):
        super().__init__()
        self.encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            pad_idx=pad_idx,
            max_seq_len=config.max_seq_len,
        )
        
        # Adapt transformer output to the fixed MPS site configuration
        # The MPS is configured to use 5 coarse sites and 10 fine sites.
        # For flat input, that corresponds to 5 * 360 = 1800 and 10 * 180 = 1800.
        # In site mode, we must feed the first branch as 5 sites of 360 dims.
        site_dim = MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites
        site_dim_2 = MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites_2
        assert site_dim == 360 and site_dim_2 == 180, (
            f"Expected MPS site sizes 360 and 180, got {site_dim} and {site_dim_2}"
        )

        self.pool_branch1 = nn.AdaptiveAvgPool1d(output_size=MODEL_CONFIG.num_sites)
        self.proj_branch1 = nn.Linear(config.embed_dim, site_dim) if config.embed_dim != site_dim else nn.Identity()

        self.mps_model = MPSFunctionalGroupClassifier(
            input_mode="site",
            input_dim=MODEL_CONFIG.input_dim,
            num_sites=MODEL_CONFIG.num_sites,
            site_dim=site_dim,
            physical_dim=MODEL_CONFIG.physical_dim,
            bond_dim=MODEL_CONFIG.bond_dim,
            num_classes=num_classes,
            dropout_rate=MODEL_CONFIG.dropout_rate,
            classifier_head=MODEL_CONFIG.classifier_head,
            num_sites_2=MODEL_CONFIG.num_sites_2,
            site_dim_2=site_dim_2,
            physical_dim_2=MODEL_CONFIG.physical_dim_2,
            bond_dim_2=MODEL_CONFIG.bond_dim_2,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        padding_mask = attention_mask == 0
        embeddings = self.encoder(input_ids, padding_mask=padding_mask)  # (batch, seq_len, embed_dim)

        x = embeddings.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.pool_branch1(x)  # (batch, embed_dim, num_sites)
        x = x.transpose(1, 2)  # (batch, num_sites, embed_dim)
        x = self.proj_branch1(x)  # (batch, num_sites, site_dim)

        return self.mps_model(x)

    def get_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _compute_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    num_samples = labels.shape[0]
    num_positives = labels.sum(axis=0, dtype=np.float32)
    num_negatives = num_samples - num_positives
    pos_weight = np.ones_like(num_positives, dtype=np.float32)
    np.divide(num_negatives, num_positives, out=pos_weight, where=num_positives > 0)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_transformer_mps_model(
    data_dir: Path | str | None = None,
    max_files: int | None = None,
    config: TransformerMPSConfig = TransformerMPSConfig(),
) -> dict[str, Any]:
    if data_dir is None:
        data_dir = Path(PATH_CONFIG.data_dir)
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        project_root = Path(__file__).parents[5]
        data_dir = project_root / "data" / "raw"

    print(f"Loading normalized IR data from {data_dir}")
    source_texts, labels = load_transformer_text_data(
        data_dir,
        max_files=max_files,
        apply_snv=DATA_CONFIG.apply_snv,
        apply_savgol=DATA_CONFIG.apply_savgol,
        savgol_window_length=DATA_CONFIG.savgol_window_length,
        savgol_polyorder=DATA_CONFIG.savgol_polyorder,
    )

    print(f"Building vocabulary from {len(source_texts)} examples")
    vocab = build_vocab(
        source_texts,
        min_freq=config.vocab_min_freq,
        max_size=config.vocab_max_size,
    )

    print(f"Vocabulary size: {vocab.vocab_size}")
    input_ids = np.stack([vocab.encode(text, config.max_seq_len) for text in source_texts])
    attention_mask = (input_ids != vocab.pad_idx).astype(np.int64)

    X_temp, X_test, mask_temp, mask_test, y_temp, y_test = train_test_split(
        input_ids,
        attention_mask,
        labels,
        test_size=TRAINING_CONFIG.test_ratio,
        random_state=TRAINING_CONFIG.random_seed,
        shuffle=True,
    )

    val_ratio_adjusted = TRAINING_CONFIG.val_ratio / (TRAINING_CONFIG.train_ratio + TRAINING_CONFIG.val_ratio)
    X_train, X_val, mask_train, mask_val, y_train, y_val = train_test_split(
        X_temp,
        mask_temp,
        y_temp,
        test_size=val_ratio_adjusted,
        random_state=TRAINING_CONFIG.random_seed,
        shuffle=True,
    )

    train_dataset = SpectraTextDataset(X_train, mask_train, y_train)
    val_dataset = SpectraTextDataset(X_val, mask_val, y_val)
    test_dataset = SpectraTextDataset(X_test, mask_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=True,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
    )

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    model = TransformerMPSClassifier(
        vocab_size=vocab.vocab_size,
        pad_idx=vocab.pad_idx,
        config=config,
    ).to(device)
    print(f"Transformer+MPS model parameters: {model.get_num_parameters():,}")

    pos_weight = compute_pos_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.7, patience=2
    )

    best_val_score = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_input_ids, batch_mask, batch_labels in train_loader:
            batch_input_ids = batch_input_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_input_ids, batch_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item()) * batch_input_ids.size(0)

        epoch_loss /= len(train_dataset)

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            thresholds=np.full(MODEL_CONFIG.num_classes, 0.5),
            use_amp=TRAINING_CONFIG.use_amp and device.type == "cuda",
        )

        scheduler.step(val_metrics["f1_micro"])

        print(
            f"Epoch {epoch}/{config.num_epochs}: train_loss={epoch_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_f1_micro={val_metrics['f1_micro']:.4f}, "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_micro"] > best_val_score:
            best_val_score = val_metrics["f1_micro"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "vocab": vocab.to_dict(),
            }

    if best_state is None:
        raise RuntimeError("No model state was saved during training.")

    model_dir = Path(PATH_CONFIG.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "transformer_mps_best.pt"
    torch.save(best_state, model_path)

    vocab_path = model_dir / "transformer_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, indent=2)

    print(f"Saved transformer+MPS model checkpoint to: {model_path}")
    print(f"Saved transformer vocabulary to: {vocab_path}")

    # Tune thresholds on validation set from the best model
    model.load_state_dict(best_state["model_state_dict"])
    _, _, val_labels, val_probs = validate(
        model,
        val_loader,
        criterion,
        device,
        thresholds=None,
        use_amp=TRAINING_CONFIG.use_amp and device.type == "cuda",
        return_probs=True,
    )
    best_thresholds = tune_thresholds(val_labels, val_probs, metric="f1_micro")

    test_loss, test_metrics = validate(
        model,
        test_loader,
        criterion,
        device,
        thresholds=best_thresholds,
        use_amp=TRAINING_CONFIG.use_amp and device.type == "cuda",
    )

    torch.save(
        {
            "model_state_dict": best_state["model_state_dict"],
            "config": config.__dict__,
            "vocab": vocab.to_dict(),
            "thresholds": best_thresholds,
            "pos_weight": pos_weight.detach().cpu().numpy(),
        },
        model_path,
    )

    print("Test metrics:", test_metrics)
    print(f"Using tuned thresholds (mean: {best_thresholds.mean():.3f})")

    return {
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "best_val_f1_micro": best_val_score,
        "best_thresholds": best_thresholds.tolist(),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }


def _evaluate_model(model: TransformerMPSClassifier, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_input_ids, batch_mask, batch_labels in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_mask = batch_mask.to(device)
            logits = model(batch_input_ids, batch_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(np.float32)
            all_preds.append(preds)
            all_labels.append(batch_labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    return _compute_metrics(all_labels, all_preds)
