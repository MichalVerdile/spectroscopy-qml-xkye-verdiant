"""Experiment 7 classifier with SNV preprocessing, derivative features, and a plain MLP encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.model import (
    SpectralDerivativeFeatureMap,
)


class SpectrumMLPEncoder(nn.Module):
    """Encode the full derivative-augmented spectrum with a simple MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        feature_dim: int = 3,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        renormalize_output: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.feature_dim = int(feature_dim)
        self.renormalize_output = bool(renormalize_output)

        flattened_dim = self.input_dim * self.feature_dim
        if hidden_dim is None:
            hidden_dim = max(1024, 8 * self.output_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_norm = nn.LayerNorm(flattened_dim)
        self.fc1 = nn.Linear(flattened_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch_size, input_dim, feature_dim).")
        if features.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {features.size(1)}.")
        if features.size(2) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {features.size(2)}.")

        flattened = features.flatten(start_dim=1)
        encoded = self.input_norm(flattened)
        encoded = self.fc1(encoded)
        encoded = self.activation(encoded)
        encoded = self.dropout(encoded)
        encoded = self.fc2(encoded)
        encoded = self.output_norm(encoded)
        if self.renormalize_output:
            encoded = F.normalize(encoded, dim=-1, eps=1e-8)
        return encoded


class MLPEncoderIRClassifier7(nn.Module):
    """Experiment 7: derivative-aware full-spectrum MLP classifier."""

    def __init__(
        self,
        num_labels: int,
        chi: int,
        input_dim: int = 1800,
        encoder_hidden_dim: int | None = None,
        encoder_dropout: float = 0.1,
        encoder_renormalize_output: bool = True,
        readout_hidden_dim: int | None = None,
        readout_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")
        if chi <= 0:
            raise ValueError("chi must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        self.num_labels = int(num_labels)
        self.chi = int(chi)
        self.input_dim = int(input_dim)

        self.feature_map = SpectralDerivativeFeatureMap()
        self.encoder = SpectrumMLPEncoder(
            input_dim=self.input_dim,
            output_dim=self.chi,
            feature_dim=3,
            hidden_dim=encoder_hidden_dim,
            dropout=encoder_dropout,
            renormalize_output=encoder_renormalize_output,
        )

        if readout_hidden_dim is None:
            readout_hidden_dim = max(128, 4 * self.chi)
        self.readout_hidden_dim = int(readout_hidden_dim)

        self.output_norm = nn.LayerNorm(self.chi)
        self.output_head = nn.Sequential(
            nn.Linear(self.chi, self.readout_hidden_dim),
            nn.GELU(),
            nn.Dropout(readout_dropout),
            nn.Linear(self.readout_hidden_dim, self.num_labels),
        )

    def forward(self, x: Tensor, apply_sigmoid: bool = False) -> Tensor:
        if not torch.is_tensor(x):
            raise TypeError("MLPEncoderIRClassifier7 expects a torch.Tensor input.")
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, input_dim), got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input feature dimension {self.input_dim}, got {x.size(1)}.")

        features = self.feature_map(x)
        encoded = self.encoder(features)
        logits = self.output_head(self.output_norm(encoded))
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
