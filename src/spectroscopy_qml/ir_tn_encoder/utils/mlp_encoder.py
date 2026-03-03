"""MLP encoder for IR spectra."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for IR spectra.

    Architecture:
    - Multiple fully connected layers with LayerNorm and ReLU
    - Final projection to embedding dimension
    """

    def __init__(
        self,
        input_length: int = 600,
        embedding_dim: int = 128,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_length = input_length
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: list[nn.Module] = []

        in_dim = input_length
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, embedding_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_length) -> (batch, embedding_dim)."""
        result: torch.Tensor = self.layers(x)
        return result
