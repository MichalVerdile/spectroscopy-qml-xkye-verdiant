"""MLP encoder for IR spectra."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):  # type: ignore[misc]
    """
    Simple MLP encoder for IR spectra.

    Architecture:
    - Multiple fully connected layers with LayerNorm and ReLU
    - Final projection to embedding dimension
    """

    def __init__(
        self,
        input_length: int = 512,
        embedding_dim: int = 128,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_length = input_length
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers: list[nn.Module] = []

        in_dim = input_length
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, embedding_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_length) -> (batch, embedding_dim)."""
        return self.layers(x)
