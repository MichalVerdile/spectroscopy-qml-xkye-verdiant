"""
CNN-based classifier adapted from Jung et al. baseline.

This classifier uses the dense layer architecture from the Jung CNN baseline,
designed to work with encoder embeddings rather than raw spectra.

Reference:
    Jung et al. (https://github.com/gj475/irchracterizationcnn)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    CNN-based classifier using Jung et al. architecture.

    Takes encoder embeddings and applies multi-layer dense network
    with dropout for functional group classification.

    Architecture (adapted from Jung baseline):
    - Dense layer 1: embedding_dim → 4927 (relu, dropout)
    - Dense layer 2: 4927 → 2785 (relu, dropout)
    - Dense layer 3: 2785 → 1574 (relu, dropout)
    - Output layer: 1574 → num_classes (sigmoid)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 37,
        dropout: float = 0.48599073736368,
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize CNN classifier.

        Args:
            embedding_dim: Dimension of encoder output
            num_classes: Number of functional group classes
            dropout: Dropout probability (default from Jung baseline)
            hidden_dims: Hidden layer dimensions (default: [4927, 2785, 1574] from Jung)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if hidden_dims is None:
            hidden_dims = [4927, 2785, 1574]

        layers: list[nn.Module] = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings of shape (batch, embedding_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(x)


class FunctionalGroupClassifier(nn.Module):
    """
    Full model: encoder + CNN classifier head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        embedding_dim: int = 128,
        dropout: float = 0.48599073736368,
    ):
        """
        Initialize classifier.

        Args:
            encoder: Encoder module (MLP, MPS, or MPSSimple)
            num_classes: Number of functional group classes
            embedding_dim: Encoder output dimension
            dropout: Dropout probability for CNN classifier
        """
        super().__init__()
        self.encoder = encoder
        self.classifier = CNNClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrum of shape (batch, input_length)

        Returns:
            Logits of shape (batch, num_classes)
        """
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits
