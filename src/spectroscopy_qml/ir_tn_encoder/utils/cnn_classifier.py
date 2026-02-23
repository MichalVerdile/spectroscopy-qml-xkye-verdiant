"""
CNN-based classifier for encoder embeddings.

This classifier uses 1D convolutional layers followed by fully connected layers
to classify functional groups from encoder embeddings.

Architecture:
- Takes encoder embeddings (1D vectors)
- Applies 1D convolutions with batch normalization and pooling
- Flattens and passes through FC layers for classification
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    CNN-based classifier for encoder embeddings.

    Takes encoder embeddings and applies 1D convolutional layers
    followed by fully connected layers for functional group classification.

    Architecture:
    - Reshape embedding to (batch, 1, embedding_dim) for 1D convolutions
    - Conv1D blocks with increasing channels and pooling
    - Flatten and pass through FC layers
    - Output layer for classification
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 37,
        dropout: float = 0.3,
        conv_channels: list[int] | None = None,
        fc_dims: list[int] | None = None,
    ):
        """
        Initialize CNN classifier.

        Args:
            embedding_dim: Dimension of encoder output
            num_classes: Number of functional group classes
            dropout: Dropout probability
            conv_channels: Number of channels for each conv layer (default: [32, 64, 128])
            fc_dims: Fully connected layer dimensions (default: [256, 128])
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if fc_dims is None:
            fc_dims = [256, 128]

        # Convolutional layers
        conv_layers: list[nn.Module] = []
        in_channels = 1  # Start with 1 channel (reshaped embedding)
        current_length = embedding_dim

        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
            current_length = current_length // 2

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened size after conv layers
        self.flatten_size = conv_channels[-1] * current_length

        # Fully connected layers
        fc_layers: list[nn.Module] = []
        in_dim = self.flatten_size

        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(in_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = fc_dim

        # Output layer
        fc_layers.append(nn.Linear(in_dim, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings of shape (batch, embedding_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Reshape to (batch, 1, embedding_dim) for 1D convolutions
        x = x.unsqueeze(1)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        logits = self.fc_layers(x)
        
        return logits


class FunctionalGroupClassifier(nn.Module):
    """
    Full model: encoder + CNN classifier head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        conv_channels: list[int] | None = None,
        fc_dims: list[int] | None = None,
    ):
        """
        Initialize classifier.

        Args:
            encoder: Encoder module (MLP, MPS, or MPSSimple)
            num_classes: Number of functional group classes
            embedding_dim: Encoder output dimension
            dropout: Dropout probability for CNN classifier
            conv_channels: Number of channels for each conv layer
            fc_dims: Fully connected layer dimensions
        """
        super().__init__()
        self.encoder = encoder
        self.classifier = CNNClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            conv_channels=conv_channels,
            fc_dims=fc_dims,
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
