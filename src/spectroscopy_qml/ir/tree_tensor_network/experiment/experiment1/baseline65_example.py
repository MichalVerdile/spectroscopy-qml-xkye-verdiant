"""Small runnable example for the restored 0.65 TTN baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

SRC_DIR = Path(__file__).parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.tree_tensor_network.baseline65_model import Baseline65TTNIRClassifier


def main() -> None:
    batch_size = 4
    input_dim = 1800
    num_labels = 12

    spectra = torch.randn(batch_size, input_dim)
    dummy_labels = torch.randint(0, 2, (batch_size, num_labels), dtype=torch.float32)

    model = Baseline65TTNIRClassifier(
        num_labels=num_labels,
        chi=64,
        num_segments=32,
        embedding_scale=0.1,
        x_max_mode="per_sample",
        use_bias=True,
        input_dim=input_dim,
    )

    logits = model(spectra)
    probabilities = model(spectra, apply_sigmoid=True)

    print(f"Input shape: {tuple(spectra.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Sigmoid output shape: {tuple(probabilities.shape)}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss = criterion(logits, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
