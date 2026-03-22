"""Small runnable example for the TTN-inspired IR classifier."""

from __future__ import annotations

import torch
from torch import nn

from spectroscopy_qml.ir.tree_tensor_network import TTNIRClassifier


def main() -> None:
    """Run a forward pass and a single dummy training step."""
    batch_size = 4
    input_dim = 1800
    num_labels = 12

    spectra = torch.randn(batch_size, input_dim)
    dummy_labels = torch.randint(0, 2, (batch_size, num_labels), dtype=torch.float32)

    model = TTNIRClassifier(
        num_labels=num_labels,
        chi=32,
        segment_window_size=48,
        segment_stride=24,
        embedding_scale=0.1,
        x_max_mode="per_sample",
        use_bias=True,
        merge_normalization="layernorm",
        merge_residual_weight=0.25,
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
