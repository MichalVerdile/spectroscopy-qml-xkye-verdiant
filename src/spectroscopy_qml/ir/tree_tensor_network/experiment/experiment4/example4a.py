"""Minimal example for Experiment4A."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model4a import TTNIRClassifier4A


def main() -> None:
    torch.manual_seed(0)
    model = TTNIRClassifier4A(
        num_labels=12,
        chi=32,
        num_segments=32,
        input_dim=1800,
    )
    x = torch.randn(4, 1800)
    y = torch.randint(0, 2, (4, 12), dtype=torch.float32)

    logits = model(x)
    probs = model(x, apply_sigmoid=True)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()

    print(f"Input shape:          {tuple(x.shape)}")
    print(f"Logits shape:         {tuple(logits.shape)}")
    print(f"Sigmoid output shape: {tuple(probs.shape)}")
    print(f"Training loss:        {loss.item():.4f}")


if __name__ == "__main__":
    main()
