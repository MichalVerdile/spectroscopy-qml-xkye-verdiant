"""Loss builders for experiment 5."""

from __future__ import annotations

import torch
from torch import nn


def build_loss(loss_name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Build the configured multi-label loss.

    The function is intentionally narrow for now so additional losses such as
    focal or asymmetric losses can be added without reshaping the training loop.
    """
    normalized_name = loss_name.strip().lower()
    if normalized_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise NotImplementedError(
        f"Unsupported loss '{loss_name}'. Experiment5 currently implements only 'bce'."
    )
