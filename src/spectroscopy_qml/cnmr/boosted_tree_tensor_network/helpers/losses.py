"""Loss builders for experiment 5."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Multi-label Focal Loss with optional per-class pos_weight.

    FL(p_t) = -pos_weight * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing exponent. 0 = standard BCE. Default: 2.0.
        pos_weight: Per-class positive weight tensor, same as BCEWithLogitsLoss.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.sigmoid(logits) * targets + (1.0 - torch.sigmoid(logits)) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def build_loss(
    loss_name: str,
    pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build the configured multi-label loss."""
    normalized_name = loss_name.strip().lower()
    if normalized_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if normalized_name == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    raise NotImplementedError(
        f"Unsupported loss '{loss_name}'. Supported: 'bce', 'focal'."
    )
