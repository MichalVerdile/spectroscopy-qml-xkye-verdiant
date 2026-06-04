"""Boosted Tree Tensor Network helpers.

This file intentionally does not replace the existing TTN architecture.
It wraps the existing TTNCnmrClassifier10_2 as a binary weak learner and
builds an additive gradient-boosting style ensemble:

    F_j(x) = base_logit_j + eta * sum_t h_{j,t}(x)

where every functional group j receives its own binary boosted TTN ensemble.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.model import TTNCnmrClassifier10_2



class BoostedTTNBinaryEnsemble(nn.Module):
    """One boosted binary TTN ensemble for one functional group.

    Args:
        base_score:
            Initial logit, usually log(pos_rate / neg_rate).
        learning_rate:
            Shrinkage factor eta. Same role as XGBoost's learning_rate.
    """

    def __init__(self, base_score: float, learning_rate: float = 0.05) -> None:
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.register_buffer(
            "base_score",
            torch.tensor([float(base_score)], dtype=torch.float32),
            persistent=True,
        )
        self.learners = nn.ModuleList()

    def add_learner(self, learner: nn.Module, freeze: bool = True) -> None:
        """Append one fitted TTN weak learner."""
        if freeze:
            learner.eval()
            for parameter in learner.parameters():
                parameter.requires_grad_(False)
        self.learners.append(learner)

    def forward(
        self,
        x: Tensor,
        *,
        max_estimators: int | None = None,
        apply_sigmoid: bool = False,
    ) -> Tensor:
        """Return logits, or probabilities when apply_sigmoid=True."""
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, input_dim), got {tuple(x.shape)}.")

        logits = self.base_score.to(device=x.device, dtype=x.dtype).expand(x.size(0), 1).clone()
        learners = self.learners if max_estimators is None else self.learners[:max_estimators]

        for learner in learners:
            logits = logits + self.learning_rate * learner(x)

        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


class BoostedTTNMultiLabelClassifier(nn.Module):
    """Multi-label wrapper containing one boosted binary ensemble per label."""

    def __init__(self, label_ensembles: Sequence[BoostedTTNBinaryEnsemble] | None = None) -> None:
        super().__init__()
        self.label_ensembles = nn.ModuleList(label_ensembles or [])

    @property
    def num_labels(self) -> int:
        return len(self.label_ensembles)

    def add_label_ensemble(self, ensemble: BoostedTTNBinaryEnsemble) -> None:
        self.label_ensembles.append(ensemble)

    def forward(
        self,
        x: Tensor,
        *,
        max_estimators: int | None = None,
        apply_sigmoid: bool = False,
    ) -> Tensor:
        if not self.label_ensembles:
            raise RuntimeError("BoostedTTNMultiLabelClassifier contains no label ensembles.")

        label_logits = [
            ensemble(x, max_estimators=max_estimators, apply_sigmoid=False)
            for ensemble in self.label_ensembles
        ]
        logits = torch.cat(label_logits, dim=1)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


def build_ttn_weak_learner(model_kwargs: dict) -> TTNCnmrClassifier10_2:
    """Build one scalar-output TTN weak learner.

    model_kwargs must contain the architecture arguments except num_labels.
    """
    return TTNCnmrClassifier10_2(num_labels=1, **model_kwargs)


def build_boosted_ttn_from_metadata(
    *,
    model_kwargs: dict,
    base_scores: Sequence[float],
    learning_rate: float,
    estimators_per_label: Sequence[int],
    freeze: bool = True,
) -> BoostedTTNMultiLabelClassifier:
    """Recreate the boosted model structure before loading a state_dict."""
    if len(base_scores) != len(estimators_per_label):
        raise ValueError("base_scores and estimators_per_label must have the same length.")

    model = BoostedTTNMultiLabelClassifier()

    for base_score, num_estimators in zip(base_scores, estimators_per_label, strict=True):
        ensemble = BoostedTTNBinaryEnsemble(
            base_score=float(base_score),
            learning_rate=float(learning_rate),
        )
        for _ in range(int(num_estimators)):
            ensemble.add_learner(build_ttn_weak_learner(model_kwargs), freeze=freeze)
        model.add_label_ensemble(ensemble)

    return model
