from __future__ import annotations

import torch
from torch import nn

from spectroscopy_qml.ir.mps_classifier.model import MPSFunctionalGroupClassifier
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import TTNIRClassifier10_2


class MergedMPSTTNClassifier(nn.Module):
    """Run the MPS and TTN branches in parallel and expose both outputs."""

    def __init__(
        self,
        input_dim: int = 1800,
        num_classes: int = 37,
        fusion_weight_init: float = 0.5,
        fusion_weight_trainable: bool = True,
        mps_num_sites: int = 5,
        mps_physical_dim: int = 450,
        mps_bond_dim: int = 128,
        mps_dropout_rate: float = 0.49,
        mps_classifier_head: str = "mps",
        mps_num_sites_2: int = 10,
        mps_physical_dim_2: int = 450,
        mps_bond_dim_2: int = 128,
        ttn_chi: int = 64,
        ttn_segment_window_size: int = 48,
        ttn_segment_stride: int = 24,
        ttn_segment_mode: str = "overlap",
        ttn_segment_offset: int | None = None,
        ttn_segment_state_normalize: bool = True,
        ttn_merge_mode: str = "relaxed",
        ttn_merge_residual_weight: float = 0.1,
        ttn_merge_renormalize_output: bool = True,
        ttn_lorentz_gamma: float = 3.0,
        ttn_lorentz_kernel_half_width: int = 15,
        ttn_lorentz_norm_mode: str = "percentile",
    ) -> None:
        super().__init__()
        if not 0.0 <= fusion_weight_init <= 1.0:
            raise ValueError("fusion_weight_init must be in [0, 1].")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.mps_model = MPSFunctionalGroupClassifier(
            input_dim=input_dim,
            num_sites=mps_num_sites,
            physical_dim=mps_physical_dim,
            bond_dim=mps_bond_dim,
            num_classes=num_classes,
            dropout_rate=mps_dropout_rate,
            classifier_head=mps_classifier_head,
            num_sites_2=mps_num_sites_2,
            physical_dim_2=mps_physical_dim_2,
            bond_dim_2=mps_bond_dim_2,
        )
        self.ttn_model = TTNIRClassifier10_2(
            num_labels=num_classes,
            chi=ttn_chi,
            input_dim=input_dim,
            segment_window_size=ttn_segment_window_size,
            segment_stride=ttn_segment_stride,
            segment_mode=ttn_segment_mode,
            segment_offset=ttn_segment_offset,
            segment_state_normalize=ttn_segment_state_normalize,
            merge_mode=ttn_merge_mode,
            merge_residual_weight=ttn_merge_residual_weight,
            merge_renormalize_output=ttn_merge_renormalize_output,
            lorentz_gamma=ttn_lorentz_gamma,
            lorentz_kernel_half_width=ttn_lorentz_kernel_half_width,
            lorentz_norm_mode=ttn_lorentz_norm_mode,
        )

        init_logit = torch.logit(torch.full((num_classes,), float(fusion_weight_init)))
        self.fusion_logits = nn.Parameter(init_logit, requires_grad=fusion_weight_trainable)

    def get_fusion_weights(self) -> torch.Tensor:
        """Return the per-class fusion weight $w$ in [0, 1]."""
        return torch.sigmoid(self.fusion_logits)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        use_mps_mask: torch.Tensor | None = None,
    ):
        logits_mps = self.mps_model(x)
        logits_ttn = self.ttn_model(x)

        probs_mps = torch.sigmoid(logits_mps)
        probs_ttn = torch.sigmoid(logits_ttn)
        if use_mps_mask is not None:
            return self.select_branch_probs(probs_mps, probs_ttn, use_mps_mask.to(torch.bool))

        fusion_weights = self.get_fusion_weights().unsqueeze(0)
        fused_probs = fusion_weights * probs_mps + (1.0 - fusion_weights) * probs_ttn

        if return_components:
            return {
                "logits_mps": logits_mps,
                "logits_ttn": logits_ttn,
                "probs_mps": probs_mps,
                "probs_ttn": probs_ttn,
                "fusion_weights": fusion_weights,
                "probs": fused_probs,
            }
        return fused_probs

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def select_branch_probs(
        probs_mps: torch.Tensor,
        probs_ttn: torch.Tensor,
        use_mps_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select per-class probabilities from the better branch."""
        return torch.where(use_mps_mask.unsqueeze(0), probs_mps, probs_ttn)