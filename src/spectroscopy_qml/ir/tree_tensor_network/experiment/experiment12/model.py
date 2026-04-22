"""Experiment 12 models for compression-vs-classification comparisons.

This experiment keeps the engineered spectral channels from experiment 10.x and
compares three downstream strategies:

1. Flat MLP on the engineered feature vector.
2. PCA reduction followed by the same MLP head.
3. TTN-inspired hierarchical compression followed by an MLP readout.

The goal is not to build another full TTN classifier, but to test whether a
structure-preserving compression stage helps more than standard linear
dimensionality reduction.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_3.model import (
    VoigtFeatureMap,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_4.model import (
    SavitzkyGolayFeatureMap,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_5.model import (
    SavitzkyGolayFeatureMapNoNorm,
)


class Experiment12FeatureExtractor(nn.Module):
    """Fixed engineered feature extractor shared across Experiment 12 models."""

    def __init__(
        self,
        feature_source: str = "sg_no_norm",
        *,
        include_raw_channel: bool = False,
        sg_window_length: int = 11,
        sg_polyorder: int = 3,
        voigt_gamma_l: float = 3.0,
        voigt_gamma_g: float = 2.0,
        voigt_eta: float = 0.5,
        voigt_kernel_half_width: int = 20,
    ) -> None:
        super().__init__()
        if feature_source == "sg_no_norm":
            feature_map = SavitzkyGolayFeatureMapNoNorm(
                window_length=sg_window_length,
                polyorder=sg_polyorder,
            )
            base_channels = 3
        elif feature_source == "sg_z_score":
            feature_map = SavitzkyGolayFeatureMap(
                window_length=sg_window_length,
                polyorder=sg_polyorder,
                norm_mode="z_score",
            )
            base_channels = 3
        elif feature_source == "voigt_z_score":
            feature_map = VoigtFeatureMap(
                gamma_l=voigt_gamma_l,
                gamma_g=voigt_gamma_g,
                eta=voigt_eta,
                kernel_half_width=voigt_kernel_half_width,
                norm_mode="z_score",
            )
            base_channels = 3
        else:
            raise ValueError(
                "feature_source must be one of 'sg_no_norm', 'sg_z_score', or 'voigt_z_score'."
            )

        self.feature_source = feature_source
        self.include_raw_channel = bool(include_raw_channel)
        self.feature_map = feature_map
        self.output_channels = base_channels + int(self.include_raw_channel)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected shape (batch_size, spectrum_length), got {tuple(x.shape)}.")
        features = self.feature_map(x)
        if self.include_raw_channel:
            features = torch.cat((x.unsqueeze(-1), features), dim=-1)
        return features


class MLPClassifier(nn.Module):
    """Small MLP baseline used after flat features or PCA."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")

        dims = [int(input_dim), *[int(d) for d in hidden_dims], int(num_labels)]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=False):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected flat feature tensor of shape (batch, dim), got {tuple(x.shape)}.")
        return self.network(x)


class HierarchicalTNCompressor(nn.Module):
    """TTN-inspired hierarchical compressor for ordered spectral feature sequences.

    The sequence is first chunked into local blocks of ``site_length`` points.
    Each block is projected into a fixed bond dimension. Neighboring blocks are
    then merged bottom-up with shared-width merge layers until a single root
    representation remains.
    """

    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        bond_dim: int = 128,
        site_length: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if bond_dim <= 0:
            raise ValueError("bond_dim must be positive.")
        if site_length <= 0:
            raise ValueError("site_length must be positive.")

        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.bond_dim = int(bond_dim)
        self.site_length = int(site_length)
        self.num_sites = math.ceil(self.sequence_length / self.site_length)
        self.padded_length = self.num_sites * self.site_length

        site_input_dim = self.site_length * self.feature_dim
        self.site_encoder = nn.Sequential(
            nn.Linear(site_input_dim, self.bond_dim),
            nn.LayerNorm(self.bond_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        num_levels = 0 if self.num_sites <= 1 else math.ceil(math.log2(self.num_sites))
        self.merge_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.bond_dim, self.bond_dim),
                    nn.LayerNorm(self.bond_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_levels)
            ]
        )
        self.output_norm = nn.LayerNorm(self.bond_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected sequence tensor of shape (batch, length, channels), got {tuple(x.shape)}."
            )
        if x.size(1) != self.sequence_length or x.size(2) != self.feature_dim:
            raise ValueError(
                "Unexpected sequence shape: "
                f"expected (*, {self.sequence_length}, {self.feature_dim}), got {tuple(x.shape)}."
            )

        if self.padded_length != self.sequence_length:
            pad = self.padded_length - self.sequence_length
            x = F.pad(x, (0, 0, 0, pad))

        batch_size = x.size(0)
        sites = x.view(batch_size, self.num_sites, self.site_length * self.feature_dim)
        nodes = self.site_encoder(sites)

        level_index = 0
        while nodes.size(1) > 1:
            paired = nodes.size(1) // 2
            left = nodes[:, : 2 * paired : 2, :]
            right = nodes[:, 1 : 2 * paired : 2, :]
            merged = self.merge_layers[level_index](torch.cat((left, right), dim=-1))
            if nodes.size(1) % 2 == 1:
                nodes = torch.cat((merged, nodes[:, -1:, :]), dim=1)
            else:
                nodes = merged
            level_index += 1

        return self.output_norm(nodes[:, 0, :])


class TNCompressionClassifier(nn.Module):
    """Hierarchical TN-like compression followed by an MLP readout head."""

    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        num_labels: int,
        bond_dim: int = 128,
        site_length: int = 16,
        head_hidden_dims: tuple[int, ...] = (256,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.compressor = HierarchicalTNCompressor(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            bond_dim=bond_dim,
            site_length=site_length,
            dropout=dropout,
        )
        self.head = MLPClassifier(
            input_dim=bond_dim,
            num_labels=num_labels,
            hidden_dims=head_hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        compressed = self.compressor(x)
        return self.head(compressed)
