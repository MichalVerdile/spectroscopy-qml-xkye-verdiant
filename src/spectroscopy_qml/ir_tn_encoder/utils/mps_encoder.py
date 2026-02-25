# ============================================================
# MPS / TT ENCODER PIPELINE (IR Spectrum -> Embedding)
#
# Notation:
#   L = input_length (e.g., 512)
#   S = num_sites    (current default: 32)
#   site_dim = L / S (current default: 512/32 = 16 values per site)
#   p = physical_dim (current default: 8)
#   D = bond_dim     (e.g., 16)
#   E = embedding_dim (e.g., 128)
#
# Input:
#   x: (batch_size, L)
#
# 1) Split spectrum into S sites (chunks)
#   x.view(batch, S, site_dim)
#   -> (batch, S, site_dim)
#
# 2) Local feature map per site (MLP)
#   For each site i:
#     site_input = x[:, i, :]        # (batch, site_dim)
#     phi_i = FeatureMap(site_input) # (batch, p)
#   Result:
#     features = [phi_0, ..., phi_{S-1}]  each (batch, p)
#
# 3) Contract MPS cores
#   Cores:
#     core_0: (1, p, D)              # left boundary
#     core_i: (D, p, D)              # bulk
#
#   3a) Forward sweep (left -> right):
#       result = einsum("ipd,bp->bid", core_0, phi_0)  # (batch, 1, D)
#       result = squeeze -> (batch, D)
#
#   3b) For i = 1..S-1:
#       result = einsum("bd,dpD,bp->bD", result, core_i, phi_i)
#       Shapes:
#         result: (batch, D_left)
#         core_i: (D_left, p, D_right)
#         phi_i:  (batch, p)
#       Output:
#         result: (batch, D_right)
#
#   3c) Backward sweep (right -> left):
#       same contraction on reversed features with a second core stack
#       -> backward_state: (batch, D)
#
#   3d) Concatenate directional states:
#       state = concat([forward_state, backward_state]) -> (batch, 2D)
#
# 4) Output projection to embedding space
#   embedding = LayerNorm(2D) + Linear(2D -> E)
#   -> (batch, E)
#
# Output:
#   embedding: (batch_size, embedding_dim)
# ============================================================

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LocalFeatureMap(nn.Module):
    """
    Local feature map for each site.

    Maps a single site input to a physical dimension vector.
    """

    def __init__(
        self,
        site_dim: int,
        physical_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        """
        Initialize local feature map.

        Args:
            site_dim: Input dimension per site
            physical_dim: Output physical dimension (p)
            hidden_dim: Hidden dimension for MLP (defaults to site_dim)
        """
        super().__init__()
        self.site_dim = site_dim
        self.physical_dim = physical_dim

        if hidden_dim is None:
            hidden_dim = max(site_dim, physical_dim * 2)

        self.mlp = nn.Sequential(
            nn.LayerNorm(site_dim),
            nn.Linear(site_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, physical_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, site_dim)

        Returns:
            Features of shape (batch, physical_dim)
        """
        result: torch.Tensor = self.mlp(x)
        return result


class MPSEncoder(nn.Module):
    """
    MPS/TT-style encoder for 1D spectra.

    Architecture:
    1. Split input into S sites (chunks)
    2. Apply local feature map to each site -> physical dimension p
    3. Contract with learned MPS cores (tensors)
    4. Output embedding vector

    The MPS cores are stored as weight matrices that get contracted with
    the local features in a sequential manner (left-to-right sweep).

    For a single sample, the computation is:
        result = (((A_1 @ phi_1) @ A_2 @ phi_2) @ ... @ A_S @ phi_S)

    where:
        - A_i are the MPS cores with shape (D_left, p, D_right)
        - phi_i are the local features with shape (p,)
        - D is the bond dimension
    """

    def __init__(
        self,
        input_length: int = 512,
        num_sites: int = 64,
        physical_dim: int = 4,
        bond_dim: int = 16,
        embedding_dim: int = 128,
        shared_feature_map: bool = True,
        normalize_state: bool = True,
        state_eps: float = 1e-6,
        bidirectional: bool = True,
        feature_dropout: float = 0.1,
    ):
        """
        Initialize MPS encoder.

        Args:
            input_length: Length of input spectrum
            num_sites: Number of sites (S) to split input into
            physical_dim: Physical dimension (p) at each site
            bond_dim: Bond dimension (D) connecting sites
            embedding_dim: Output embedding dimension
            shared_feature_map: Whether to share feature map across sites
        """
        super().__init__()
        self.input_length = input_length
        self.num_sites = num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.embedding_dim = embedding_dim
        self.normalize_state = normalize_state
        self.state_eps = state_eps
        self.bidirectional = bidirectional

        # Compute site dimension (input values per site)
        assert (
            input_length % num_sites == 0
        ), f"input_length ({input_length}) must be divisible by num_sites ({num_sites})"
        self.site_dim = input_length // num_sites

        # Local feature maps
        if shared_feature_map:
            feature_map = LocalFeatureMap(self.site_dim, physical_dim, dropout=feature_dropout)
            self.feature_maps = nn.ModuleList([feature_map] * num_sites)
        else:
            self.feature_maps = nn.ModuleList(
                [
                    LocalFeatureMap(self.site_dim, physical_dim, dropout=feature_dropout)
                    for _ in range(num_sites)
                ]
            )
        self.shared_feature_map = shared_feature_map

        self.cores_forward = self._build_core_stack(physical_dim, bond_dim)
        if self.bidirectional:
            self.cores_backward = self._build_core_stack(physical_dim, bond_dim)
            self.output_projection = nn.Sequential(
                nn.LayerNorm(2 * bond_dim),
                nn.Linear(2 * bond_dim, embedding_dim),
            )
        else:
            self.output_projection = nn.Sequential(
                nn.LayerNorm(bond_dim),
                nn.Linear(bond_dim, embedding_dim),
            )

    def _build_core_stack(self, physical_dim: int, bond_dim: int) -> nn.ParameterList:
        """
        Build one directional MPS core stack with stable initialization.
        """
        cores = nn.ParameterList()
        left = torch.empty(1, physical_dim, bond_dim)
        nn.init.xavier_uniform_(left)
        cores.append(nn.Parameter(left))

        for _ in range(self.num_sites - 1):
            core = torch.empty(bond_dim, physical_dim, bond_dim)
            nn.init.xavier_uniform_(core)
            cores.append(nn.Parameter(core))
        return cores

    def _contract_direction(
        self, features: list[torch.Tensor], cores: nn.ParameterList
    ) -> torch.Tensor:
        """
        Contract one MPS direction and return the hidden state (batch, bond_dim).
        """
        result = torch.einsum("ipd,bp->bid", cores[0], features[0]).squeeze(1)
        if self.normalize_state:
            result = result / (result.norm(dim=1, keepdim=True) + self.state_eps)

        for i in range(1, self.num_sites):
            result = torch.einsum("bd,dpD,bp->bD", result, cores[i], features[i])
            if self.normalize_state:
                result = result / (result.norm(dim=1, keepdim=True) + self.state_eps)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_length)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        batch_size = x.shape[0]

        x = x.view(batch_size, self.num_sites, self.site_dim)

        # Apply feature maps to each site
        # Result: list of (batch, physical_dim) tensors
        features = []
        for i in range(self.num_sites):
            site_input = x[:, i, :]  # (batch, site_dim)
            if self.shared_feature_map:
                phi = self.feature_maps[0](site_input)
            else:
                phi = self.feature_maps[i](site_input)
            features.append(phi)

        # Forward contraction
        forward_state = self._contract_direction(features, self.cores_forward)
        if self.bidirectional:
            backward_state = self._contract_direction(list(reversed(features)), self.cores_backward)
            state = torch.cat([forward_state, backward_state], dim=1)
        else:
            state = forward_state

        embedding: torch.Tensor = self.output_projection(state)

        return embedding
