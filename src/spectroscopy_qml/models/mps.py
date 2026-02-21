# ============================================================
# MPS / TT ENCODER PIPELINE (IR Spectrum -> Embedding)
#
# Notation:
#   L = input_length (e.g., 512)
#   S = num_sites    (e.g., 64)
#   site_dim = L / S (e.g., 512/64 = 8 values per site)
#   p = physical_dim (e.g., 4)
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
# 3) Contract MPS cores left-to-right (sequential sweep)
#   Cores:
#     core_0: (1, p, D)              # left boundary
#     core_i: (D, p, D)              # bulk (and last core in this implementation)
#
#   3a) Initialize with left boundary:
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
#   Final contracted state:
#     result: (batch, D)
#
# 4) Output projection to embedding space
#   embedding = Linear(D -> E)
#   -> (batch, E)
#
# Output:
#   embedding: (batch_size, embedding_dim)
# ============================================================

"""
Tensor Network (MPS/TT) encoder for IR spectra.

Implements a Matrix Product State (MPS) / Tensor Train (TT) style encoder
that processes the spectrum as a tensor network.

Key concepts:
- Input spectrum is split into S sites
- Each site has a local feature map with physical dimension p
- Sites are connected via bond dimensions D
- The contracted network produces an embedding vector
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LocalFeatureMap(nn.Module):  # type: ignore[misc]
    """
    Local feature map for each site.

    Maps a single site input to a physical dimension vector.
    """

    def __init__(
        self,
        site_dim: int,
        physical_dim: int,
        hidden_dim: int | None = None,
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
            nn.Linear(site_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, physical_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, site_dim)

        Returns:
            Features of shape (batch, physical_dim)
        """
        return self.mlp(x)


class MPSEncoder(nn.Module):  # type: ignore[misc]
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

        # Compute site dimension (input values per site)
        assert (
            input_length % num_sites == 0
        ), f"input_length ({input_length}) must be divisible by num_sites ({num_sites})"
        self.site_dim = input_length // num_sites

        # Local feature maps
        if shared_feature_map:
            feature_map = LocalFeatureMap(self.site_dim, physical_dim)
            self.feature_maps = nn.ModuleList([feature_map] * num_sites)
        else:
            self.feature_maps = nn.ModuleList(
                [LocalFeatureMap(self.site_dim, physical_dim) for _ in range(num_sites)]
            )
        self.shared_feature_map = shared_feature_map

        # MPS cores
        # Left boundary: (1, p, D)
        # Bulk: (D, p, D)
        # Right boundary: (D, p, 1)
        self.cores = nn.ParameterList()

        # Left boundary core
        self.cores.append(
            nn.Parameter(torch.randn(1, physical_dim, bond_dim) / math.sqrt(physical_dim))
        )

        # Bulk cores
        for _ in range(num_sites - 2):
            self.cores.append(
                nn.Parameter(
                    torch.randn(bond_dim, physical_dim, bond_dim)
                    / math.sqrt(bond_dim * physical_dim)
                )
            )

        # Right boundary core
        self.cores.append(
            nn.Parameter(
                torch.randn(bond_dim, physical_dim, 1) / math.sqrt(bond_dim * physical_dim)
            )
        )

        # Output projection
        # The contracted MPS outputs a scalar per sample, but we want embedding_dim
        # Solution: Use multiple MPS "channels" and project
        self.num_channels = embedding_dim
        self.output_cores = nn.ParameterList()

        # Create separate output cores for each embedding channel
        # Alternative: use a single MPS with bond_dim * embedding_dim and reshape
        # Here we use the simpler approach: project from bond_dim at the end

        # Actually, let's use a different approach:
        # Contract the MPS to get a (batch, bond_dim) vector and project
        self._use_final_projection = True

        # Override cores for final projection approach
        # Right boundary now outputs bond_dim instead of 1
        self.cores[-1] = nn.Parameter(
            torch.randn(bond_dim, physical_dim, bond_dim) / math.sqrt(bond_dim * physical_dim)
        )

        self.output_projection = nn.Linear(bond_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_length)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        batch_size = x.shape[0]

        # Split input into sites: (batch, num_sites, site_dim)
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

        # Contract MPS from left to right
        # Start with left boundary
        # core_0: (1, p, D), phi_0: (batch, p)
        # Result: (batch, 1, D) -> (batch, D)
        core = self.cores[0]  # (1, p, D)
        phi = features[0]  # (batch, p)

        # Einsum: 'ipd,bp->bid' then squeeze i
        # (batch, 1, D) from (1, p, D) and (batch, p)
        result = torch.einsum("ipd,bp->bid", core, phi)  # (batch, 1, D)
        result = result.squeeze(1)  # (batch, D)
        if self.normalize_state:
            result = result / (result.norm(dim=1, keepdim=True) + self.state_eps)

        # Contract bulk cores
        for i in range(1, self.num_sites):
            core = self.cores[i]  # (D, p, D) or (D, p, D) for last
            phi = features[i]  # (batch, p)

            # Einsum: 'bd,dpD,bp->bD'
            # result: (batch, D_left)
            # core: (D_left, p, D_right)
            # phi: (batch, p)
            # Output: (batch, D_right)
            result = torch.einsum("bd,dpD,bp->bD", result, core, phi)
            if self.normalize_state:
                result = result / (result.norm(dim=1, keepdim=True) + self.state_eps)

        # Project to embedding dimension
        embedding = self.output_projection(result)

        return embedding


# ============================================================
# MPS ENCODER SIMPLE PIPELINE
#
# Input:
#   x: (batch, L)
#
# 1) Split into sites:
#   x -> (batch, S, site_dim)
#
# 2) Shared feature map per site:
#   phi_i = feature_map(x[:, i, :]) -> (batch, p)
#
# 3) Initialize hidden state from first site:
#   hidden = W_left(phi_0) -> (batch, D)
#
# 4) Bulk transfer for i = 1..S-2:
#   transferred = einsum("bd,pde->bpe", hidden, W_bulk)  -> (batch, p, D)
#   hidden      = einsum("bp,bpd->bd", phi_i, transferred)-> (batch, D)
#
# 5) Last site output:
#   phi_last = feature_map(last_site) -> (batch, p)
#   combined = einsum("bd,bp->bdp", hidden, phi_last) -> (batch, D, p)
#   flatten  -> (batch, D*p)
#   embedding = W_right(combined) -> (batch, E)
#
# Output:
#   embedding: (batch, embedding_dim)
# ============================================================
class MPSEncoderSimple(nn.Module):  # type: ignore[misc]
    """
    Simplified MPS encoder using matrix multiplications.

    This version is more readable and uses explicit weight matrices
    instead of tensor cores. Mathematically equivalent but clearer.
    """

    def __init__(
        self,
        input_length: int = 512,
        num_sites: int = 64,
        physical_dim: int = 4,
        bond_dim: int = 16,
        embedding_dim: int = 128,
    ):
        """
        Initialize simplified MPS encoder.

        Args:
            input_length: Length of input spectrum
            num_sites: Number of sites to split input into
            physical_dim: Feature dimension at each site
            bond_dim: Width of hidden state (bond dimension)
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.input_length = input_length
        self.num_sites = num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.embedding_dim = embedding_dim

        assert input_length % num_sites == 0
        self.site_dim = input_length // num_sites

        # Shared local feature map
        self.feature_map = nn.Sequential(
            nn.Linear(self.site_dim, physical_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(physical_dim * 2, physical_dim),
        )

        # MPS transfer matrices
        # For each physical state p, we have a D x D transfer matrix
        # T_i(p) of shape (physical_dim, bond_dim, bond_dim)
        # Using a bilinear form: T(phi) = sum_p phi_p * T_p

        # Left boundary: maps physical -> bond
        self.W_left = nn.Linear(physical_dim, bond_dim)

        # Bulk: bilinear transfer
        # W_bulk: (physical_dim, bond_dim, bond_dim)
        self.W_bulk = nn.Parameter(
            torch.randn(physical_dim, bond_dim, bond_dim) / math.sqrt(physical_dim * bond_dim)
        )

        # Right boundary: maps bond -> embedding
        self.W_right = nn.Linear(bond_dim * physical_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_length)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        batch_size = x.shape[0]

        # Split into sites
        x = x.view(batch_size, self.num_sites, self.site_dim)

        # First site: initialize hidden state
        phi_0 = self.feature_map(x[:, 0, :])  # (batch, physical_dim)
        hidden = self.W_left(phi_0)  # (batch, bond_dim)

        # Bulk sites: apply transfer operation
        for i in range(1, self.num_sites - 1):
            phi = self.feature_map(x[:, i, :])  # (batch, physical_dim)

            # Bilinear transfer: hidden' = sum_p phi_p * (hidden @ W_bulk[p])
            # W_bulk: (p, D, D)
            # hidden: (batch, D)
            # phi: (batch, p)

            # Efficient implementation:
            # (batch, D) @ (p, D, D) -> (batch, p, D)
            # then (batch, p) . (batch, p, D) -> (batch, D)
            transferred = torch.einsum("bd,pde->bpe", hidden, self.W_bulk)
            hidden = torch.einsum("bp,bpd->bd", phi, transferred)

        # Last site: produce output
        phi_last = self.feature_map(x[:, -1, :])  # (batch, physical_dim)

        # Combine hidden and phi_last for output
        combined = torch.einsum("bd,bp->bdp", hidden, phi_last)  # (batch, D, p)
        combined = combined.view(batch_size, -1)  # (batch, D * p)
        embedding = self.W_right(combined)

        return embedding
