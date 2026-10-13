import torch
import torch.nn as nn


class MPSEncoder(nn.Module):
    """
    Bidirectional MPS encoder with forward and backward contractions.

    Implements a trainable Matrix Product State that contracts features sequentially
    in both forward and backward directions, then combines them for a rich
    representation.

    Args:
        num_sites: Number of sites in the MPS chain (default: 36)
        physical_dim: Physical dimension at each site (default: 8)
        bond_dim: Bond dimension connecting MPS cores (default: 16)
        output_dim: Dimension of final embedding (default: 128)
        eps: Small constant for numerical stability (default: 1e-6)
    """

    def __init__(
        self,
        num_sites: int = 36,
        physical_dim: int = 8,
        bond_dim: int = 16,
        output_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_sites = num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.output_dim = output_dim
        self.eps = eps

        # Initialize MPS cores for forward pass
        self.forward_cores = nn.ParameterList()

        # First core: (1, physical_dim, bond_dim)
        first_core = torch.empty(1, physical_dim, bond_dim)
        nn.init.xavier_uniform_(first_core)
        self.forward_cores.append(nn.Parameter(first_core))

        # Remaining cores: (bond_dim, physical_dim, bond_dim)
        for _ in range(num_sites - 1):
            core = torch.empty(bond_dim, physical_dim, bond_dim)
            nn.init.xavier_uniform_(core)
            self.forward_cores.append(nn.Parameter(core))

        # Initialize MPS cores for backward pass
        self.backward_cores = nn.ParameterList()

        # First core for backward: (1, physical_dim, bond_dim)
        first_core_back = torch.empty(1, physical_dim, bond_dim)
        nn.init.xavier_uniform_(first_core_back)
        self.backward_cores.append(nn.Parameter(first_core_back))

        # Remaining cores for backward: (bond_dim, physical_dim, bond_dim)
        for _ in range(num_sites - 1):
            core = torch.empty(bond_dim, physical_dim, bond_dim)
            nn.init.xavier_uniform_(core)
            self.backward_cores.append(nn.Parameter(core))

        # Output projection
        self.output_norm = nn.LayerNorm(2 * bond_dim)
        self.output_proj = nn.Linear(2 * bond_dim, output_dim)

    def _contract_forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Perform forward contraction of MPS with features.

        Args:
            features: Tensor of shape (batch_size, num_sites, physical_dim)

        Returns:
            Final state of shape (batch_size, bond_dim)
        """
        batch_size = features.size(0)

        # Initialize state with first core
        # features[:, 0]: (batch_size, physical_dim)
        # forward_cores[0]: (1, physical_dim, bond_dim)
        # Result: (batch_size, bond_dim)
        state = torch.einsum("bp,ipj->bj", features[:, 0], self.forward_cores[0])

        # Normalize
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)

        # Contract remaining sites
        for i in range(1, self.num_sites):
            # state: (batch_size, bond_dim)
            # features[:, i]: (batch_size, physical_dim)
            # forward_cores[i]: (bond_dim, physical_dim, bond_dim)
            # Result: (batch_size, bond_dim)
            state = torch.einsum("bi,ipj,bp->bj", state, self.forward_cores[i], features[:, i])

            # Normalize
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)

        return state

    def _contract_backward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Perform backward contraction of MPS with reversed features.

        Args:
            features: Tensor of shape (batch_size, num_sites, physical_dim)

        Returns:
            Final state of shape (batch_size, bond_dim)
        """
        batch_size = features.size(0)

        # Reverse features
        features_rev = torch.flip(features, dims=[1])

        # Initialize state with first core
        state = torch.einsum("bp,ipj->bj", features_rev[:, 0], self.backward_cores[0])

        # Normalize
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)

        # Contract remaining sites
        for i in range(1, self.num_sites):
            state = torch.einsum("bi,ipj,bp->bj", state, self.backward_cores[i], features_rev[:, i])

            # Normalize
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)

        return state

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features using bidirectional MPS.

        Args:
            features: Tensor of shape (batch_size, num_sites, physical_dim)

        Returns:
            Embedding of shape (batch_size, output_dim)
        """
        # Perform forward and backward contractions
        forward_state = self._contract_forward(features)
        backward_state = self._contract_backward(features)

        # Concatenate states
        combined = torch.cat([forward_state, backward_state], dim=1)

        # Project to output dimension
        embedding = self.output_norm(combined)
        embedding = self.output_proj(embedding)

        return embedding


class MPSFunctionalGroupClassifier(nn.Module):
    """
    Complete MPS-based multi-label classifier for functional group prediction.

    This model processes IR spectra through a pure tensor network approach:
    1. Splits input into sites
    2. Encodes the raw site values with bidirectional MPS
    3. Classifies functional groups

    Args:
        input_dim: Length of input spectrum (default: 1800)
        num_sites: Number of sites to split input into (default: 36)
        physical_dim: Deprecated compatibility argument. When provided, it must
            match the raw site dimension because split spectra are passed
            directly to the MPS.
        bond_dim: Bond dimension of MPS (default: 16)
        num_classes: Number of functional groups to predict (default: 37)
        dropout_rate: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        input_dim: int = 1800,
        num_sites: int = 36,
        physical_dim: int | None = None,
        bond_dim: int = 16,
        num_classes: int = 37,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # Validate input dimensions
        assert (
            input_dim % num_sites == 0
        ), f"input_dim ({input_dim}) must be divisible by num_sites ({num_sites})"

        self.input_dim = input_dim
        self.num_sites = num_sites
        self.site_dim = input_dim // num_sites
        self.physical_dim = self.site_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes

        if physical_dim is not None and physical_dim != self.site_dim:
            raise ValueError(
                "Direct MPS input requires physical_dim to match the raw site dimension. "
                f"Expected {self.site_dim}, got {physical_dim}."
            )

        # MPS encoder
        self.mps_encoder = MPSEncoder(
            num_sites=num_sites,
            physical_dim=self.site_dim,
            bond_dim=bond_dim,
            output_dim=128,
        )

        # Classifier head
        self.classifier_norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 64)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MPS classifier.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes) for BCEWithLogitsLoss
        """
        batch_size = x.size(0)

        # Reshape input into sites: (batch_size, num_sites, site_dim)
        features = x.view(batch_size, self.num_sites, self.site_dim)

        # Encode with MPS
        embedding = self.mps_encoder(features)

        # Classify
        x = self.classifier_norm(embedding)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
