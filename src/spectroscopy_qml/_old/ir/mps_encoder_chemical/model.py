import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Predefined spectral chunks (start, end) indices — inclusive.
# Each chunk covers a chemically meaningful region of the IR spectrum.
SPECTRAL_CHUNKS = [
    (0, 154),
    (155, 411),
    (412, 591),
    (592, 771),
    (772, 894),
    (895, 977),
    (978, 1028),
    (1029, 1131),
    (1132, 1208),
    (1209, 1285),
    (1286, 1388),
    (1389, 1542),
    (1543, 1799),
]


class LocalFeatureMap(nn.Module):
    """
    Shared local feature map MLP applied to each site.

    Transforms raw site features into a lower-dimensional physical representation
    suitable for MPS encoding.

    Args:
        site_dim: Dimension of input features at each site (default: 50)
        physical_dim: Output dimension for MPS encoding (default: 8)
        dropout_rate: Dropout probability (default: 0.1)
    """

    def __init__(self, site_dim: int = 50, physical_dim: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.site_dim = site_dim
        self.physical_dim = physical_dim

        # Calculate hidden dimension
        self.hidden_dim = max(site_dim, physical_dim * 2)

        # Build feature map layers
        self.layer_norm = nn.LayerNorm(site_dim)
        self.fc1 = nn.Linear(site_dim, self.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, physical_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply local feature map to site features.

        Args:
            x: Input tensor of shape (batch_size, site_dim)

        Returns:
            Transformed features of shape (batch_size, physical_dim)
        """
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x


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


class ChunkMPSBranch(nn.Module):
    """
    MPS pipeline for a single spectral chunk.

    Pads the chunk to the nearest multiple of *num_sites*, then runs the
    standard site-split → feature-map → MPS-encoder pipeline.

    Args:
        chunk_size: Number of spectral points in this chunk.
        num_sites: Number of sub-sites to split the (padded) chunk into.
        physical_dim: Output dimension of the local feature map.
        bond_dim: Bond dimension of the MPS encoder.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        chunk_size: int,
        num_sites: int,
        physical_dim: int,
        bond_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_sites = num_sites
        self.padded_size = math.ceil(chunk_size / num_sites) * num_sites
        self.site_dim = self.padded_size // num_sites
        self.pad_amount = self.padded_size - chunk_size

        self.feature_map = LocalFeatureMap(
            site_dim=self.site_dim, physical_dim=physical_dim, dropout_rate=dropout_rate
        )
        self.mps_encoder = MPSEncoder(
            num_sites=num_sites, physical_dim=physical_dim, bond_dim=bond_dim, output_dim=128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, chunk_size)

        Returns:
            embedding: (batch_size, 128)
        """
        batch_size = x.size(0)

        # Zero-pad to make length divisible by num_sites
        if self.pad_amount > 0:
            x = F.pad(x, (0, self.pad_amount))

        # Reshape into sites: (batch_size, num_sites, site_dim)
        x = x.reshape(batch_size, self.num_sites, self.site_dim)

        # Apply local feature map to each site
        x_flat = x.reshape(batch_size * self.num_sites, self.site_dim)
        features_flat = self.feature_map(x_flat)
        features = features_flat.reshape(batch_size, self.num_sites, -1)

        # MPS encode
        embedding = self.mps_encoder(features)
        return embedding


class MPSFunctionalGroupClassifier(nn.Module):
    """
    Chunk-based MPS multi-label classifier for functional group prediction.

    The input spectrum is first split into predefined spectral chunks
    (chemically meaningful regions).  Each chunk is independently processed
    through its own MPS pipeline (site-split → feature map → MPS encoder).
    The resulting embeddings are concatenated and fed to a shared classifier
    head.

    Args:
        input_dim: Length of input spectrum (default: 1800)
        num_sites: Number of sub-sites per chunk (default: 36)
        physical_dim: Dimension of physical indices in MPS (default: 8)
        bond_dim: Bond dimension of MPS (default: 16)
        num_classes: Number of functional groups to predict (default: 37)
        dropout_rate: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        input_dim: int = 1800,
        num_sites: int = 36,
        physical_dim: int = 8,
        bond_dim: int = 16,
        num_classes: int = 37,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sites = num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.chunks = SPECTRAL_CHUNKS
        self.num_chunks = len(SPECTRAL_CHUNKS)

        # One MPS branch per spectral chunk
        self.branches = nn.ModuleList()
        for start, end in SPECTRAL_CHUNKS:
            chunk_size = end - start + 1
            self.branches.append(
                ChunkMPSBranch(
                    chunk_size=chunk_size,
                    num_sites=num_sites,
                    physical_dim=physical_dim,
                    bond_dim=bond_dim,
                    dropout_rate=dropout_rate,
                )
            )

        # Classifier head over concatenated chunk embeddings
        total_embedding_dim = self.num_chunks * 128
        self.classifier_norm = nn.LayerNorm(total_embedding_dim)
        self.fc1 = nn.Linear(total_embedding_dim, 64)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the chunk-based MPS classifier.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes) for BCEWithLogitsLoss
        """
        # Process each spectral chunk through its own MPS branch
        embeddings = []
        for i, (start, end) in enumerate(self.chunks):
            chunk = x[:, start : end + 1]
            embeddings.append(self.branches[i](chunk))

        # Concatenate all chunk embeddings
        combined = torch.cat(embeddings, dim=1)  # (batch, num_chunks * 128)

        # Classify
        x = self.classifier_norm(combined)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
