"""
Pure MPS/Tensor Network Classifier for IR Spectra.

This module implements a multi-label classifier based on Matrix Product States (MPS)
for predicting functional groups from IR spectroscopy data.
"""

import torch
import torch.nn as nn


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

        # Output projection: all site states from both directions
        # Total features = 2 * num_sites * bond_dim
        total_features = 2 * num_sites * bond_dim
        self.output_norm = nn.LayerNorm(total_features)
        self.output_proj = nn.Linear(total_features, output_dim)

    def _contract_forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        """
        Perform forward contraction of MPS with features, collecting
        the bond vector at every site.

        Args:
            features: Tensor of shape (batch_size, num_sites, physical_dim)

        Returns:
            List of num_sites tensors, each of shape (batch_size, bond_dim)
        """
        states = []

        # Initialize state with first core
        state = torch.einsum("bp,ipj->bj", features[:, 0], self.forward_cores[0])
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
        states.append(state)

        # Contract remaining sites, keeping every intermediate state
        for i in range(1, self.num_sites):
            state = torch.einsum("bi,ipj,bp->bj", state, self.forward_cores[i], features[:, i])
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
            states.append(state)

        return states

    def _contract_backward(self, features: torch.Tensor) -> list[torch.Tensor]:
        """
        Perform backward contraction of MPS with reversed features, collecting
        the bond vector at every site.

        Args:
            features: Tensor of shape (batch_size, num_sites, physical_dim)

        Returns:
            List of num_sites tensors, each of shape (batch_size, bond_dim)
        """
        states = []
        features_rev = torch.flip(features, dims=[1])

        # Initialize state with first core
        state = torch.einsum("bp,ipj->bj", features_rev[:, 0], self.backward_cores[0])
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
        states.append(state)

        # Contract remaining sites, keeping every intermediate state
        for i in range(1, self.num_sites):
            state = torch.einsum("bi,ipj,bp->bj", state, self.backward_cores[i], features_rev[:, i])
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
            states.append(state)

        return states

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features using bidirectional MPS.

        Collects the bond vector after every core in both directions
        and concatenates all of them.

        Args:
            features: Tensor of shape (batch_size, num_sites, physical_dim)

        Returns:
            Embedding of shape (batch_size, output_dim)
        """
        # Collect all intermediate states from both directions
        forward_states = self._contract_forward(features)    # num_sites × (batch, bond_dim)
        backward_states = self._contract_backward(features)  # num_sites × (batch, bond_dim)

        # Concatenate all states: (batch, 2 * num_sites * bond_dim)
        combined = torch.cat(forward_states + backward_states, dim=1)

        # Project to output dimension
        embedding = self.output_norm(combined)
        embedding = self.output_proj(embedding)

        return embedding


class MPSFunctionalGroupClassifier(nn.Module):
    """
    Complete MPS-based multi-label classifier for functional group prediction.

    This model processes IR spectra through a dual tensor network approach:
    1. Splits input into sites (coarse-grained and fine-grained)
    2. Applies local feature maps to each
    3. Encodes with two bidirectional MPS encoders
    4. Combines both embeddings and classifies functional groups

    Args:
        input_dim: Length of input spectrum (default: 1800)
        num_sites: Number of sites for the first (coarse) MPS (default: 36)
        physical_dim: Dimension of physical indices in first MPS (default: 8)
        bond_dim: Bond dimension of first MPS (default: 16)
        num_classes: Number of functional groups to predict (default: 37)
        dropout_rate: Dropout probability (default: 0.2)
        classifier_head: "cnn" for MPS+CNN hybrid, "mps" for pure MPS classifier (default: "mps")
        num_sites_2: Number of sites for the second (fine) MPS (default: 1800)
        physical_dim_2: Dimension of physical indices in second MPS (default: 8)
        bond_dim_2: Bond dimension of second MPS (default: 16)
    """

    def __init__(
        self,
        input_dim: int = 1800,
        num_sites: int = 36,
        physical_dim: int = 8,
        bond_dim: int = 16,
        num_classes: int = 37,
        dropout_rate: float = 0.2,
        classifier_head: str = "mps",
        num_sites_2: int = 1800,
        physical_dim_2: int = 8,
        bond_dim_2: int = 16,
    ):
        super().__init__()

        # Validate input dimensions
        assert (
            input_dim % num_sites == 0
        ), f"input_dim ({input_dim}) must be divisible by num_sites ({num_sites})"
        assert (
            input_dim % num_sites_2 == 0
        ), f"input_dim ({input_dim}) must be divisible by num_sites_2 ({num_sites_2})"

        assert classifier_head in ("cnn", "mps"), (
            f"classifier_head must be 'cnn' or 'mps', got '{classifier_head}'"
        )

        self.input_dim = input_dim
        self.num_sites = num_sites
        self.site_dim = input_dim // num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.classifier_head = classifier_head

        # Second MPS config
        self.num_sites_2 = num_sites_2
        self.site_dim_2 = input_dim // num_sites_2
        self.physical_dim_2 = physical_dim_2
        self.bond_dim_2 = bond_dim_2

        # Local feature maps (separate for each MPS since site_dim may differ)
        self.feature_map = LocalFeatureMap(
            site_dim=self.site_dim, physical_dim=physical_dim, dropout_rate=dropout_rate
        )
        self.feature_map_2 = LocalFeatureMap(
            site_dim=self.site_dim_2, physical_dim=physical_dim_2, dropout_rate=dropout_rate
        )

        # Intermediate embedding dimension for each MPS encoder
        intermediate_dim = 128

        if classifier_head == "cnn":
            # MPS encoder produces an intermediate embedding for the CNN
            self.mps_encoder = MPSEncoder(
                num_sites=num_sites, physical_dim=physical_dim, bond_dim=bond_dim, output_dim=256
            )
            self.mps_encoder_2 = MPSEncoder(
                num_sites=num_sites_2, physical_dim=physical_dim_2, bond_dim=bond_dim_2, output_dim=256
            )

            # CNN classifier head operating on per-site MPS features + raw spectrum
            cnn_in_channels = 2 * bond_dim + self.site_dim  # MPS bond vectors + raw site values

            # 1st CNN layer
            self.conv1 = nn.Conv1d(cnn_in_channels, 31, kernel_size=11, stride=1, padding="same")
            self.bn1 = nn.BatchNorm1d(31)
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

            # 2nd CNN layer
            self.conv2 = nn.Conv1d(31, 62, kernel_size=11, stride=1, padding="same")
            self.bn2 = nn.BatchNorm1d(62)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

            # Compute flattened size after conv layers
            conv_out_length = num_sites // 2 // 2  # after two MaxPool1d(2, 2)
            flat_size = 62 * conv_out_length

            # Dense layers (input includes both MPS encoder embeddings)
            self.fc1 = nn.Linear(flat_size + 256, 4927)
            self.fc2 = nn.Linear(4927, 2785)
            self.fc3 = nn.Linear(2785, 1574)
            self.fc_out = nn.Linear(1574, num_classes)

            self.cnn_dropout = nn.Dropout(0.48599073736368)
            self.relu = nn.ReLU()

        else:  # "mps" — pure MPS classifier
            self.mps_encoder = MPSEncoder(
                num_sites=num_sites, physical_dim=physical_dim, bond_dim=bond_dim,
                output_dim=intermediate_dim,
            )
            self.mps_encoder_2 = MPSEncoder(
                num_sites=num_sites_2, physical_dim=physical_dim_2, bond_dim=bond_dim_2,
                output_dim=intermediate_dim,
            )
            combined_dim = 2 * intermediate_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Linear(combined_dim, 256),
                nn.GELU(),
                nn.Dropout(0.49),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes) for BCEWithLogitsLoss
        """
        batch_size = x.size(0)

        # === First MPS (coarse-grained) ===
        x_sites = x.view(batch_size, self.num_sites, self.site_dim)
        x_flat = x_sites.view(batch_size * self.num_sites, self.site_dim)
        features_flat = self.feature_map(x_flat)
        features = features_flat.view(batch_size, self.num_sites, self.physical_dim)

        # === Second MPS (fine-grained) ===
        x_sites_2 = x.view(batch_size, self.num_sites_2, self.site_dim_2)
        x_flat_2 = x_sites_2.view(batch_size * self.num_sites_2, self.site_dim_2)
        features_flat_2 = self.feature_map_2(x_flat_2)
        features_2 = features_flat_2.view(batch_size, self.num_sites_2, self.physical_dim_2)

        if self.classifier_head == "mps":
            # Pure MPS path: combine both encoder embeddings → classifier
            with torch.amp.autocast("cuda", enabled=False):
                emb1 = self.mps_encoder(features.float())
                emb2 = self.mps_encoder_2(features_2.float())
            logits = self.classifier(torch.cat([emb1, emb2], dim=1))
            return logits

        # CNN path: use per-site MPS bond vectors + raw site features from first MPS
        with torch.amp.autocast("cuda", enabled=False):
            features_f32 = features.float()
            forward_states = self.mps_encoder._contract_forward(features_f32)
            backward_states = self.mps_encoder._contract_backward(features_f32)

        # Stack per-site features: (batch, num_sites, 2 * bond_dim)
        fw = torch.stack(forward_states, dim=1)
        bw = torch.stack(backward_states, dim=1)
        per_site_mps = torch.cat([fw, bw], dim=2)

        # Concatenate raw site values with MPS features: (batch, num_sites, 2*bond_dim + site_dim)
        per_site = torch.cat([per_site_mps, x_sites], dim=2)

        # Transpose for Conv1D: (batch, channels, length)
        x = per_site.transpose(1, 2)

        # 1st CNN layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 2nd CNN layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Get second MPS embedding and concatenate with CNN features
        with torch.amp.autocast("cuda", enabled=False):
            emb2 = self.mps_encoder_2(features_2.float())
        x = torch.cat([x, emb2], dim=1)

        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.cnn_dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.cnn_dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.cnn_dropout(x)

        logits = self.fc_out(x)
        return logits

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
