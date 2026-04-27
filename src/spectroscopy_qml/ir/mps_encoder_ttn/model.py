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

    This model processes IR spectra through a pure tensor network approach:
    1. Splits input into sites
    2. Applies local feature maps
    3. Encodes with bidirectional MPS
    4. Classifies functional groups

    Args:
        input_dim: Length of input spectrum (default: 1800)
        num_sites: Number of sites to split input into (default: 36)
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

        # Validate input dimensions
        assert (
            input_dim % num_sites == 0
        ), f"input_dim ({input_dim}) must be divisible by num_sites ({num_sites})"

        self.input_dim = input_dim
        self.num_sites = num_sites
        self.site_dim = input_dim // num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes

        # Local feature map (shared across all sites)
        self.feature_map = LocalFeatureMap(
            site_dim=self.site_dim, physical_dim=physical_dim, dropout_rate=dropout_rate
        )

        # MPS encoder
        mps_output_dim = 2 * num_sites * bond_dim  # intermediate size before projection
        self.mps_encoder = MPSEncoder(
            num_sites=num_sites, physical_dim=physical_dim, bond_dim=bond_dim, output_dim=256
        )

        # Classifier head
        self.classifier_norm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 128)
        self.fc_mid = nn.Linear(128, 64)
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
        x = x.view(batch_size, self.num_sites, self.site_dim)

        # Apply local feature map to each site
        # Process all sites in parallel by reshaping
        x_flat = x.view(batch_size * self.num_sites, self.site_dim)
        features_flat = self.feature_map(x_flat)
        features = features_flat.view(batch_size, self.num_sites, self.physical_dim)

        # Encode with MPS
        embedding = self.mps_encoder(features)

        # Classify
        x = self.classifier_norm(embedding)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc_mid(x)
        x = self.gelu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TTNNode(nn.Module):
    """
    A single node in the Tensor Tree Network.

    Performs a factored bilinear contraction of two child vectors using
    Hadamard product of projections for parameter efficiency.

    Args:
        left_dim: Dimension of the left child vector
        right_dim: Dimension of the right child vector
        output_dim: Dimension of the output vector
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.left_proj = nn.Linear(left_dim, output_dim, bias=False)
        self.right_proj = nn.Linear(right_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Contract left and right child vectors via factored bilinear map.

        Args:
            left: (batch_size, left_dim)
            right: (batch_size, right_dim)

        Returns:
            Contracted output of shape (batch_size, output_dim)
        """
        l = self.left_proj(left)
        r = self.right_proj(right)
        x = l * r  # Hadamard product (factored bilinear interaction)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        return x


class TTNLayer(nn.Module):
    """
    One level of the Tensor Tree Network.

    Pairs up adjacent nodes and contracts them. If there is an odd number
    of inputs, the last one is projected independently.

    Args:
        num_inputs: Number of input nodes at this level
        input_dim: Dimension of each input vector
        output_dim: Dimension of each output vector
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        num_inputs: int,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_pairs = num_inputs // 2
        self.has_remainder = num_inputs % 2 == 1

        self.nodes = nn.ModuleList(
            [
                TTNNode(input_dim, input_dim, output_dim, dropout_rate)
                for _ in range(self.num_pairs)
            ]
        )

        if self.has_remainder:
            self.remainder_proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Contract pairs of adjacent feature vectors.

        Args:
            features: List of num_inputs tensors, each (batch_size, input_dim)

        Returns:
            List of ceil(num_inputs/2) tensors, each (batch_size, output_dim)
        """
        outputs = []
        for i in range(self.num_pairs):
            left = features[2 * i]
            right = features[2 * i + 1]
            outputs.append(self.nodes[i](left, right))

        if self.has_remainder:
            outputs.append(self.remainder_proj(features[-1]))

        return outputs


class TensorTreeNetwork(nn.Module):
    """
    Full Tensor Tree Network (TTN) classifier.

    Hierarchically contracts a sequence of feature vectors through a binary
    tree structure, producing a single root representation for classification.

    Args:
        num_leaves: Number of input leaf nodes
        leaf_dim: Dimension of each leaf vector
        internal_dim: Dimension of internal node vectors
        num_classes: Number of output classes
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        num_leaves: int,
        leaf_dim: int,
        internal_dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_leaves = num_leaves
        self.leaf_dim = leaf_dim
        self.internal_dim = internal_dim
        self.num_classes = num_classes

        # Leaf projection if input dim differs from internal dim
        if leaf_dim != internal_dim:
            self.leaf_proj = nn.Linear(leaf_dim, internal_dim)
        else:
            self.leaf_proj = nn.Identity()

        # Build tree layers
        self.layers = nn.ModuleList()
        current_num = num_leaves

        while current_num > 1:
            layer = TTNLayer(current_num, internal_dim, internal_dim, dropout_rate)
            self.layers.append(layer)
            current_num = (current_num + 1) // 2  # ceil division

        # Root classification head
        self.root_norm = nn.LayerNorm(internal_dim)
        self.classifier = nn.Linear(internal_dim, num_classes)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Hierarchically contract features and classify.

        Args:
            features: List of num_leaves tensors, each (batch_size, leaf_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Project leaves to internal dimension
        current = [self.leaf_proj(f) for f in features]

        # Contract through tree
        for layer in self.layers:
            current = layer(current)

        # Root → classification
        root = current[0]
        root = self.root_norm(root)
        logits = self.classifier(root)

        return logits


class MPSTTNClassifier(nn.Module):
    """
    MPS Encoder + Tensor Tree Network classifier.

    Uses the MPS encoder to produce per-site bidirectional bond vectors,
    then applies a TTN to hierarchically combine them for classification.

    The MPS encoder can optionally be frozen to train only the TTN on top
    of pre-trained encoder features.

    Args:
        input_dim: Length of input spectrum (default: 1800)
        num_sites: Number of sites to split input into (default: 36)
        physical_dim: Dimension of physical indices in MPS (default: 8)
        bond_dim: Bond dimension of MPS (default: 16)
        ttn_internal_dim: Internal dimension of TTN nodes (default: 128)
        num_classes: Number of functional groups to predict (default: 37)
        dropout_rate: Dropout probability (default: 0.2)
        freeze_encoder: Whether to freeze the MPS encoder (default: False)
    """

    def __init__(
        self,
        input_dim: int = 1800,
        num_sites: int = 36,
        physical_dim: int = 8,
        bond_dim: int = 16,
        ttn_internal_dim: int = 128,
        num_classes: int = 37,
        dropout_rate: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        assert (
            input_dim % num_sites == 0
        ), f"input_dim ({input_dim}) must be divisible by num_sites ({num_sites})"

        self.input_dim = input_dim
        self.num_sites = num_sites
        self.site_dim = input_dim // num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

        # Local feature map (shared across all sites)
        self.feature_map = LocalFeatureMap(
            site_dim=self.site_dim,
            physical_dim=physical_dim,
            dropout_rate=dropout_rate,
        )

        # MPS encoder (architecture kept as is, output_dim=256 for weight compatibility)
        self.mps_encoder = MPSEncoder(
            num_sites=num_sites,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            output_dim=256,
        )

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.feature_map.parameters():
                param.requires_grad = False
            for param in self.mps_encoder.parameters():
                param.requires_grad = False

        # Per-site feature dimension: forward bond + backward bond
        site_feature_dim = 2 * bond_dim

        # TTN classifier replaces the FC head
        self.ttn = TensorTreeNetwork(
            num_leaves=num_sites,
            leaf_dim=site_feature_dim,
            internal_dim=ttn_internal_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    def load_encoder_weights(self, checkpoint_path: str, device: torch.device | None = None):
        """
        Load pre-trained MPS encoder weights from a checkpoint.

        Loads only the feature_map and mps_encoder parameters, leaving the
        TTN weights untouched.

        Args:
            checkpoint_path: Path to the saved checkpoint (.pt file)
            device: Device to load weights to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        own_state = self.state_dict()
        loaded = 0
        for name, param in state_dict.items():
            if name.startswith(("feature_map.", "mps_encoder.")):
                if name in own_state:
                    own_state[name].copy_(param)
                    loaded += 1

        print(f"Loaded {loaded} encoder parameter tensors from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: feature map → MPS contraction → TTN classification.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Reshape input into sites: (batch_size, num_sites, site_dim)
        x = x.view(batch_size, self.num_sites, self.site_dim)

        # Apply local feature map to each site
        x_flat = x.view(batch_size * self.num_sites, self.site_dim)
        features_flat = self.feature_map(x_flat)
        features = features_flat.view(batch_size, self.num_sites, self.physical_dim)

        # Get per-site MPS bond vectors using encoder's contraction methods
        if self.freeze_encoder:
            with torch.no_grad():
                fwd_states = self.mps_encoder._contract_forward(features)
                bwd_states = self.mps_encoder._contract_backward(features)
        else:
            fwd_states = self.mps_encoder._contract_forward(features)
            bwd_states = self.mps_encoder._contract_backward(features)

        # Align backward states with forward sites
        # backward_states[0] corresponds to last site, so reverse
        bwd_states_aligned = list(reversed(bwd_states))

        # Combine forward and backward bond vectors per site
        site_features = [
            torch.cat([fwd_states[i], bwd_states_aligned[i]], dim=1)
            for i in range(self.num_sites)
        ]

        # Apply TTN classifier
        logits = self.ttn(site_features)

        return logits

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _BatchedTTNLevel(nn.Module):
    """
    Single level of a weight-tied TTN processed in parallel.

    All node pairs within this level share the same projection weights,
    enabling efficient batched computation for large numbers of leaves.

    Args:
        input_dim: Dimension of each input node vector
        output_dim: Dimension of each output node vector
        dropout_rate: Dropout probability
    """

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.left_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.right_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Contract adjacent pairs in parallel.

        Args:
            x: (batch, num_nodes, input_dim)

        Returns:
            (batch, ceil(num_nodes/2), output_dim)
        """
        b, n, d = x.shape
        pairs = n // 2
        has_remainder = n % 2 == 1

        parts = []

        if pairs > 0:
            paired = x[:, : 2 * pairs, :].reshape(b, pairs, 2, d)
            left = paired[:, :, 0, :].reshape(b * pairs, d)
            right = paired[:, :, 1, :].reshape(b * pairs, d)

            l = self.left_proj(left)
            r = self.right_proj(right)
            h = l * r
            h = self.norm(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.out_proj(h)
            parts.append(h.reshape(b, pairs, -1))

        if has_remainder:
            rem = self.left_proj(x[:, -1, :])  # (batch, output_dim)
            parts.append(rem.unsqueeze(1))

        return torch.cat(parts, dim=1)


class PerSiteTTN(nn.Module):
    """
    Efficient weight-tied TTN for scalar leaf inputs.

    Takes a vector and treats each element as a scalar leaf node,
    then hierarchically contracts pairs through a binary tree to a
    single root vector.  All nodes within a level share weights.

    Args:
        num_leaves: Number of scalar leaf inputs (e.g. 256)
        internal_dim: Dimension of internal node representations
        dropout_rate: Dropout probability
    """

    def __init__(self, num_leaves: int, internal_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.num_leaves = num_leaves
        self.internal_dim = internal_dim

        self.levels = nn.ModuleList()
        current = num_leaves
        is_first = True

        while current > 1:
            in_dim = 1 if is_first else internal_dim
            self.levels.append(_BatchedTTNLevel(in_dim, internal_dim, dropout_rate))
            current = (current + 1) // 2
            is_first = False

        self.root_norm = nn.LayerNorm(internal_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_leaves) — scalar values at each leaf

        Returns:
            (batch, internal_dim) — root vector
        """
        current = x.unsqueeze(-1)  # (batch, num_leaves, 1)

        for level in self.levels:
            current = level(current)

        return self.root_norm(current.squeeze(1))  # (batch, internal_dim)


class TTNEncoder(nn.Module):
    """
    TTN encoder that contracts leaf vectors to a single root vector.

    Unlike ``TensorTreeNetwork`` this has no classification head — it
    returns the root embedding directly.  Uses per-node independent
    weights (``TTNLayer`` / ``TTNNode``), which is suitable for small
    numbers of leaves (e.g. the cross-site tree with 5 leaves).

    Args:
        num_leaves: Number of input leaf vectors
        leaf_dim: Dimension of each leaf vector
        internal_dim: Internal node dimension
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        num_leaves: int,
        leaf_dim: int,
        internal_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.internal_dim = internal_dim

        if leaf_dim != internal_dim:
            self.leaf_proj = nn.Linear(leaf_dim, internal_dim)
        else:
            self.leaf_proj = nn.Identity()

        self.layers = nn.ModuleList()
        current = num_leaves
        while current > 1:
            self.layers.append(TTNLayer(current, internal_dim, internal_dim, dropout_rate))
            current = (current + 1) // 2

        self.root_norm = nn.LayerNorm(internal_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of num_leaves tensors, each (batch, leaf_dim)

        Returns:
            Root vector of shape (batch, internal_dim)
        """
        current = [self.leaf_proj(f) for f in features]
        for layer in self.layers:
            current = layer(current)
        return self.root_norm(current[0])


class MPSTTNHybridClassifier(nn.Module):
    """
    MPS Encoder + per-site TTNs + cross-site TTN hybrid classifier.

    Architecture (6 trees total):
        1. Feature map splits the spectrum into sites
        2. MPS encoder produces bidirectional bond vectors per site
        3. **5 per-site TTNs**: each site's bond vector (2 × bond_dim scalars)
           is contracted through a weight-tied binary tree → root vector
        4. **1 cross-site TTN**: the full per-site bond vectors are contracted
           across sites → root vector capturing inter-site correlations
        5. All 6 root vectors are concatenated → linear → logits

    Args:
        input_dim: Length of input spectrum (default: 1800)
        num_sites: Number of sites (default: 5)
        physical_dim: MPS physical dimension (default: 450)
        bond_dim: MPS bond dimension (default: 128)
        ttn_internal_dim: Cross-site TTN internal dim (default: 128)
        per_site_internal_dim: Per-site TTN internal node dim (default: 16)
        num_classes: Number of output classes (default: 37)
        dropout_rate: Dropout probability (default: 0.0)
        freeze_encoder: Freeze feature map and MPS encoder (default: False)
        share_site_trees: Share weights across the 5 per-site trees (default: True)
    """

    def __init__(
        self,
        input_dim: int = 1800,
        num_sites: int = 5,
        physical_dim: int = 450,
        bond_dim: int = 128,
        ttn_internal_dim: int = 128,
        per_site_internal_dim: int = 16,
        num_classes: int = 37,
        dropout_rate: float = 0.0,
        freeze_encoder: bool = False,
        share_site_trees: bool = True,
    ):
        super().__init__()

        assert (
            input_dim % num_sites == 0
        ), f"input_dim ({input_dim}) must be divisible by num_sites ({num_sites})"

        self.input_dim = input_dim
        self.num_sites = num_sites
        self.site_dim = input_dim // num_sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.share_site_trees = share_site_trees

        # Local feature map (shared across all sites)
        self.feature_map = LocalFeatureMap(
            site_dim=self.site_dim,
            physical_dim=physical_dim,
            dropout_rate=dropout_rate,
        )

        # MPS encoder
        self.mps_encoder = MPSEncoder(
            num_sites=num_sites,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            output_dim=256,
        )

        if freeze_encoder:
            for p in self.feature_map.parameters():
                p.requires_grad = False
            for p in self.mps_encoder.parameters():
                p.requires_grad = False

        site_feature_dim = 2 * bond_dim  # forward + backward bond vectors

        # --- Per-site trees (5 trees) ---
        if share_site_trees:
            self.per_site_ttn = PerSiteTTN(
                num_leaves=site_feature_dim,
                internal_dim=per_site_internal_dim,
                dropout_rate=dropout_rate,
            )
        else:
            self.per_site_ttns = nn.ModuleList(
                [
                    PerSiteTTN(
                        num_leaves=site_feature_dim,
                        internal_dim=per_site_internal_dim,
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(num_sites)
                ]
            )

        # --- Cross-site tree (1 tree) ---
        self.cross_site_ttn = TTNEncoder(
            num_leaves=num_sites,
            leaf_dim=site_feature_dim,
            internal_dim=ttn_internal_dim,
            dropout_rate=dropout_rate,
        )

        # --- Final classifier ---
        combined_dim = num_sites * per_site_internal_dim + ttn_internal_dim
        self.classifier_norm = nn.LayerNorm(combined_dim)
        self.classifier = nn.Linear(combined_dim, num_classes)

    def load_encoder_weights(self, checkpoint_path: str, device: torch.device | None = None):
        """Load pre-trained MPS encoder weights (feature_map + mps_encoder only)."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        own_state = self.state_dict()
        loaded = 0
        for name, param in state_dict.items():
            if name.startswith(("feature_map.", "mps_encoder.")):
                if name in own_state:
                    own_state[name].copy_(param)
                    loaded += 1

        print(f"Loaded {loaded} encoder parameter tensors from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: feature map → MPS → per-site TTNs + cross-site TTN → logits.

        Args:
            x: (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Split into sites and apply feature map
        x = x.view(batch_size, self.num_sites, self.site_dim)
        x_flat = x.view(batch_size * self.num_sites, self.site_dim)
        features_flat = self.feature_map(x_flat)
        features = features_flat.view(batch_size, self.num_sites, self.physical_dim)

        # MPS contraction → per-site bond vectors
        if self.freeze_encoder:
            with torch.no_grad():
                fwd = self.mps_encoder._contract_forward(features)
                bwd = self.mps_encoder._contract_backward(features)
        else:
            fwd = self.mps_encoder._contract_forward(features)
            bwd = self.mps_encoder._contract_backward(features)

        bwd_aligned = list(reversed(bwd))

        # Per-site bond vectors: list of (batch, 2*bond_dim)
        site_features = [
            torch.cat([fwd[i], bwd_aligned[i]], dim=1) for i in range(self.num_sites)
        ]

        # --- Per-site trees: each (batch, 2*bond_dim) → (batch, per_site_internal_dim) ---
        per_site_roots = []
        for i in range(self.num_sites):
            if self.share_site_trees:
                root = self.per_site_ttn(site_features[i])
            else:
                root = self.per_site_ttns[i](site_features[i])
            per_site_roots.append(root)

        # --- Cross-site tree: 5 leaves → (batch, ttn_internal_dim) ---
        cross_root = self.cross_site_ttn(site_features)

        # Combine all 6 roots and classify
        combined = torch.cat(per_site_roots + [cross_root], dim=1)
        combined = self.classifier_norm(combined)
        logits = self.classifier(combined)

        return logits

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
