"""
Copied MPS classifier used only by the QCNN hybrid experiment.

This file intentionally duplicates the IR MPS classifier so the hybrid pipeline
can expose latent features without modifying the original baseline module.
"""

import torch
import torch.nn as nn


class LocalFeatureMap(nn.Module):
    def __init__(self, site_dim: int = 50, physical_dim: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.site_dim = site_dim
        self.physical_dim = physical_dim
        self.hidden_dim = max(site_dim, physical_dim * 2)
        self.layer_norm = nn.LayerNorm(site_dim)
        self.fc1 = nn.Linear(site_dim, self.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, physical_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x


class MPSEncoder(nn.Module):
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

        self.forward_cores = nn.ParameterList()
        first_core = torch.empty(1, physical_dim, bond_dim)
        nn.init.xavier_uniform_(first_core)
        self.forward_cores.append(nn.Parameter(first_core))
        for _ in range(num_sites - 1):
            core = torch.empty(bond_dim, physical_dim, bond_dim)
            nn.init.xavier_uniform_(core)
            self.forward_cores.append(nn.Parameter(core))

        self.backward_cores = nn.ParameterList()
        first_core_back = torch.empty(1, physical_dim, bond_dim)
        nn.init.xavier_uniform_(first_core_back)
        self.backward_cores.append(nn.Parameter(first_core_back))
        for _ in range(num_sites - 1):
            core = torch.empty(bond_dim, physical_dim, bond_dim)
            nn.init.xavier_uniform_(core)
            self.backward_cores.append(nn.Parameter(core))

        total_features = 2 * num_sites * bond_dim
        self.output_norm = nn.LayerNorm(total_features)
        self.output_proj = nn.Linear(total_features, output_dim)

    def _contract_forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        states = []
        state = torch.einsum("bp,ipj->bj", features[:, 0], self.forward_cores[0])
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
        states.append(state)

        for i in range(1, self.num_sites):
            state = torch.einsum("bi,ipj,bp->bj", state, self.forward_cores[i], features[:, i])
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
            states.append(state)

        return states

    def _contract_backward(self, features: torch.Tensor) -> list[torch.Tensor]:
        states = []
        features_rev = torch.flip(features, dims=[1])
        state = torch.einsum("bp,ipj->bj", features_rev[:, 0], self.backward_cores[0])
        state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
        states.append(state)

        for i in range(1, self.num_sites):
            state = torch.einsum("bi,ipj,bp->bj", state, self.backward_cores[i], features_rev[:, i])
            state = state / (torch.norm(state, dim=1, keepdim=True) + self.eps)
            states.append(state)

        return states

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        forward_states = self._contract_forward(features)
        backward_states = self._contract_backward(features)
        combined = torch.cat(forward_states + backward_states, dim=1)
        embedding = self.output_norm(combined)
        embedding = self.output_proj(embedding)
        return embedding


class MPSFunctionalGroupClassifier(nn.Module):
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

        assert input_dim % num_sites == 0, (
            f"input_dim ({input_dim}) must be divisible by num_sites ({num_sites})"
        )
        assert input_dim % num_sites_2 == 0, (
            f"input_dim ({input_dim}) must be divisible by num_sites_2 ({num_sites_2})"
        )
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
        self.num_sites_2 = num_sites_2
        self.site_dim_2 = input_dim // num_sites_2
        self.physical_dim_2 = physical_dim_2
        self.bond_dim_2 = bond_dim_2

        self.feature_map = LocalFeatureMap(
            site_dim=self.site_dim, physical_dim=physical_dim, dropout_rate=dropout_rate
        )
        self.feature_map_2 = LocalFeatureMap(
            site_dim=self.site_dim_2, physical_dim=physical_dim_2, dropout_rate=dropout_rate
        )

        intermediate_dim = 128

        if classifier_head == "cnn":
            self.mps_encoder = MPSEncoder(
                num_sites=num_sites, physical_dim=physical_dim, bond_dim=bond_dim, output_dim=256
            )
            self.mps_encoder_2 = MPSEncoder(
                num_sites=num_sites_2,
                physical_dim=physical_dim_2,
                bond_dim=bond_dim_2,
                output_dim=256,
            )
            cnn_in_channels = 2 * bond_dim + self.site_dim
            self.conv1 = nn.Conv1d(cnn_in_channels, 31, kernel_size=11, stride=1, padding="same")
            self.bn1 = nn.BatchNorm1d(31)
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv1d(31, 62, kernel_size=11, stride=1, padding="same")
            self.bn2 = nn.BatchNorm1d(62)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            conv_out_length = num_sites // 2 // 2
            flat_size = 62 * conv_out_length
            self.fc1 = nn.Linear(flat_size + 256, 4927)
            self.fc2 = nn.Linear(4927, 2785)
            self.fc3 = nn.Linear(2785, 1574)
            self.fc_out = nn.Linear(1574, num_classes)
            self.cnn_dropout = nn.Dropout(0.48599073736368)
            self.relu = nn.ReLU()
        else:
            self.mps_encoder = MPSEncoder(
                num_sites=num_sites,
                physical_dim=physical_dim,
                bond_dim=bond_dim,
                output_dim=intermediate_dim,
            )
            self.mps_encoder_2 = MPSEncoder(
                num_sites=num_sites_2,
                physical_dim=physical_dim_2,
                bond_dim=bond_dim_2,
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

    def _encode_site_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_sites = x.view(batch_size, self.num_sites, self.site_dim)
        x_flat = x_sites.view(batch_size * self.num_sites, self.site_dim)
        features_flat = self.feature_map(x_flat)
        features = features_flat.view(batch_size, self.num_sites, self.physical_dim)

        x_sites_2 = x.view(batch_size, self.num_sites_2, self.site_dim_2)
        x_flat_2 = x_sites_2.view(batch_size * self.num_sites_2, self.site_dim_2)
        features_flat_2 = self.feature_map_2(x_flat_2)
        features_2 = features_flat_2.view(batch_size, self.num_sites_2, self.physical_dim_2)
        return x_sites, features, features_2

    def _combine_embeddings(self, features: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            emb1 = self.mps_encoder(features.float())
            emb2 = self.mps_encoder_2(features_2.float())
        return torch.cat([emb1, emb2], dim=1)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        _, features, features_2 = self._encode_site_features(x)
        return self._combine_embeddings(features, features_2)

    def classify_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.classifier_head != "mps":
            raise ValueError("classify_from_latent is only available for classifier_head='mps'")
        return self.classifier(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sites, features, features_2 = self._encode_site_features(x)

        if self.classifier_head == "mps":
            return self.classify_from_latent(self._combine_embeddings(features, features_2))

        with torch.amp.autocast("cuda", enabled=False):
            features_f32 = features.float()
            forward_states = self.mps_encoder._contract_forward(features_f32)
            backward_states = self.mps_encoder._contract_backward(features_f32)

        fw = torch.stack(forward_states, dim=1)
        bw = torch.stack(backward_states, dim=1)
        per_site_mps = torch.cat([fw, bw], dim=2)
        per_site = torch.cat([per_site_mps, x_sites], dim=2)
        x = per_site.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x_sites.size(0), -1)

        with torch.amp.autocast("cuda", enabled=False):
            emb2 = self.mps_encoder_2(features_2.float())
        x = torch.cat([x, emb2], dim=1)
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
