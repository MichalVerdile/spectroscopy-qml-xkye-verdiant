"""CNN/MPS/TTN classifiers that start with the complete canonical raw tensor.

All normalization below is explicitly part of the trainable architecture. There
are no fixed signal filters, derivatives or external signal normalizations here.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def check_input(x, length, channels):
    if x.ndim != 3 or x.shape[1:] != (length, channels):
        raise ValueError(f"Expected (batch, {length}, {channels}), got {tuple(x.shape)}")


class ResidualConv(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm1, self.norm2 = nn.BatchNorm1d(width), nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = F.gelu(self.norm1(self.conv1(x))) * mask
        return F.gelu(x + self.dropout(self.norm2(self.conv2(h)))) * mask


class NativeCNN(nn.Module):
    architecture = "cnn_raw_native_v1"

    def __init__(self, length, channels, labels=37, width=48, dropout=0.2, **_):
        super().__init__()
        self.length, self.channels = length, channels
        if width < 3:
            raise ValueError("CNN width must be >=3")
        branch_width = max(1, width // 3)
        width = 3 * branch_width
        self.branches = nn.ModuleList(
            [nn.Conv1d(channels, branch_width, k, padding=k // 2) for k in (3, 11, 31)]
        )
        self.position = nn.Parameter(torch.randn(1, width, length) * 0.001)
        self.blocks = nn.ModuleList([ResidualConv(width, d, dropout) for d in (1, 2, 4)])
        self.head = nn.Sequential(
            nn.Linear(2 * width, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, labels)
        )

    def forward(self, x):
        check_input(x, self.length, self.channels)
        mask = x[..., -1:].transpose(1, 2)
        h = torch.cat([conv(x.transpose(1, 2)) for conv in self.branches], dim=1)
        h = F.gelu(h + self.position) * mask
        for block in self.blocks:
            h = block(h, mask)
            if h.shape[-1] > 1:
                h = F.avg_pool1d(h, 2, ceil_mode=True)
                mask = F.max_pool1d(mask, 2, ceil_mode=True)
        average = (h * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        maximum = h.masked_fill(mask == 0, -torch.inf).amax(-1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat([average, maximum], dim=1))


class SegmentEncoder(nn.Module):
    """Cover every coordinate, including the tail, before learned compression.

    Four feature-map ablations: small MLP, historical-style local MLP (including
    input LayerNorm), learned angles WITH a linear raw branch, latent products.
    """

    def __init__(
        self, length, channels, width, window=128, stride=None, mapping="small", dropout=0.2
    ):
        super().__init__()
        stride = window if stride is None else stride
        if (
            window < 1
            or not 1 <= stride <= window
            or mapping not in {"small", "legacy_mlp", "angle", "product"}
        ):
            raise ValueError("Invalid segment or feature-map configuration")
        self.window, self.stride, self.mapping = window, stride, mapping
        self.sites = max(1, math.ceil(max(0, length - window) / stride) + 1)
        self.padded_length = (self.sites - 1) * stride + window
        in_features = window * channels
        hidden = max(width * 2, in_features) if mapping == "legacy_mlp" else width * 2
        self.input_norm = nn.LayerNorm(in_features) if mapping == "legacy_mlp" else nn.Identity()
        self.project = nn.Linear(in_features, hidden)
        self.output = nn.Linear(hidden, width)
        self.raw_branch = (
            nn.Linear(in_features, width, bias=False) if mapping in {"angle", "product"} else None
        )
        self.position = nn.Parameter(torch.randn(1, self.sites, width) * 0.001)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        padded = F.pad(x, (0, 0, 0, self.padded_length - x.shape[1]))
        segments = padded.unfold(1, self.window, self.stride).flatten(2)
        valid = padded[..., -1].unfold(1, self.window, self.stride).any(-1)
        h = self.project(self.input_norm(segments))
        if self.mapping == "angle":
            h = torch.sin(h) + torch.cos(h)
        elif self.mapping == "product":
            h = torch.tanh(h) * torch.tanh(torch.roll(h, 1, dims=-1))
        else:
            h = F.gelu(h)
        h = self.output(self.dropout(h))
        if self.raw_branch is not None:
            h = h + self.raw_branch(segments)
        h = self.norm(h + self.position)
        return h * valid[..., None], valid


class MPSChain(nn.Module):
    def __init__(self, sites, physical, bond, bidirectional=True, residual=0.1, normalize=True):
        super().__init__()
        self.bond, self.residual, self.normalize = bond, residual, normalize
        self.directions = 2 if bidirectional else 1
        # Constant feature gives each transfer an identity channel at initialization.
        cores = torch.randn(self.directions, sites, bond, physical + 1, bond) * (
            0.01 / math.sqrt(physical * bond)
        )
        cores[:, :, :, 0, :] += torch.eye(bond)
        self.cores = nn.Parameter(cores)
        self.boundary = nn.Parameter(torch.randn(self.directions, bond) / math.sqrt(bond))

    def forward(self, features, valid):
        representations = []
        phi = torch.cat(
            [torch.ones_like(features[..., :1]), features / math.sqrt(features.shape[-1])], dim=-1
        )
        for direction in range(self.directions):
            state = self.boundary[direction].expand(len(features), -1)
            states = []
            magnitude = torch.zeros(len(features), 1, device=features.device)
            indices = (
                range(features.shape[1]) if direction == 0 else range(features.shape[1] - 1, -1, -1)
            )
            for index in indices:
                transfer = torch.einsum("bp,ipj->bij", phi[:, index], self.cores[direction, index])
                updated = torch.bmm(state[:, None, :], transfer).squeeze(1)
                updated = updated + self.residual * state
                if self.normalize:
                    scale = updated.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    updated = updated / scale
                    magnitude = magnitude + torch.where(valid[:, index, None], scale.log(), 0)
                state = torch.where(valid[:, index, None], updated, state)
                states.append(state)
            # Keep position-dependent states (cores are site-specific); mean and
            # endpoint readout plus log-magnitude provide a compact stable head.
            representations.extend([state, torch.stack(states, 1).mean(1), magnitude])
        return torch.cat(representations, 1)


class NativeMPS(nn.Module):
    architecture = "mps_raw_native_v1"

    def __init__(
        self,
        length,
        channels,
        labels=37,
        physical_dim=16,
        bond_dim=32,
        window=128,
        dual_scale=False,
        bidirectional=True,
        mapping="small",
        dropout=0.2,
        residual=0.1,
        normalize=True,
        **_,
    ):
        super().__init__()
        self.length, self.channels = length, channels
        if physical_dim < 2 or bond_dim < 2:
            raise ValueError("MPS physical/bond dimensions must be >=2")
        windows = [window, max(1, window // 2)] if dual_scale else [window]
        self.encoders, self.chains = nn.ModuleList(), nn.ModuleList()
        output_dim = 0
        for size in windows:
            encoder = SegmentEncoder(
                length, channels, physical_dim, size, mapping=mapping, dropout=dropout
            )
            self.encoders.append(encoder)
            self.chains.append(
                MPSChain(encoder.sites, physical_dim, bond_dim, bidirectional, residual, normalize)
            )
            output_dim += (2 if bidirectional else 1) * (2 * bond_dim + 1) + physical_dim
        self.head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, labels),
        )

    def forward(self, x):
        check_input(x, self.length, self.channels)
        representations = []
        for encoder, chain in zip(self.encoders, self.chains, strict=True):
            h, valid = encoder(x)
            representations.extend(
                [chain(h, valid), h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)]
            )
        return self.head(torch.cat(representations, 1))


class TreeMerge(nn.Module):
    def __init__(self, chi, mode, residual, normalize):
        super().__init__()
        if mode not in {"strict", "relaxed"}:
            raise ValueError("Unknown merge mode")
        self.mode, self.residual, self.normalize = (
            mode,
            0.0 if mode == "strict" else residual,
            normalize,
        )
        self.weight = nn.Parameter(torch.empty(chi * chi, chi))
        nn.init.orthogonal_(self.weight)

    def forward(self, left, right):
        # Strict QR once per level, shared across all node pairs in the batch.
        # Relaxed variant is a trainable unconstrained tensor initialized isometric.
        weight = (
            torch.linalg.qr(self.weight, mode="reduced")[0]
            if self.mode == "strict"
            else self.weight
        )
        pair = torch.einsum("bni,bnj->bnij", left, right).flatten(-2)
        h = pair @ weight + self.residual * (left + right) * 0.5
        return F.normalize(h, dim=-1, eps=1e-8) if self.normalize else h


class NativeTTN(nn.Module):
    architecture = "ttn_raw_native_v1"

    def __init__(
        self,
        length,
        channels,
        labels=37,
        chi=32,
        window=128,
        overlap=0.5,
        merge_mode="relaxed",
        residual=0.1,
        multiscale=True,
        normalize=True,
        mapping="small",
        dropout=0.2,
        **_,
    ):
        super().__init__()
        if chi < 2 or not 0 <= overlap < 1:
            raise ValueError("Invalid TTN dimensions/overlap")
        self.length, self.channels, self.multiscale = length, channels, multiscale
        self.encoder = SegmentEncoder(
            length, channels, chi, window, max(1, round(window * (1 - overlap))), mapping, dropout
        )
        levels = math.ceil(math.log2(self.encoder.sites))
        self.merges = nn.ModuleList(
            [TreeMerge(chi, merge_mode, residual, normalize) for _ in range(levels)]
        )
        features = chi * (levels + 1) if multiscale else chi
        self.head = nn.Sequential(
            nn.LayerNorm(features), nn.Dropout(dropout), nn.Linear(features, labels)
        )

    def forward(self, x):
        check_input(x, self.length, self.channels)
        h, valid = self.encoder(x)
        readouts = [h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)]
        for merge in self.merges:
            if h.shape[1] % 2:
                h = F.pad(h, (0, 0, 0, 1))
                valid = F.pad(valid, (0, 1))
            left, right = h[:, ::2], h[:, 1::2]
            lm, rm = valid[:, ::2, None], valid[:, 1::2, None]
            merged = merge(left, right)
            # A lone leaf is carried unchanged; never multiplied by zero padding.
            h = torch.where(lm & rm, merged, torch.where(lm, left, right))
            valid = (lm | rm).squeeze(-1)
            h = h * valid[..., None]
            readouts.append(h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1))
        return self.head(torch.cat(readouts, 1) if self.multiscale else h[:, 0])


class HistoricalCNNAdapter(nn.Module):
    """PyTorch port of the historical 31/62 + 4927/2785/1574 architecture.

    The input channel count includes the common mask; native length is retained.
    This is an architecture control, NOT bit-equivalent TensorFlow reproduction.
    """

    architecture = "cnn_jung_native_adapter_v1"

    def __init__(self, length, channels, labels=37, parameter_limit=100_000_000, **_):
        super().__init__()
        flat = (length // 4) * 62
        estimate = flat * 4927 + 4927 * 2785 + 2785 * 1574
        if length < 4 or estimate > parameter_limit:
            raise ValueError(
                f"Historical CNN needs ~{estimate:,} weights. Set --parameter-limit explicitly or use optimized; no approximation applied."
            )
        self.net = nn.Sequential(
            nn.Conv1d(channels, 31, 11, padding=5),
            nn.BatchNorm1d(31),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(31, 62, 11, padding=5),
            nn.BatchNorm1d(62),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(flat, 4927),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(4927, 2785),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(2785, 1574),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(1574, labels),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class HistoricalMPSAdapter(nn.Module):
    architecture = "mps_legacy_native_adapter_v1"

    def __init__(
        self, length, channels, labels=37, modality="ir", parameter_limit=100_000_000, **_
    ):
        super().__init__()
        from spectroscopy_qml.mps_classifier.ir.model import MPSFunctionalGroupClassifier

        sites = {"cnmr": (25, 10), "hnmr": (10, 5)}.get(modality, (5, 10))
        n = length * channels
        if any(n % s for s in sites):
            raise ValueError(
                "Historical MPS requires divisibility; no truncation/padding approximation applied"
            )
        estimate = sum(2 * (s - 1) * 128 * 450 * 128 for s in sites)
        if estimate > parameter_limit:
            raise ValueError(
                f"Historical MPS cores alone need {estimate:,} weights; raise --parameter-limit explicitly"
            )
        self.model = MPSFunctionalGroupClassifier(
            input_dim=n,
            num_sites=sites[0],
            num_sites_2=sites[1],
            physical_dim=450,
            physical_dim_2=450,
            bond_dim=128,
            bond_dim_2=128,
            num_classes=labels,
            dropout_rate=0.49,
        )

    def forward(self, x):
        return self.model(x.flatten(1))


def build_model(
    model,
    length,
    channels,
    config,
    *,
    variant="optimized",
    modality="ir",
    parameter_limit=100_000_000,
):
    options = {**config, "length": length, "channels": channels}
    if variant == "baseline":
        if model == "cnn":
            return HistoricalCNNAdapter(**options, parameter_limit=parameter_limit)
        if model == "mps":
            return HistoricalMPSAdapter(
                **options, modality=modality, parameter_limit=parameter_limit
            )
        if model == "ttn":
            raise ValueError(
                "Historical TTN uses fixed Lorentz/derivative filters and is not strict raw. Run optimized ttn_raw_native_v1; legacy is a separate feature-map ablation."
            )
    constructors = {"cnn": NativeCNN, "mps": NativeMPS, "ttn": NativeTTN}
    if model not in constructors:
        raise ValueError(f"Unknown neural model {model}")
    with torch.device("meta"):
        preview = constructors[model](**options)
    count = sum(p.numel() for p in preview.parameters())
    if count > parameter_limit:
        raise ValueError(
            f"Model needs {count:,} parameters, above limit {parameter_limit:,}; no allocation or hidden shrinking performed"
        )
    del preview
    return constructors[model](**options)
