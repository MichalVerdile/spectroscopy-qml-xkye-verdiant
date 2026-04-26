"""
Pure MPS/Tensor Network Classifier for IR Spectra.

This module implements a multi-label classifier based on Matrix Product States (MPS)
for predicting functional groups from IR spectroscopy data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Neutral boundary to start contractions from an interior core when
        # aggregating suffix subchains.
        self.register_buffer(
            "left_boundary",
            torch.full((bond_dim,), 1.0 / (bond_dim**0.5)),
        )

        # Output projection: collect every intermediate state for every suffix
        # subchain in both directions. For num_sites=N, each direction yields
        # N + (N-1) + ... + 1 = N * (N + 1) / 2 states.
        total_suffix_states = num_sites * (num_sites + 1) // 2
        total_features = 2 * total_suffix_states * bond_dim
        self.output_norm = nn.LayerNorm(total_features)
        self.output_proj = nn.Linear(total_features, output_dim)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize bond states to keep contractions numerically stable."""
        return state / (torch.norm(state, dim=1, keepdim=True) + self.eps)

    def _start_forward_state(self, site_features: torch.Tensor, start_idx: int) -> torch.Tensor:
        """Start a forward contraction at any core index."""
        if start_idx == 0:
            state = torch.einsum("bp,ipj->bj", site_features, self.forward_cores[0])
        else:
            state = torch.einsum(
                "i,ipj,bp->bj",
                self.left_boundary,
                self.forward_cores[start_idx],
                site_features,
            )
        return self._normalize_state(state)

    def _contract_forward(self, features: torch.Tensor, start_idx: int = 0) -> list[torch.Tensor]:
        """
        Perform forward contraction on a suffix subchain, collecting the bond
        vector after every active core.

        Args:
            features: Tensor of shape (batch_size, suffix_length, physical_dim)
            start_idx: Index of the first active core in the full chain

        Returns:
            List of suffix_length tensors, each of shape (batch_size, bond_dim)
        """
        states = []
        suffix_length = features.size(1)

        state = self._start_forward_state(features[:, 0], start_idx)
        states.append(state)

        for offset in range(1, suffix_length):
            core_idx = start_idx + offset
            state = torch.einsum(
                "bi,ipj,bp->bj",
                state,
                self.forward_cores[core_idx],
                features[:, offset],
            )
            state = self._normalize_state(state)
            states.append(state)

        return states

    def _contract_backward(self, features: torch.Tensor) -> list[torch.Tensor]:
        """
        Perform backward contraction on a suffix subchain, collecting the bond
        vector after every active core.

        Args:
            features: Tensor of shape (batch_size, suffix_length, physical_dim)

        Returns:
            List of suffix_length tensors, each of shape (batch_size, bond_dim)
        """
        states = []
        features_rev = torch.flip(features, dims=[1])
        suffix_length = features_rev.size(1)

        state = torch.einsum("bp,ipj->bj", features_rev[:, 0], self.backward_cores[0])
        state = self._normalize_state(state)
        states.append(state)

        for offset in range(1, suffix_length):
            state = torch.einsum(
                "bi,ipj,bp->bj",
                state,
                self.backward_cores[offset],
                features_rev[:, offset],
            )
            state = self._normalize_state(state)
            states.append(state)

        return states

    def _collect_suffix_states(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Collect intermediate states for all left-trimmed suffix subchains."""
        states = []
        for start_idx in range(self.num_sites):
            states.extend(self._contract_forward(features[:, start_idx:], start_idx=start_idx))
        return states

    def _collect_backward_suffix_states(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Collect backward states for all left-trimmed suffix subchains."""
        states = []
        for start_idx in range(self.num_sites):
            states.extend(self._contract_backward(features[:, start_idx:]))
        return states

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features using bidirectional MPS.

        Collects the bond vector after every active core for every suffix
        subchain in both directions and concatenates all of them.

        Args:
            features: Tensor of shape (batch_size, num_sites, physical_dim)

        Returns:
            Embedding of shape (batch_size, output_dim)
        """
        forward_states = self._collect_suffix_states(features)
        backward_states = self._collect_backward_suffix_states(features)

        combined = torch.cat(forward_states + backward_states, dim=1)

        embedding = self.output_norm(combined)
        embedding = self.output_proj(embedding)

        return embedding


# ---------------------------------------------------------------------------
# Quantum-branch constants
# ---------------------------------------------------------------------------

# The 16 functional groups predicted by the quantum segment branch.
# Indices into the FUNCTIONAL_GROUPS ordered dict (alphabetical order as in data_loader.py).
QUANTUM_LABEL_NAMES: list[str] = [
    "Alkene",       # index  5
    "Thioamide",    # index 35
    "Aldehyde",     # index  3
    "Enol",         # index 14
    "Ketone",       # index 24
    "Enamine",      # index 13
    "Sulfide",      # index 28
    "Hydrazone",    # index 19
    "Imine",        # index 21
    "Acyl halide",  # index  1
    "Acid anhydride",  # index  0
    "Phosphine",    # index 27
    "Sulfoxide",    # index 33
    "Azo compound", # index 10
    "Thial",        # index 34
    "Sulfonic acid",# index 32
]
QUANTUM_LABEL_INDICES: list[int] = [5, 35, 3, 14, 24, 13, 28, 19, 21, 1, 0, 27, 33, 10, 34, 32]
NUM_QUANTUM_LABELS: int = 16


# ---------------------------------------------------------------------------
# Qiskit parameter-shift autograd bridge
# ---------------------------------------------------------------------------

class _QiskitParamShiftFn(torch.autograd.Function):
    """
    Bridges Qiskit statevector simulation with PyTorch autograd via the
    parameter-shift rule.

    The parameter-shift rule computes the gradient of a rotation gate
    parameter θ as:  ∂⟨O⟩/∂θ = [⟨O⟩(θ+π/2) − ⟨O⟩(θ−π/2)] / 2

    NOTE: This requires 2 × N_params additional forward passes per backward
    step (here N_params = 40).  Use the PennyLane backend for full training
    runs and reserve this backend for small-scale simulation / verification.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weights: torch.Tensor, qmodule) -> torch.Tensor:  # type: ignore[override]
        x_np = inputs.detach().cpu().numpy().astype(np.float64)
        w_np = weights.detach().cpu().numpy().astype(np.float64)
        out_np = qmodule._evaluate_batch(x_np, w_np)
        ctx.save_for_backward(inputs, weights)
        ctx.qmodule = qmodule
        return torch.from_numpy(out_np).to(dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        inputs, weights = ctx.saved_tensors
        qm = ctx.qmodule
        x_np = inputs.detach().cpu().numpy().astype(np.float64)
        w_np = weights.detach().cpu().numpy().astype(np.float64)
        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)

        n_weights = len(w_np)
        grad_w = np.zeros(n_weights, dtype=np.float64)
        shift = np.pi / 2.0

        for i in range(n_weights):
            w_p = w_np.copy(); w_p[i] += shift
            w_m = w_np.copy(); w_m[i] -= shift
            out_p = qm._evaluate_batch(x_np, w_p)
            out_m = qm._evaluate_batch(x_np, w_m)
            # chain rule: sum over (batch × observable) dimensions
            grad_w[i] = (grad_np * (out_p - out_m) / 2.0).sum()

        return (
            None,  # no gradient w.r.t. inputs (data-encoding, not trained)
            torch.from_numpy(grad_w).to(dtype=weights.dtype, device=weights.device),
            None,  # no gradient w.r.t. qmodule reference
        )


# ---------------------------------------------------------------------------
# Quantum circuit implementations
# ---------------------------------------------------------------------------

class PennyLaneDataReuploadingCircuit(nn.Module):
    """
    Data-reuploading variational quantum circuit backed by PennyLane.

    Architecture per segment (4 qubits, 5 layers):
      For each layer l = 0 … 4:
        ‑ Encode 4 data values:   RY(x[l·4 + q])          on qubit q  (data re-uploading)
        ‑ Trainable rotations:    RY(θ_y[l,q]),  RZ(θ_z[l,q])  on qubit q
        ‑ Entanglement:           CNOT(q → q+1)  chain

      Measurements: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩
      Optional extra: ⟨ZᵢZⱼ⟩ for all i < j  (6 two-qubit correlations)

    Uses ``diff_method="backprop"`` on PennyLane's ``default.qubit`` device,
    so gradients flow through PyTorch autograd with no extra overhead.

    Args:
        measure_correlations: If True, also measure ⟨ZᵢZⱼ⟩ correlations
                              (output dim increases from 4 to 10).

    Input:  ``(batch, 20)``  – segment values pre-scaled to angle range
    Output: ``(batch, 4)``   or  ``(batch, 10)``  with correlations,
            all values in [−1, 1].
    """

    def __init__(
        self,
        measure_correlations: bool = False,
        num_qubits: int = 4,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        try:
            import pennylane as qml  # lazy import
        except ImportError as exc:
            raise ImportError(
                "PennyLane is required for the quantum branch. "
                "Install it with:  pip install pennylane"
            ) from exc

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_size = num_qubits * num_layers
        self.measure_correlations = measure_correlations
        num_corr_pairs = num_qubits * (num_qubits - 1) // 2
        self.out_features: int = num_qubits + (num_corr_pairs if measure_correlations else 0)

        nq = self.num_qubits
        nl = self.num_layers
        mc = measure_correlations

        dev = qml.device("default.qubit", wires=nq)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            # Use `inputs[..., idx]` so the circuit works correctly whether
            # TorchLayer passes a 1-D single-sample tensor (shape: (20,)) or
            # a 2-D batched tensor (shape: (B, 20)).  The ellipsis index
            # selects feature column `idx` across all batch rows, enabling
            # PennyLane parameter-broadcasting for the batched case and
            # returning a plain scalar for the unbatched case.
            for layer in range(nl):
                # Data re-uploading: encode 4 values per layer via RY gates
                for q in range(nq):
                    qml.RY(inputs[..., layer * nq + q], wires=q)
                # Trainable single-qubit rotations
                for q in range(nq):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)

                for q in range(nq):
                    qml.Hadamard(wires=q)
                # CNOT entanglement chain: q → q+1
                for q in range(nq - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Measurements
            meas = [qml.expval(qml.PauliZ(q)) for q in range(nq)]
            if mc:
                meas += [
                    qml.expval(qml.PauliZ(q0) @ qml.PauliZ(q1))
                    for q0 in range(nq)
                    for q1 in range(q0 + 1, nq)
                ]
            return meas

        weight_shapes = {"weights": (nl, nq, 2)}
        self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, 20)`` segment values in [−π, π].
        Returns:
            ``(batch, out_features)`` expectation values in [−1, 1].
        """
        return self.qlayer(x)


class QiskitDataReuploadingCircuit(nn.Module):
    """
    Data-reuploading variational quantum circuit backed by Qiskit ≥ 1.0.

    Architecture is identical to :class:`PennyLaneDataReuploadingCircuit`
    (4 qubits, 5 layers, CNOT chain, Z-expectation measurements).
    Exact statevector simulation is performed via
    ``qiskit.primitives.StatevectorEstimator`` (no shots, no Aer needed).

    Gradients use the **parameter-shift rule** (see :class:`_QiskitParamShiftFn`),
    requiring 2 × 40 = 80 additional circuit evaluations per backward step.
    Prefer :class:`PennyLaneDataReuploadingCircuit` for full training runs;
    use this class for simulation, verification, or cross-framework comparison.

    Args:
        measure_correlations: If True, also measure ⟨ZᵢZⱼ⟩ correlations.

    Input:  ``(batch, 20)``
    Output: ``(batch, 4)``  or  ``(batch, 10)``  with correlations.
    """

    def __init__(
        self,
        measure_correlations: bool = False,
        num_qubits: int = 4,
        num_layers: int = 5,
        ibm_backend_name: str | None = None,
        ibm_instance: str | None = None,
        ibm_token: str | None = None,
        ibm_shots: int = 4096,
    ) -> None:
        super().__init__()
        try:
            from qiskit import QuantumCircuit as _QC  # noqa: F401 – availability check
            from qiskit.circuit import ParameterVector
            from qiskit.primitives import StatevectorEstimator  # noqa: F401
            from qiskit.quantum_info import SparsePauliOp
        except ImportError as exc:
            raise ImportError(
                "Qiskit ≥ 1.0 is required for the Qiskit quantum backend. "
                "Install it with:  pip install qiskit"
            ) from exc

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_size = num_qubits * num_layers
        self.measure_correlations = measure_correlations
        nq = self.num_qubits
        nl = self.num_layers
        num_corr_pairs = nq * (nq - 1) // 2
        self.out_features: int = nq + (num_corr_pairs if measure_correlations else 0)

        # Trainable weights stored as a flat PyTorch Parameter so they appear
        # in model.parameters() and are optimised by the standard Adam step.
        self.weights = nn.Parameter(
            torch.empty(nl * nq * 2).uniform_(-float(np.pi), float(np.pi))
        )

        # Parametric circuit symbols (not PyTorch tensors)
        self._input_params = ParameterVector("x", self.input_size)
        self._weight_params = ParameterVector("θ", nl * nq * 2)
        self._circuit = self._build_circuit()

        # Observables  ⟨Zᵢ⟩ — Qiskit uses little-endian qubit ordering:
        # qubit 0 is the rightmost character in the Pauli string.
        from qiskit.quantum_info import SparsePauliOp
        self._observables: list = []
        for q in range(nq):
            pauli_str = "I" * q + "Z" + "I" * (nq - 1 - q)
            self._observables.append(SparsePauliOp(pauli_str))
        if measure_correlations:
            for q0 in range(nq):
                for q1 in range(q0 + 1, nq):
                    parts = ["I"] * nq
                    parts[q0] = "Z"
                    parts[q1] = "Z"
                    self._observables.append(SparsePauliOp("".join(reversed(parts))))

        # ── Optional IBM Quantum Runtime backend ─────────────────────────
        # Set up when ibm_backend_name is provided; otherwise all evaluation
        # falls back to the local StatevectorEstimator.
        self._ibm_estimator = None
        self._isa_circuit = None
        self._isa_observables: list = self._observables  # default: no layout remap

        if ibm_backend_name is not None:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
                from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            except ImportError as exc:
                raise ImportError(
                    "qiskit-ibm-runtime is required for the IBM Quantum backend. "
                    "Install it with:  pip install qiskit-ibm-runtime"
                ) from exc

            service = QiskitRuntimeService(
                channel="ibm_cloud",
                token=ibm_token,
                instance=ibm_instance,
            )
            ibm_backend = service.backend(ibm_backend_name)

            # Transpile the parametric circuit once to the backend's ISA form.
            pm = generate_preset_pass_manager(backend=ibm_backend, optimization_level=1)
            self._isa_circuit = pm.run(self._circuit)

            # Remap observables to the post-transpilation qubit layout.
            self._isa_observables = [
                obs.apply_layout(self._isa_circuit.layout)
                for obs in self._observables
            ]

            # qiskit-ibm-runtime ≥ 0.20: EstimatorV2 takes `mode` (not `backend`),
            # and options are set via attribute assignment after construction.
            self._ibm_estimator = EstimatorV2(mode=ibm_backend)
            self._ibm_estimator.options.default_shots = ibm_shots

    def _build_circuit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.num_qubits)
        w_idx = 0
        for layer in range(self.num_layers):
            # Data encoding
            for q in range(self.num_qubits):
                qc.ry(self._input_params[layer * self.num_qubits + q], q)
            # Trainable rotations
            for q in range(self.num_qubits):
                qc.ry(self._weight_params[w_idx], q); w_idx += 1
                qc.rz(self._weight_params[w_idx], q); w_idx += 1
                
            for q in range(self.num_qubits):
                qc.h(q)
            # CNOT chain
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def _evaluate_batch(self, x_np: np.ndarray, w_np: np.ndarray) -> np.ndarray:
        """Run circuit evaluation for a batch of segments.

        Dispatches to the IBM Quantum Runtime backend when one has been
        configured (``ibm_backend_name`` was passed at construction time),
        otherwise falls back to local exact statevector simulation.

        Args:
            x_np: ``(N, 20)`` input values as float64.
            w_np: ``(40,)``   weight values as float64.

        Returns:
            ``(N, out_features)`` expectation values as float32.
        """
        if self._ibm_estimator is not None:
            return self._evaluate_batch_ibm(x_np, w_np)

        from qiskit.primitives import StatevectorEstimator
        estimator = StatevectorEstimator()
        batch_size = x_np.shape[0]
        results = np.zeros((batch_size, self.out_features), dtype=np.float32)

        for i in range(batch_size):
            # Build explicit parameter dict to avoid Qiskit's lexicographic sort
            param_dict = {p: float(x_np[i, k]) for k, p in enumerate(self._input_params)}
            param_dict.update({p: float(w_np[k]) for k, p in enumerate(self._weight_params)})
            pubs = [(self._circuit, obs, param_dict) for obs in self._observables]
            res = estimator.run(pubs).result()
            for j in range(len(self._observables)):
                results[i, j] = float(res[j].data.evs)

        return results

    def _evaluate_batch_ibm(self, x_np: np.ndarray, w_np: np.ndarray) -> np.ndarray:
        """Run on IBM Quantum hardware via EstimatorV2 (all samples in one job).

        Args:
            x_np: ``(N, 20)`` input values as float64.
            w_np: ``(40,)``   weight values as float64.

        Returns:
            ``(N, out_features)`` expectation values as float32.
        """
        batch_size = x_np.shape[0]
        results = np.zeros((batch_size, self.out_features), dtype=np.float32)

        # Build a (batch_size, num_params) value matrix, ordering columns to
        # match the ISA circuit's sorted parameter list.
        sorted_params = sorted(self._isa_circuit.parameters, key=lambda p: p.name)
        x_cols = {p.name: x_np[:, k] for k, p in enumerate(self._input_params)}
        w_cols = {
            p.name: np.full(batch_size, float(w_np[k]))
            for k, p in enumerate(self._weight_params)
        }
        all_cols = {**x_cols, **w_cols}
        param_values = np.stack(
            [all_cols[p.name] for p in sorted_params], axis=1
        )  # (batch_size, num_params)

        # One PUB per observable; all batch samples submitted in a single job.
        pubs = [
            (self._isa_circuit, obs, param_values)
            for obs in self._isa_observables
        ]
        job = self._ibm_estimator.run(pubs)
        result = job.result()

        for j in range(len(self._isa_observables)):
            results[:, j] = np.asarray(result[j].data.evs, dtype=np.float32)

        return results

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, 20)`` segment values.
        Returns:
            ``(batch, out_features)`` expectation values in [−1, 1].
        """
        return _QiskitParamShiftFn.apply(x, self.weights, self)


# ---------------------------------------------------------------------------
# Quantum segment branch (third branch of the classifier)
# ---------------------------------------------------------------------------

class QuantumSegmentBranch(nn.Module):
    """
    Quantum-circuit third branch for the MPS functional group classifier.

    Processing pipeline
    -------------------
    1. Split each 1800-point spectrum into ``num_segments`` non-overlapping
       (or overlapping) segments of length 20  (= NUM_QUBITS × NUM_LAYERS).
    2. Per-segment tanh-SNV normalisation → values scaled to [-π, π].
    3. Apply a **shared** quantum circuit to every segment independently
       (shared weights = all segments contracted by the same VQC).
    4. Aggregate the sequence of segment embeddings with a 1-D CNN +
       adaptive global average pooling.
    5. Linear projection → ``num_classes`` logits (full label set).
       Competition with the MPS branches is restricted to the 16 quantum-specific
       labels (see :data:`QUANTUM_LABEL_NAMES` / :data:`QUANTUM_LABEL_INDICES`).

    Args:
        input_dim:             Total spectrum length (default: 1800).
        segment_length:        Segment length – must equal 20.
        segment_stride:        Step between consecutive segment starts
                               (default: 20 → no overlap; set < 20 for overlap).
        backend:               ``"pennylane"`` (default, recommended for training)
                               or ``"qiskit"`` (statevector simulation / verification).
        measure_correlations:  Also measure ⟨ZᵢZⱼ⟩ two-qubit correlations
                               (increases per-segment feature dim from 4 to 10).
        num_classes:           Number of output logits (should match the full
                               classifier label count, default: 37).
    """

    def __init__(
        self,
        input_dim: int = 1800,
        segment_length: int = 20,
        segment_stride: int = 20,
        backend: str = "pennylane",
        measure_correlations: bool = False,
        num_classes: int = 37,
        num_qubits: int = 4,
        num_layers: int = 5,
        ibm_backend_name: str | None = None,
        ibm_instance: str | None = None,
        ibm_token: str | None = None,
        ibm_shots: int = 4096,
    ) -> None:
        super().__init__()
        expected_segment_length = num_qubits * num_layers
        if segment_length != expected_segment_length:
            raise ValueError(
                f"segment_length must be {expected_segment_length} "
                f"(num_qubits × num_layers = {num_qubits} × {num_layers}), "
                f"got {segment_length}"
            )
        if backend not in ("pennylane", "qiskit"):
            raise ValueError(f"backend must be 'pennylane' or 'qiskit', got {backend!r}")

        self.input_dim = input_dim
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        self.backend = backend
        self.num_classes = num_classes
        self.num_segments: int = (input_dim - segment_length) // segment_stride + 1

        # Shared quantum circuit (one set of weights used for every segment)
        if backend == "pennylane":
            self.qcircuit: nn.Module = PennyLaneDataReuploadingCircuit(measure_correlations, num_qubits, num_layers)
        else:
            self.qcircuit = QiskitDataReuploadingCircuit(
                measure_correlations,
                num_qubits,
                num_layers,
                ibm_backend_name=ibm_backend_name,
                ibm_instance=ibm_instance,
                ibm_token=ibm_token,
                ibm_shots=ibm_shots,
            )

        qc_out = self.qcircuit.out_features  # 4 or 10

        # 1-D CNN: (B, qc_out, num_segments) → (B, 64)
        self.conv1 = nn.Conv1d(qc_out, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Dense output
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, input_dim)`` raw spectrum tensor.
        Returns:
            ``(B, num_classes)`` logits for all labels (full label set).
        """
        B = x.size(0)

        # ── 1. Segment ────────────────────────────────────────────────────
        # segments: (B, num_segments, segment_length)
        segments = x.unfold(
            dimension=1,
            size=self.segment_length,
            step=self.segment_stride,
        ).contiguous()

        # ── 2. Per-segment tanh-SNV normalisation → angle range [−π, π] ──
        seg_mean = segments.mean(dim=-1, keepdim=True)
        seg_std = segments.std(dim=-1, keepdim=True).clamp(min=1e-8)
        segments = torch.tanh((segments - seg_mean) / seg_std) * float(np.pi)

        # ── 3. Quantum circuit (batched over B × num_segments) ────────────
        # Force float32: PennyLane backprop and Qiskit statevector require it.
        segs_flat = segments.reshape(B * self.num_segments, self.segment_length).float()
        qc_out = self.qcircuit(segs_flat)                    # (B·S, qc_out)
        qc_out = qc_out.reshape(B, self.num_segments, -1)   # (B, S, qc_out)

        # ── 4. CNN aggregation ────────────────────────────────────────────
        h = qc_out.transpose(1, 2)                           # (B, qc_out, S)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.global_pool(h).squeeze(-1)                  # (B, 64)

        return self.fc(h)                                    # (B, num_classes)


class MPSFunctionalGroupClassifier(nn.Module):
    """
    Complete MPS-based multi-label classifier for functional group prediction.

    This model processes IR spectra through a **triple-branch** approach:
    1. Coarse MPS branch  – splits the spectrum into ``num_sites`` sites and
       encodes with a bidirectional MPS.
    2. Fine MPS branch    – splits the same spectrum into ``num_sites_2`` sites
       and encodes with a second bidirectional MPS.
    3. Quantum branch (optional) – segments the spectrum into 20-point windows,
       processes each with a shared 4-qubit data-reuploading variational circuit,
       aggregates via 1-D CNN, and outputs ``num_classes`` logits (full label set).
       The competitive loss with the MPS branches is restricted to the 16
       quantum-specific labels (see :data:`QUANTUM_LABEL_INDICES`).

    The ``forward()`` method always returns the full MPS logits
    ``(batch, num_classes)``.  The quantum branch lives at
    ``self.quantum_branch`` and can be called separately to obtain the 16
    competing logits used by the training loop's competitive-loss term.

    Args:
        input_dim: Length of input spectrum (default: 1800).
        num_sites: Number of sites for the first (coarse) MPS (default: 36).
        physical_dim: Physical dimension of first MPS (default: 8).
        bond_dim: Bond dimension of first MPS (default: 16).
        num_classes: Number of functional groups to predict (default: 37).
        dropout_rate: Dropout probability (default: 0.2).
        classifier_head: ``"mps"`` for pure MPS or ``"cnn"`` for MPS+CNN hybrid.
        num_sites_2: Number of sites for the second (fine) MPS (default: 1800).
        physical_dim_2: Physical dimension of second MPS (default: 8).
        bond_dim_2: Bond dimension of second MPS (default: 16).
        use_quantum_branch: If True, attach a :class:`QuantumSegmentBranch`
            as the third branch (default: False).
        quantum_backend: Backend for the quantum circuit –
            ``"pennylane"`` (default, gradient-compatible, recommended for
            training) or ``"qiskit"`` (statevector simulation / verification).
        quantum_segment_length: Length of each spectrum segment fed to the
            quantum circuit – must be 20 (= 4 qubits × 5 layers).
        quantum_segment_stride: Stride between segment starts (default: 20,
            no overlap; set smaller for overlapping segments).
        quantum_measure_correlations: Also measure ⟨ZᵢZⱼ⟩ two-qubit
            correlations in the quantum circuit (default: False).
        quantum_ibm_backend_name: IBM Quantum backend name to target, e.g.
            ``"ibm_fez"``, ``"ibm_kingston"``, or ``"ibm_marrakesh"``.
            Only used when ``quantum_backend="qiskit"`` (default: None →
            local statevector simulation).
        quantum_ibm_instance: IBM Cloud CRN of the service instance, e.g.
            ``"crn:v1:bluemix:public:quantum-computing:us-east:...::"`
            (default: None).
        quantum_ibm_token: IBM Cloud API key for authentication (default: None).
        quantum_ibm_shots: Number of shots per circuit execution on IBM hardware
            (default: 4096).
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
        # ── Quantum branch ──────────────────────────────────────────────
        use_quantum_branch: bool = True,
        quantum_backend: str = "qiskit",
        quantum_segment_length: int = 20,
        quantum_segment_stride: int = 20,
        quantum_measure_correlations: bool = True,
        quantum_num_qubits: int = 4,
        quantum_num_layers: int = 5,
        # ── IBM Quantum Runtime (requires quantum_backend="qiskit") ──────
        quantum_ibm_backend_name: str | None = "ibm_kingston",
        quantum_ibm_instance: str | None = "crn:v1:bluemix:public:quantum-computing:us-east:a/66c55298ad344d73a10696ed758e49a2:0f5504b0-3c5a-4f15-a64f-426282fa04b7::",
        quantum_ibm_token: str | None = "K9LhP9k-Hd3jY162C-W8zQ7urRm3svHsWRQWGXrLCSu5",
        quantum_ibm_shots: int = 4096,
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

        # ── Quantum segment branch (optional third branch) ────────────────
        # Registered as a submodule so its parameters are included in
        # model.parameters() and model.train() / model.eval() propagates.
        if use_quantum_branch:
            self.quantum_branch: QuantumSegmentBranch | None = QuantumSegmentBranch(
                input_dim=input_dim,
                segment_length=quantum_segment_length,
                segment_stride=quantum_segment_stride,
                backend=quantum_backend,
                measure_correlations=quantum_measure_correlations,
                num_classes=num_classes,
                num_qubits=quantum_num_qubits,
                num_layers=quantum_num_layers,
                ibm_backend_name=quantum_ibm_backend_name,
                ibm_instance=quantum_ibm_instance,
                ibm_token=quantum_ibm_token,
                ibm_shots=quantum_ibm_shots,
            )
        else:
            self.quantum_branch = None

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
