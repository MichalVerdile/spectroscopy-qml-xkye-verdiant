# Tensor-network architecture study

The new models are classical, trainable tensor networks. No quantum hardware or
quantum computational advantage is claimed. Smaller parameter count is an
observed architectural property; better predictive performance is a hypothesis.

## Existing MPS and measurements

The historical MPS uses physical dimensions 450, bond dimensions 128, two site
scales, bidirectional contractions, local input LayerNorm/MLPs and a 256-unit
classification head. The original configured input length is 600. Exact counts
were obtained by constructing the actual existing modules on PyTorch's meta
device, not by allocating their full tensors or estimating from one core.

| Source configuration | Sites in the two scales | Trainable parameters | Float32 weights, bytes |
|---|---|---:|---:|
| C-NMR | 25 / 10 | 488,964,537 | 1,955,858,148 |
| H-NMR | 10 / 5 | 193,473,529 | 773,894,116 |
| IR, MS/MS+, MS/MS− | 5 / 10 | 193,473,529 | 773,894,116 |

Parameters, gradients and two Adam moments alone imply a lower bound of 16
bytes per float32 parameter. Activations, workspaces, checkpoints and allocator
overhead are additional. This is **not measured GPU consumption**. Legacy GPU
memory, gradient norms and execution times have not been measured here because
no GPU is available. `scripts/profile_raw_native_models.py --execute-legacy`
performs these checks on disposable models on suitable hardware, with an
explicit `--parameter-limit`; it never warms up an experiment's training model.

## New MPS: `mps_raw_native_v1`

Every canonical input coordinate, including the mask and the final partial
segment, reaches a trainable local encoder. Default windows contain 128 source
positions, physical dimension is 16 and bond dimension is 32. A near-identity
core initialization, residual contraction, internal state normalization and
log-magnitude bookkeeping improve numerical conditioning. Endpoint and mean
readouts plus a learned local skip feed the classifier. Bidirectionality is on
by default; a second segment scale is optional. No source resampling occurs.

The size of a segment refers to source positions. On fixed NMR/IR axes its
physical span can be derived from the retained axis endpoints; on MS/MS it
refers to consecutive source peaks and is **not a fixed m/z interval**. Native
10,000-point input with window 128 has 79 sites; window 64 has 157 and window
256 has 40. Different scales therefore change sites without discarding points.

| Factor | Declared alternatives |
|---|---|
| Physical dimension | 8, 16, 32, 64 |
| Bond dimension | 16, 32, 64, 128 |
| Window | 64, 128, 256 |
| Scale / direction | Single or dual; unidirectional or bidirectional |
| Residual | 0, .1, .2 |
| Internal normalization | Enabled or disabled |
| Dropout | .1, .2, .3 |
| Learning rate / weight decay | .0003/.001/.003; .00001/.0001/.001 |

All 16 requested physical/bond combinations have finite forward/backward tests
on short synthetic inputs. This is shape/numerical validation, not validation
of their learning quality on the full dataset.

## New TTN: `ttn_raw_native_v1`

The historical Lorentzian feature map applies fixed filtering and analytical
derivatives, followed by state normalization. It is retained in the original
modules and must be reported as `ttn_lorentz_feature_map`. It is deliberately
ineligible for the strict raw leaderboard.

The new TTN begins with a trainable segment encoder. Default chi is 32, window
128, overlap .5 and multiscale readout enabled. A merge maps two chi-dimensional
states through a trainable bilinear tensor. Strict mode computes a reduced QR
isometry; relaxed mode starts from an isometric initialization and permits
unconstrained learning with a residual contribution. Odd surviving children
are carried unchanged, and empty padded leaves do not invent observations.

The architecture study includes chi 16/32/64/128, windows 64/128/256, overlap
0/.25/.5, strict/relaxed merges, residual 0/.1/.2, multiscale readout, hidden
normalization, feature mappings and dropout .1/.2/.3. Tests cover each chi in
both merge modes, QR orthogonality, odd trees and finite gradients. Full-size
QR cost and GPU stability still require measurement on the cluster.

## Four local feature mappings

| Name | Definition and raw boundary |
|---|---|
| `small` | Small trainable projection/MLP beginning directly from canonical raw segments. |
| `legacy_mlp` | Historical-style local MLP with input LayerNorm, explicitly labelled as a within-model normalization ablation. |
| `angle` | Sine/cosine of a learned projection plus an independent learned linear branch from original inputs. |
| `product` | Products of learned latent features plus an independent learned linear branch from original inputs. |

The angle/product mappings are classical feature encodings, not quantum
circuits. The raw branch avoids relying solely on a periodic or product map.
Trainable dimensional reduction is not claimed to be invertible. The data
invariant is exact access to every source observation at the model boundary.

Default `small` mappings apply LayerNorm after trainable projection; MPS state
normalization and TTN merge normalization act on learned hidden states. CNN
BatchNorm similarly follows convolution. None transforms the shared cached raw
measurements. The `legacy_mlp` input-LayerNorm ablation is disclosed separately
and is not the default strict input-first architecture.

## Run architecture ablations on validation first

Each command changes only one factor relative to defaults or a frozen override
JSON. Give each factor a separate experiment ID. `--trials` is superseded by
the declared factor values in ablation mode, and all effective trials are saved.

```bash
python scripts/run_raw_native_benchmark.py --phase tune --model mps --modality cnmr --seeds 41 42 43 44 45 --ablation-factor physical_dim --experiment-id C_mps_physical --device cuda --parameter-limit 500000000
python scripts/run_raw_native_benchmark.py --phase tune --model mps --modality cnmr --seeds 41 42 43 44 45 --ablation-factor mapping --experiment-id C_mps_mapping --device cuda --parameter-limit 500000000
python scripts/run_raw_native_benchmark.py --phase tune --model ttn --modality cnmr --seeds 41 42 43 44 45 --ablation-factor merge_mode --experiment-id C_ttn_merge --device cuda --parameter-limit 500000000
```

Never open a test set for every candidate and select the best test score. Freeze
the architecture choice and scientific claims from validation first. If the B
test results have already informed C design, obtain an independent external
holdout or prospectively reserve data for C; another seed on the same molecules
does not create an independent confirmation cohort.

## Default native parameter census

| Input | CNN | MPS | TTN |
|---|---:|---:|---:|
| 10,000 positions × 2 channels | 554,837 | 2,784,457 | 296,997 |
| 1,800 positions × 2 channels | 161,237 | 555,209 | 190,853 |
| 31 peaks × 3 channels | 77,045 | 71,657 | 28,101 |

The 31-peak example is the maximum of the inspected shard, not a universal
dataset width. Position/site parameters make counts depend on input shape.
These defaults completed CPU forward/backward passes with finite gradients.
Recorded single-pass timings have no warmup and are diagnostics, not a fair
speed ranking. See `docs/validation/model_profiles.json` for the measured record.
