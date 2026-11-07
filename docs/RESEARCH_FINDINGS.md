# Engineering findings and outstanding research

Audit base: `ablation-further-study` at
`3cd25ab398293f519562858e425b7c346683fe69`. Implementation branch:
`research/raw-native-fair-tensor-network-benchmark`. The original source branch,
model modules, scripts, dependencies and archived results are preserved.

No full-data five-seed result is reported. Nothing in the engineering tests
establishes that a tensor network outperforms CNN or XGBoost.

## Findings in the original execution paths

| Finding / source | Consequence | Treatment in the new path |
|---|---|---|
| `benchmarking/preprocessing.py` and old scripts describe `none`/`shared_mssd` as linear interpolation | Those configurations are not native raw data | Independent loader with original dimensions; preprocessing tripwire and invariance tests |
| `make_msms_spectrum` bins into integer m/z positions, clamps range and overwrites collisions | Original peak pairs, duplicates and coordinates can be lost | Original ordered pairs, explicit occupancy mask, no binning/truncation |
| Legacy loaders clean nonfinite values and filter per modality | Models/modalities can silently see different measurements or rows | All-five-modality cohort validation, common rejection ledger or hard error |
| Existing split caches do not establish molecule-group isolation | Duplicate molecule rows may cross evaluation partitions | Conservative molecule grouping, optional scaffold grouping and leakage assertions |
| MPS threshold optimization's micro-F1 loop is commented out | Threshold remains .5 despite the optimization option | Shared tested bounded global and per-class calibration |
| TTN per-class threshold branch returns constant .5 values | Per-class mode is a stub | Same central functioning calibrator as every other model |
| CNN val-loss, XGBoost per-label log loss, MPS micro F1 and TTN blended F1 select checkpoints differently | Architecture comparisons conflate selection policies | Common fixed-.5 selection micro F1, explicit joint XGB checkpoints |
| TTN fixed Lorentz/derivative feature map and state normalization | External raw mode alone does not remove fixed internal signal filtering | Separate native trainable feature encoder and architecture name |
| TTN preflight routines call `optimizer.step()` on the passed model | Preflight can alter the initialization used for training | New runner does no preflight updates; profiling uses disposable models |
| Native CNN flatten/dense layers and legacy MPS dimensions are very large | Memory and parameter efficiency differ substantially | New scalable CNN/MPS plus explicit preallocation caps and historical adapters |
| Cluster/multiseed scripts allow 1,000,000 epochs/rounds | Uncontrolled practical run ceilings | Separate finite runner and Slurm configuration |
| Historical requirements and pyproject dependency pins disagree | Environment recreation can fail before training | Isolated `requirements-raw-native.txt`, actual environment recorded |

These are fixed or avoided **in the new raw experiment**. Historical threshold
and preflight code is deliberately not rewritten, preserving old-run behavior.
Invoking old scripts does not gain the new fixes. The inspected source already
has XGBoost per-label validation forwarding; it is retained conceptually and
tested against independently trained binary references in the new trainer.

## Implementation map

| Files | Responsibility |
|---|---|
| `raw_native.py`, `raw_labels.py`, `raw_guard.py` | Canonical cohort, exact SMARTS, streaming memmaps, values/masks/source hashes, preprocessing rejection |
| `raw_splits.py` | Deterministic grouped split and stratification quality |
| `raw_evaluation.py` | Shared checkpoint, threshold, metric and calibration definitions |
| `raw_models/neural.py` | Versioned CNN/MPS/TTN, mappings and explicit historical adapters |
| `raw_training.py`, `raw_xgb_memory.py`, `raw_resources.py` | Shared neural loop, joint multi-label boosting, CPU/GPU external-memory path and resource telemetry |
| `raw_experiments.py`, `raw_runner.py`, `raw_smoke.py` | Search/ablation schedules, strict identity/resume, test sealing, synthetic fixtures |
| `raw_aggregate.py` | Verified paired summaries, rare-label analysis and efficiency figures |
| `scripts/*raw_native*`, `cluster/train_raw_native.sbatch` | Python/Bash/PowerShell, parameter profiler and H100 array entrypoints |
| `tests/test_raw_native*`, `.github/workflows/raw-native.yml` | Contracts, numerical/recovery/integration tests and Linux/Windows CI definitions |

## Data actually inspected

One real repository shard, `data/raw/aligned_chunk_0.parquet`, was downloaded
from Git LFS and its 48,477,068-byte content verified against SHA-256
`d98564fbd5e7b92d79920672c2649bdb62ebbabba973b187f8a2c8bc21d55c73`.
All 3,235 rows passed the new common data validation, with no rejected rows.
Observed dense lengths are exactly 10,000 C-NMR, 10,000 H-NMR and 1,800 IR.
Both 40 eV MS/MS lists range from 1 to 31 original peaks in this shard.
This verifies the metadata for that shard only; all remaining shards must pass
the same validation before a full run.

The first 128 rows were also compared directly against Parquet: exact stored
float64 values, original peak order/counts, masks and metadata axes. Canonical
float32 conversion obeys the documented relative tolerance. A real CPU pilot
completed all 20 family/modality combinations with one neural epoch, three
boosting rounds, seed 42 and reduced model widths. It exercised actual fitting,
selection, calibration fallback, prediction and result serialization. Those
tiny-run scores are deliberately not presented as scientific model rankings.

## Executed checks

The checked-in `validation/verification.json` records the final test command,
actual outcome and environment. Coverage includes the existing suite plus raw
invariance, every input shape, all 20 synthetic end-to-end combinations at full
native dense lengths, threshold/non-stub tests, group leakage, corruption and
incompatible-reuse rejection, exact neural interrupt/resume, CPU external-memory
XGBoost, per-label validation references and the optional real-Parquet test.

The existing suite passed before adding the raw implementation (87 tests).
The final whole suite passed **165 tests, zero failures and zero skips in
63.26 seconds**, including the optional real-source test. The final real pilot
also completed all 20 combinations, and its partial aggregation passed.
Ruff, basic mypy, Python compilation, Python/Bash CLI help and Bash/Slurm syntax
were also exercised. `validation/model_profiles.json` records exact original
MPS parameter counts and finite native forward/backward measurements on CPU.

## Not executed and remaining limits

- Full multi-shard, five-seed training and the proposed 12-trial searches.
- GPU/H100 training, legacy MPS GPU memory/runtime/gradient profiling, GPU
  external-memory XGBoost and Slurm submission. Telemetry and runners exist;
  values from unavailable hardware are null, not invented measurements.
- Native Windows PowerShell 5.1/7 runtime checks. The wrapper is provided and
  Windows CI jobs are defined; they have not run in this Linux environment.
- A complete strict historical four-model baseline: TTN's fixed feature map
  is incompatible and some historical native dimensions/costs are infeasible.
- Full-dataset scalability of grouped splitting, cache disk use, maximum peak
  length and external-memory resource needs. One-shard validation cannot
  certify all 245 source shards. GPU external memory additionally needs CuPy.
- Training-quality conclusions for any architecture, mapping or dimension;
  forward/backward stability does not establish better generalization.
- Independent inferential evidence from five seeds. Repeated test splits
  overlap; the supplied intervals and sign-flip calculations are descriptive.
- Bit-identical reproduction across hardware/software releases; recorded
  environments and strict reuse prevent silent comparisons of changed setups.

The next action is validation on the target GPU and Windows environment,
followed by full cohort preparation and inspection of group/label balance.
Run feasible A controls, predeclare B search budgets and C hypotheses, tune on
selection partitions, freeze all choices, and only then evaluate held-out data.
Maintain an independent confirmation cohort if test feedback informs later C
development. Preprocessing studies belong to a subsequent, separate protocol.
