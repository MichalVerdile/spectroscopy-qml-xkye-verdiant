# Strict raw-native benchmark

This is a new, isolated benchmark. It does **not** reinterpret `none`,
`shared_mssd` or historical results as raw. Those paths still resample.
The new runner has only `--mode raw_native`; it never calls the historical
preprocessing module, and a runtime tripwire rejects such a call.

The hypothesis is that a tensor-network architecture may outperform a classical
baseline. No outcome is assumed. See [the protocol](EXPERIMENT_PROTOCOL.md)
before opening a final test set and [verified findings](RESEARCH_FINDINGS.md)
for what has actually been executed.

## Setup

Use Python 3.12 and a separate environment. The old `pyproject.toml` and
`requirements.txt` contain inconsistent historical pins; do not install the
project editable with dependency resolution for this standalone experiment.
Entrypoints add `src` themselves.

```powershell
python -m venv .venv-raw
.\.venv-raw\Scripts\Activate.ps1
python -m pip install -r requirements-raw-native.txt
git lfs pull --include="data/raw/*.parquet"
```

On Linux activate with `source .venv-raw/bin/activate`. On the H100 cluster use
the installed PyTorch/CUDA wheel compatible with the driver. GPU external-memory
XGBoost additionally requires a compatible CuPy wheel (for CUDA 12, for example,
`python -m pip install cupy-cuda12x`). CPU runs do not require CuPy. Every result
records the actual package versions, device, source digest and Git commit.

## First run: synthetic, full native dimensions

```powershell
.\scripts\run_raw_native_benchmark.ps1 --smoke-test --phase all --device cpu --threads 2 --experiment-id smoke
```

This runs CNN, XGBoost, MPS and TTN across all five modalities at seed 42,
using 10,000/10,000/1,800 dense points and variable peak lists. It explicitly
uses small model sizes, two neural epochs and three boosting rounds. Effective
settings are recorded. Outputs are always under `synthetic_smoke`, not `smoke`.
Small calibration sets fall back to .5. This is not a performance experiment.

Equivalent entrypoints:

```bash
bash scripts/run_raw_native_benchmark.sh --smoke-test --phase all --device cpu
python scripts/run_raw_native_benchmark.py --help
```

## Real data preparation and audit

```bash
python scripts/run_raw_native_benchmark.py --phase prepare --data-dir data/raw
```

All five modalities are validated together, even when only one model or
modality is requested. The default policy **fails** on missing spectra, NaN/Inf,
invalid SMILES, malformed peak pairs, dimension mismatches or unacceptable
float32 conversion. `--invalid-policy exclude` instead excludes the entire row
from every model and modality, with a `rejected.json` ledger. An empty observed
MS/MS list is distinct from a missing list and is allowed.

`--max-samples 128` selects the first 128 source rows in stable lexicographic file
order for a pilot; it is part of dataset identity and is not a full-data result.
Never use this switch to choose favorable samples. Source files must be real
Parquet, not Git LFS pointer text. Expected real dimensions cannot be overridden.

The builder streams Parquet twice, retains original float64 measurements in
memory maps, and stores occupancy masks, original lengths, canonical SMILES,
molecule/scaffold IDs, stable sample IDs and the 37-label vocabulary. Source
SHA-256 values, cache file hashes, input/mask hashes and implementation hashes
are verified. Metadata axes are copied from `spectrum_dimensions.json`; no
spectral values are interpolated to match an axis.

MS/MS columns accept exactly one of the repository's `msms_positive_40ev` /
`msms_negative_40ev` names or the documented `msms_cfmid_*_40ev` aliases.
Ambiguous duplicate columns cause an error. Stored peak order and both values
are retained, including duplicate m/z values, zero intensities and large m/z.

The canonical model input is:

| Source | Canonical row at one position | Padding |
|---|---|---|
| NMR/IR | `[intensity, 1]` | No source padding |
| MS/MS | `[m/z, intensity, 1]` | `[0, 0, 0]` |

A real `(0,0)` peak is `[0,0,1]`. All models receive these exact float32 values;
XGBoost only flattens the tensor. Float64 originals remain available. Conversion
must satisfy relative error ≤1e-6 with **zero absolute tolerance**; overflow and
nonzero underflow fail. No model-specific rescaling or filtering exists.

Padding width is the label-blind maximum source length, fixed before fitting.
This transductive **shape metadata** is disclosed; no test intensity statistics
or labels are used to choose a padding size or normalization reference. An
unseen deployment spectrum with a longer peak list requires a new model shape;
it must never be truncated.

Raw means **as stored in the supplied Parquet**, not an assertion that upstream
spectral simulation or normalization never occurred. In particular, metadata
already describes MS/MS intensity conventions. This benchmark adds none.

## Full optimized benchmark: tune, then final evaluation

Use the same arguments in both commands. Freeze the code and protocol first.
The following declares 12 trials, five seeds, a common soft time ceiling per
trial, and a parameter cap large enough for the full declared TN search space.

```bash
python scripts/run_raw_native_benchmark.py --phase tune --experiment-id B_optimized_raw --model all --modality all --seeds 41 42 43 44 45 --variant optimized --device cuda --trials 12 --epochs 100 --patience 15 --time-limit-per-trial 1200 --parameter-limit 500000000 --batch-size 64 --resume
python scripts/run_raw_native_benchmark.py --phase final --experiment-id B_optimized_raw --model all --modality all --seeds 41 42 43 44 45 --variant optimized --device cuda --trials 12 --epochs 100 --patience 15 --time-limit-per-trial 1200 --parameter-limit 500000000 --batch-size 64 --resume
python scripts/aggregate_raw_native.py benchmark/raw_native_v1/B_optimized_raw
```

PowerShell uses the same arguments after
`.\scripts\run_raw_native_benchmark.ps1`. `--phase all` is a convenience for a
previously frozen experiment; it fits/selects/calibrates and then evaluates.
Default `--phase tune` does not produce test predictions.

Results live in
`benchmark/raw_native_v1/<experiment>/<modality>/seed_<seed>/<model>/`.
Each family has input/code/config identity, the split and its balance report,
individual trial histories and checkpoints, a selection record, thresholds,
a seal written before test evaluation, raw probabilities/labels/sample IDs,
central metrics and an integrity-checked completion marker.

`--resume` reuses a completed run only when **all** recorded identity fields
match, including source, dataset, masks, exact split, seed, package environment,
code version, thresholds protocol, architecture, search and training limits.
Corruption or mismatches fail; they never silently overwrite results. Use a new
experiment ID for changes. A per-run file lock prevents concurrent writers.
Neural checkpoints resume weights, optimizer, scheduler, patience and RNG.
XGBoost restarts an incomplete trial deterministically; completed trials are
reused. This is explicitly not mid-round XGBoost recovery.

## Resources and cluster

Prepare the shared cache once before submitting arrays. Cache space can be much
larger than compressed Parquet: two dense NMR modalities alone require roughly
`N * 10000 * 2 * 8` bytes for original values, plus masks, IR and MS/MS. Full
datasets can require over 100 GB of cache disk. A selected model never loads
all raw spectra into Python RAM.

XGBoost `--xgb-memory-mode auto` selects external-memory histogram training when
the full training/selection feature matrix exceeds 1 GiB. `external` forces
the streamed path; `ram` uses memory-mapped staging plus in-memory DMatrix.
`--xgb-external-batch-size 4096` controls streamed batches. On CPU the external
cache uses disk. GPU uses host caching by default; `--xgb-external-cache disk`
reduces host memory pressure at a potential speed cost. Use local fast storage.
No spectral coordinates are binned or discarded before the tree algorithm.
The tree learner's ordinary histogram approximation is an architectural choice.
Final XGBoost inference is streamed in batches of 4096 rows.

```bash
mkdir -p logs
sbatch --export=ALL,EXPERIMENT_ID=B_optimized_raw,PHASE=tune,TRIALS=12 cluster/train_raw_native.sbatch
# After reviewing all selection results, without modifying code or options:
sbatch --export=ALL,EXPERIMENT_ID=B_optimized_raw,PHASE=final,TRIALS=12 cluster/train_raw_native.sbatch
```

The array covers five seeds × five modalities, three simultaneous H100 jobs,
one GPU and 64 GB host RAM each. These settings are a starting point, not a
measured full-dataset memory guarantee. Training ceilings are finite. Time caps
are checked at epoch/round boundaries and may be exceeded by one full unit.
GPU external-memory training and Slurm execution require cluster validation.

## Verification and old experiments

```bash
PYTHONPATH=src python -m pytest -o addopts='' -q
RAW_NATIVE_REAL_PARQUET=/path/to/aligned_chunk_0.parquet PYTHONPATH=src python -m pytest -o addopts='' -q tests/test_raw_native_integration.py::test_real_parquet_measurements_and_metadata
python scripts/profile_raw_native_models.py --output benchmark/raw_native_v1/model_profiles.json --device cpu
```

All old scripts, loaders, model code, result directories and dependencies remain
available. Reproduce a historical result with its original commit, environment,
configuration and preprocessing; do not use the new native runner as a claim of
bit-identical reproduction. See [comparability](MODEL_COMPARABILITY.md).
