# Spectroscopy-QML Thesis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Research: leverage quantum-mechanical origins of spectroscopic data to design better ML models.

This repository contains code and experiments for functional-group classification from multi-modal spectroscopic data (H-NMR, C-NMR, IR, MS/MS).

**Quick links:** [benchmark scripts](benchmark/), [configs](configs/), [data layout](data/), [src package](src/)

**Reproducible environment (first)**

- **Python:** 3.12 recommended.
- Preferred: create an isolated virtual environment and install pinned dependencies from `requirements.txt`.

Windows PowerShell (recommended for Windows users):

```powershell
# create venv with Python 3.12
python -m venv .venv
.venv\Scripts\Activate.ps1

# install pinned dependencies
pip install -r requirements.txt

# (optional) install dev extras if present
pip install -e .[dev]
```

macOS / Linux (bash):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
```

Notes:
- If you use `uv` as an alternative, replace the `pip` steps with `uv pip install -r requirements.txt` as needed.
- Some experiments may require GPU drivers and CUDA for PyTorch; see `requirements.txt` for the PyTorch spec and install the matching CUDA build if you plan to run on GPU.

**Reproducibility: run main experiments**

This project provides baseline experiment scripts for XGBoost and CNN benchmarks. Two convenient entry points:

- XGBoost baseline (Python):

```powershell
# PowerShell / bash
python benchmark/xgb/scripts/run_xgb_baseline.py --config configs/xgb_baseline.yaml
```

or via the included shell helper: [benchmark/xgb/run_xgboost.sh](benchmark/xgb/run_xgboost.sh#L1)

- CNN baseline (Python):

```powershell
python benchmark/cnn/scripts/run_cnn_jung_baseline.py --config configs/cnn_baseline.yaml
```

or via the helper: [benchmark/cnn/run_cnn.sh](benchmark/cnn/run_cnn.sh#L1)

Recommended reproducible workflow:

1. Prepare data under `data/` as described below.
2. Edit or confirm experiment settings in the YAML config (examples in `configs/`).
3. Run the script above and capture `results/` output (each script writes outputs under `benchmark/.../results/` or a configured `results/` path).

If you need to run cross-validation or k-fold experiments, check the `configs/` files and the `scripts/` subfolders under `benchmark/` for the available flags.

**Reproduce MPS and TTN experiments (per modality)**

The codebase contains dedicated entry scripts for the MPS and TTN experiments for each modality. Run the scripts below from the repository root. Each `run.py` or `train.py` exposes `--help` for available options; common useful flags are `--device` (auto|cpu|cuda|mps), `--data-dir`, and `--output-dir`.

- C‑NMR

   - MPS (run / train):

   ```powershell
   python src/spectroscopy_qml/mps_classifier/cnmr/run.py --train
   python src/spectroscopy_qml/mps_classifier/cnmr/run.py --evaluate
   python src/spectroscopy_qml/mps_classifier/cnmr/run.py --check
   ```

   - TTN (experiment 10.2):

   ```powershell
   python src/spectroscopy_qml/tree_tensor_network/cnmr/train.py --data-dir data/raw --output-dir benchmark/cnmr_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/tree_tensor_network/cnmr/train.py --help
   ```

- H‑NMR

   - MPS:

   ```powershell
   python src/spectroscopy_qml/mps_classifier/hnmr/run.py --train
   python src/spectroscopy_qml/mps_classifier/hnmr/run.py --evaluate
   python src/spectroscopy_qml/mps_classifier/hnmr/run.py --check
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/tree_tensor_network/hnmr/train.py --data-dir data/raw --output-dir benchmark/hnmr_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/tree_tensor_network/hnmr/train.py --help
   ```

- IR

   - MPS (the older `mps_classifier_over` variant now lives under `src/spectroscopy_qml/_old/ir/`):

   ```powershell
   python src/spectroscopy_qml/mps_classifier/ir/run.py --train
   python src/spectroscopy_qml/mps_classifier/ir/run.py --evaluate
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/tree_tensor_network/ir/train.py --data-dir data/raw --output-dir benchmark/ir_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/tree_tensor_network/ir/train.py --help
   ```

- MS/MS (positive)

   - MPS:

   ```powershell
   python src/spectroscopy_qml/mps_classifier/msms_pos/run.py --train
   python src/spectroscopy_qml/mps_classifier/msms_pos/run.py --evaluate
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/tree_tensor_network/msms_pos/train.py --data-dir data/raw --output-dir benchmark/msms_pos_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/tree_tensor_network/msms_pos/train.py --help
   ```

- MS/MS (negative)

   - MPS:

   ```powershell
   python src/spectroscopy_qml/mps_classifier/msms_neg/run.py --train
   python src/spectroscopy_qml/mps_classifier/msms_neg/run.py --evaluate
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/tree_tensor_network/msms_neg/train.py --data-dir data/raw --output-dir benchmark/msms_neg_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/tree_tensor_network/msms_neg/train.py --help
   ```

Tips:

- Use `--check` (for MPS run scripts) to validate dependencies and quick model sanity checks before launching full training.
- For reproducible results, fix `--seed` where available and capture the used config file (most training scripts write the args/config to the `--output-dir`).
- To run on GPU, set `--device cuda` and ensure your PyTorch installation matches your CUDA drivers.

**Repository structure (top-level)**

```
LICENSE
pyproject.toml
requirements.txt
benchmark/
   cnn/
      run_cnn.sh
      scripts/
         run_cnn_jung_baseline.py
      models/
   xgb/
      run_xgboost.sh
      scripts/
         run_xgb_baseline.py
      models/
configs/
data/
   raw/
   processed/
src/
   spectroscopy_qml/
tests/
README.md
```

**Data layout**

Place your data under `data/` (not tracked by git). Example:

```
data/
├── raw/                # original parquet files
└── processed/          # generated preprocessed datasets used for training
```

Config example (`configs/example_config.yaml`):

```yaml
experiment:
   name: example
   seed: 42

data:
   path: data/processed/spectra.parquet
   train_split: 0.8

model:
   type: cnn
   batch_size: 64
```

**Preprocessing and representation ablation**

The models historically used different preprocessing (CNN/XGBoost: linear
interpolation only; MPS: + Savitzky-Golay 25/4 + SNV; TTN: + SNV). Every
training entrypoint and `scripts/run_multiseed_benchmark.sh` now take a
`--preprocessing` spec (`PREPROCESSING=` for the runner). `legacy` (the default)
keeps the historical per-model behaviour; any other spec is applied identically
to all four models via `src/spectroscopy_qml/benchmarking/preprocessing.py`:

```
resample=linear|bin_mean|bin_max|fir
smooth=none|savgol:<window>:<order>|gaussian:<sigma>|moving_average:<window>|lorentz:<gamma>[:<half_width>]
repr=intensity|sqrt|log1p|derivative
norm=none|snv|minmax|max|area|l2|quantile|pqn
```

`lorentz` convolves with a normalised Lorentzian kernel (HWHM `gamma` in data
points, truncated at `±half_width`, default 15, mirror padding), identical to the
TTN `LorentzianFeatureMap` smoothing channel. `quantile` and `pqn` (probabilistic
quotient normalization) are cross-sample methods taken from the original
quantile/PQN CNN scripts. In training runs, their reference distribution/profile
is fitted exclusively on the fixed training split and then applied unchanged to
training, validation and test data. Their split-independent resampling/smoothing
output remains reusable in the shared dataset cache. Quantile/PQN runs therefore
require a fixed split artifact; CNN and XGBoost additionally require `--no_kfold`
so one unambiguous training reference is used. The benchmark runner already
supplies these options.

The default ablation runs four shared configurations: the paper reference and
one alternative method per preprocessing factor (one factor at a time), on shared
splits. It writes separate architecture, preprocessing, representation and
effect-size tables to `benchmark/preprocessing_ablation/summary/report.md`:

```
paper_lorentz_snv  reference: linear resampling, Lorentzian kernel (gamma 3, half width 15), SNV
resample_bin_max   alternative downsampling: max-pooling keeps narrow NMR/MS peaks
smooth_savgol      alternative smoothing/filtering: Savitzky-Golay 25/4 (paper MPS filter)
norm_area          alternative normalization: area (total intensity) instead of SNV
```

Input representation is compared through the input dimension (600 vs 1800).
The configurations use the same canonical specification for every spectrum
modality and all four architectures. The TTN keeps its Lorentzian feature map
inside the model, so under the reference its input is Lorentz-smoothed twice.
`legacy` (historical per-model preprocessing), `shared_mssd` (linear
interpolation only) and further one-factor variants of the reference
(`resample_bin_mean`, `resample_fir`, `smooth_none`, `smooth_gaussian`,
`smooth_moving_average`, `norm_none`, `norm_minmax`, `norm_pqn`, `norm_max`,
`norm_l2`, `norm_quantile`, `repr_sqrt`, `repr_derivative`, `repr_log1p`) are not
run by default but can be named in `CONFIGS` (`--list --include-legacy
--include-extra` shows them).

Each configuration is applied with the same specification and the same input
dimension to all five spectra (IR, H-NMR, C-NMR, MS/MS+, MS/MS-). The runner takes
exactly one input dimension per call, and the aggregation refuses results where
configurations at one input dimension cover different spectra. Note that IR
already has 1800 points, so at 1800 the `resample_*` configs give IR the same
input as the reference.

For every non-legacy run, all four models record a SHA-256 digest of their
actual preprocessed `float32` input array. Aggregation fails if these digests
differ, so architecture comparisons cannot silently use different inputs.
Existing result files are reused only when model, seed, split, input dimension,
and preprocessing spec match the requested run and a valid input digest is
present. CNN, XGBoost, MPS and TTN cache their preprocessed arrays under
`data/cache/`; cache keys include the source parquet fingerprint, modality,
input dimension and preprocessing configuration. A changed input or configuration
therefore rebuilds the cache automatically. The key also contains a SHA-256 hash
of the preprocessing implementation, so editing the code invalidates old caches.
CNN and XGBoost additionally accept
`--cache_dir` and `--overwrite_cache`.

Precompute the 4 default shared preprocessing configurations before starting any model
training (five modalities at 600 and 1800 input points):

```bash
PYTHONPATH=src python scripts/precompute_preprocessing_cache.py --dry-run
PYTHONPATH=src python scripts/precompute_preprocessing_cache.py
```

The first command only prints the plan. The second reads the raw parquet data and
populates `data/cache/preprocessed/`. To prepare a smaller experiment, select the
dimensions, modalities or configurations explicitly:

```bash
PYTHONPATH=src python scripts/precompute_preprocessing_cache.py \
  --input-dims 600 \
  --modalities ir cnmr \
  --configs paper_lorentz_snv norm_area smooth_savgol
```

Non-legacy caches are shared by CNN, XGBoost, MPS and TTN. The precompute runner
creates separate legacy variants because the historical pipelines differ between
the model families. Existing compatible caches are reused; pass `--overwrite` to
rebuild them. Before processing, the runner reports a conservative uncompressed
storage estimate and refuses to start if it exceeds the available disk space.
Prefer a smaller selection; `--allow-low-disk` overrides this guard only after
the expected NPZ compression and available storage have been checked manually.

```bash
PYTHONPATH=src python -m spectroscopy_qml.benchmarking.preprocessing --list
DEVICE=mps scripts/run_preprocessing_ablation.sh 600
DEVICE=mps scripts/run_preprocessing_ablation.sh 1800
# other configs (still all five spectra): CONFIGS="legacy paper_lorentz_snv norm_pqn" scripts/run_preprocessing_ablation.sh 600
```

Model-specific 1800-point runners execute seeds 41--45 across C-NMR, H-NMR,
IR, positive MS/MS and negative MS/MS without starting the other architectures:

```bash
scripts/run_cnn_all_modalities_1800.sh
DEVICE=cpu scripts/run_mps_all_modalities_1800.sh
DEVICE=mps scripts/run_ttn_all_modalities_1800.sh
```

They default to `PREPROCESSING=shared_mssd` and share split artifacts under
`benchmark/shared_splits_1800`. Their results share the experiment root
`benchmark/multiseed_1800/<preprocessing>` so the architectures remain directly
comparable. `PREPROCESSING`, `SEEDS`, `MODALITIES`,
`EPOCHS`, `DEVICE`, `DATA_DIR`, `SPLIT_DIR`, and `EXPERIMENT_DIR` can be
overridden through environment variables where applicable.

**Testing & quality checks**

```bash
# run tests
pytest

# lint + format
ruff check src/ benchmark/ tests/ --fix
ruff format src/ benchmark/ tests/

# type checks
mypy src/
```

**Troubleshooting & tips**

- If a script fails due to missing data, ensure the `data.path` in the config points to an existing parquet file.
- For GPU runs, ensure PyTorch and CUDA versions match your drivers.
- Logs and model checkpoints are written to the `benchmark/.../results/` folders unless overridden by config.

**Contributing**

- Create a branch: `git checkout -b feature/my-feature`
- Run `pre-commit run --all-files` and tests
- Commit and push, then open a pull request

**License**

MIT — see [LICENSE](LICENSE)
