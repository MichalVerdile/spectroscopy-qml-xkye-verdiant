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
   python src/spectroscopy_qml/cnmr/mps_classifier_cnmr/run.py --train
   python src/spectroscopy_qml/cnmr/mps_classifier_cnmr/run.py --evaluate
   python src/spectroscopy_qml/cnmr/mps_classifier_cnmr/run.py --check
   ```

   - TTN (experiment 10.2):

   ```powershell
   python src/spectroscopy_qml/cnmr/tree_tensor_network/train.py --data-dir data/raw --output-dir benchmark/cnmr_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/cnmr/tree_tensor_network/train.py --help
   ```

- H‑NMR

   - MPS:

   ```powershell
   python src/spectroscopy_qml/hnmr/mps_classifier_hnmr/run.py --train
   python src/spectroscopy_qml/hnmr/mps_classifier_hnmr/run.py --evaluate
   python src/spectroscopy_qml/hnmr/mps_classifier_hnmr/run.py --check
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/hnmr/tree_tensor_network/train.py --data-dir data/raw --output-dir benchmark/hnmr_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/hnmr/tree_tensor_network/train.py --help
   ```

- IR

   - MPS (two variants exist: `mps_classifier` and `mps_classifier_over` — use the one matching your config):

   ```powershell
   python src/spectroscopy_qml/ir/mps_classifier/run.py --train
   python src/spectroscopy_qml/ir/mps_classifier/run.py --evaluate
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/ir/tree_tensor_network/train.py --data-dir data/raw --output-dir benchmark/ir_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/ir/tree_tensor_network/train.py --help
   ```

- MS/MS (positive)

   - MPS:

   ```powershell
   python src/spectroscopy_qml/msms_pos/mps_classifier_msms_pos/run.py --train
   python src/spectroscopy_qml/msms_pos/mps_classifier_msms_pos/run.py --evaluate
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/msms_pos/tree_tensor_network/train.py --data-dir data/raw --output-dir benchmark/msms_pos_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/msms_pos/tree_tensor_network/train.py --help
   ```

- MS/MS (negative)

   - MPS:

   ```powershell
   python src/spectroscopy_qml/msms_neg/mps_classifier_msms_neg/run.py --train
   python src/spectroscopy_qml/msms_neg/mps_classifier_msms_neg/run.py --evaluate
   ```

   - TTN:

   ```powershell
   python src/spectroscopy_qml/msms_neg/tree_tensor_network/train.py --data-dir data/raw --output-dir benchmark/msms_neg_ttn_results --device auto --epochs 100
   python src/spectroscopy_qml/msms_neg/tree_tensor_network/train.py --help
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
