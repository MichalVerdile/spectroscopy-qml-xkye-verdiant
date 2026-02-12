# Spectroscopy-QML Thesis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Question:** Can we leverage the quantum-mechanical origins of spectroscopic data to design better ML models?

This repository contains the research code and experiments for exploring domain-specific machine learning architectures for spectroscopic data analysis.

## Project Overview

We investigate machine learning approaches for multi-modal spectroscopic data including:
- **H-NMR** (Hydrogen Nuclear Magnetic Resonance)
- **C-NMR** (Carbon-13 Nuclear Magnetic Resonance)
- **IR** (Infrared Spectroscopy)
- **MS/MS** (Tandem Mass Spectrometry - positive/negative modes)

**Task:** Functional group classification (37 classes) from spectroscopic signatures

**Data Format:** Parquet files containing spectra + SMILES molecular representations

## Repository Structure

```
spectroscopy-qml-thesis/
├── src/spectroscopy_qml/      # Main Python package (placeholder modules)
│   ├── __init__.py
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model architectures
│   ├── training/              # Training loops and utilities
│   ├── evaluation/            # Metrics and evaluation
│   └── utils/                 # Shared utilities (logging, config, etc.)
├── jupyter/                   # Jupyter notebooks for exploration
├── benchmark/                 # Baseline implementations
│   ├── run_xgb_baseline.py    # XGBoost baseline (placeholder)
│   └── run_cnn_jung_baseline.py  # CNN baseline (placeholder)
├── tests/                     # Unit and integration tests
├── configs/                   # YAML configuration files for experiments
├── data/                      # Data directory (not tracked by git)
│   ├── raw/                   # Original Parquet files
│   ├── processed/             # Preprocessed data
│   └── README.md              # Data documentation
├── results/                   # Experiment outputs (not tracked by git)
├── .github/                   # GitHub Actions CI and templates
├── pyproject.toml             # Project metadata and tool configs
├── requirements.txt           # Pinned dependencies
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
└── README.md                  # This file
```

## Quick Start

### Prerequisites

- **Python 3.12** (required)
- **uv** (recommended) or pip for dependency management
- Git for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spectroscopy-qml-thesis.git
   cd spectroscopy-qml-thesis
   ```

2. **Install uv** (if not already installed)
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Create a virtual environment and install dependencies**
   ```bash
   # Create venv with Python 3.12
   uv venv --python 3.12

   # Activate the environment
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Windows (CMD):
   .venv\Scripts\activate.bat
   # macOS/Linux:
   source .venv/bin/activate

   # Install all dependencies (including dev tools)
   uv pip install -r requirements.txt
   uv pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Quality Checks

```bash
# Lint code (auto-fix when possible)
ruff check src/ benchmark/ tests/ --fix

# Format code
ruff format src/ benchmark/ tests/

# Type checking
mypy src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run all checks at once (as done in CI)
ruff check src/ benchmark/ tests/ && \
ruff format --check src/ benchmark/ tests/ && \
mypy src/ && \
pytest tests/
```

### Working with Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# The kernel will use the project's virtual environment
# Notebooks should be placed in the jupyter/ directory
```

**Note:** Jupyter notebooks are gitignored by default to avoid committing large outputs. Use `jupyter nbconvert --clear-output` if you need to commit notebooks.

## Dependency Management

This project uses **uv** for fast, reliable dependency management with pinned versions in `requirements.txt`.

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
uv pip compile pyproject.toml -o requirements.txt --upgrade

# Add a new dependency to pyproject.toml first, then:
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

### Alternative: Using pip-tools

If you prefer pip-tools over uv:
```bash
pip install pip-tools
pip-compile pyproject.toml -o requirements.txt
pip-sync requirements.txt
```

## Running Experiments

### Baseline Benchmarks

```bash
# XGBoost baseline (placeholder - not yet implemented)
python benchmark/run_xgb_baseline.py --config configs/xgb_baseline.yaml

# CNN baseline (placeholder - not yet implemented)
python benchmark/run_cnn_jung_baseline.py --config configs/cnn_baseline.yaml
```

### Configuration

All experiments are configured via YAML files in `configs/`. Example structure:

```yaml
# configs/example_config.yaml
experiment:
  name: "experiment_name"
  seed: 42

data:
  path: "data/processed/spectra.parquet"
  train_split: 0.8

model:
  type: "cnn"
  # ... model-specific params
```

## Data Setup

Place your spectroscopic data in the `data/` directory:

```bash
data/
├── raw/
│   └── spectra_dataset.parquet  # Original data
├── processed/
│   └── (generated files)
└── README.md  # Document your data sources and preprocessing
```

**Important:** Data files are gitignored. Document your data pipeline in `data/README.md`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run only fast tests (skip slow tests marked with @pytest.mark.slow)
pytest -m "not slow"

# Run specific test file
pytest tests/test_data_loader.py
```

## Contributing

1. Create a new branch for your feature: `git checkout -b feature/my-feature`
2. Make your changes and ensure all checks pass
3. Run `pre-commit run --all-files` to ensure code quality
4. Commit your changes: `git commit -m "Add my feature"`
5. Push to the branch: `git push origin feature/my-feature`
6. Open a Pull Request

### Code Quality Standards

- **Linting:** Ruff enforces PEP 8 style with additional rules
- **Formatting:** Ruff formatter (100 char line length)
- **Type hints:** Required for all functions (checked by mypy)
- **Tests:** Maintain >80% code coverage
- **Documentation:** Docstrings for all public APIs (Google style)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ZHAW for supporting this Bachelor's thesis research
- RDKit for molecular structure handling
- PyTorch for deep learning framework
