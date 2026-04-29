# MPS + TTN Training Export

This package contains the minimal code needed to train the IR `MPS_TTN` pipeline
outside the main repository.

## Included

- Joint runner: `src/spectroscopy_qml/ir/MPS_TTN/run.py`
- MPS branch: `src/spectroscopy_qml/ir/mps_encoder_final/*`
- TTN 10.2 branch: `src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/*`
- Required TTN support modules from experiments 5, 6, and 10
- Minimal `requirements.txt`
- Convenience launch scripts:
  - `train_both.sh`
  - `train_sequential.sh`
  - `train_mps.sh`
  - `train_ttn.sh`

## Not Included

- Raw data
- Existing checkpoints
- Result artifacts from the main repository

Provide your parquet files under:

```text
data/raw/
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Run

Train both branches:

```bash
./train_both.sh
```

Train MPS first and then TTN automatically in two separate processes:

```bash
./train_sequential.sh
```

Train only MPS:

```bash
./train_mps.sh
```

Train only TTN:

```bash
./train_ttn.sh
```

## Notes

- The default output root is `src/spectroscopy_qml/ir/MPS_TTN/results/export_run`.
- The shared spectra cache and shared split are created automatically inside that result root.
- For CUDA hosts, edit the shell scripts if you want different batch sizes or devices.
