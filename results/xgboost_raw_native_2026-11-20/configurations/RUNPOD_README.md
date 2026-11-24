# RunPod: 25 raw-native XGBoost experiments

Scientific source is pinned to `01e270ee8ad79bb0978f96a709113df3d1f3b017`
on `ablation-further-study`. This directory adds infrastructure only. The
launcher checks that the trainer, preprocessing, split implementation and
validators are byte-identical to that reference before starting.

## Protocol

| Setting | Value |
| --- | --- |
| Model | Existing 37 independent binary XGBoost classifiers |
| Modalities | cnmr, hnmr, ir, msms_pos, msms_neg |
| Width | 10,000; IR 1,800 |
| Seeds | 41, 42, 43, 44, 45 |
| Order | One modality, all five seeds; sequential on one CUDA GPU |
| Preprocessing | `none` |
| Split | 80/10/10 multilabel stratification; no K-fold |
| Maximum rounds | 1,000,000 per label |
| Early stopping | 15 rounds; validation **log loss**, separately per label |
| Hyperparameters | learning_rate=.05, max_depth=6, subsample=.8, colsample_bytree=.8 |
| Predictions | Original XGBClassifier probability **> .5** rule; no threshold tuning |
| Learning curves | Retrospective train/validation micro/macro F1 every 20 rounds, plus final fitted step |

`none` is an alias for linear resampling with neutral remaining steps. The
preflight rejects non-native spectrum widths, so no width resampling occurs.
The existing float32 conversion and nonfinite-to-zero handling still apply.
MS/MS is not a lossless native peak representation: the existing code uses
`int(m/z * 10)`, clips positions to 0..9999, and overwrites a bin with the last
peak assigned to it. This representation is unchanged.

## Storage and environment

The actual mount is checked with `findmnt`. A RunPod **Global Volume** uses
object storage, unlike a regional Network Volume. Keep Git, the virtualenv,
local locks and caches requiring atomic rename under `/root`. Publish closed
files with SHA-256 verification; do not depend on chmod or remote file locking.
Only one pod may run this experiment plan at a time: the local lock does not
coordinate writers on different pods.

Use the official CUDA 12.8.1/PyTorch container and Python 3.12. The requirements
file contains the original pinned training dependencies and CuPy for CUDA
prediction matrices in evaluation. The template's CUDA-enabled PyTorch is reused
for the preflight. Versions and actual CUDA-device checks are recorded.
The preflight checks effective cgroup RAM, rather than trusting host `free`.

The infrastructure branch is saved on this pod and in the persistent Git
bundle. GitHub denied write access, so this branch is **not available from the
GitHub remote**. For a new pod, restore the latest verified bundle using the
restoration instructions below, then create the environment:

```bash
cd /root/spectroscopy-qml
python3.12 -m venv --system-site-packages /root/runpod-work/venv
/root/runpod-work/venv/bin/python -m pip install -r runpod/requirements-xgb.txt
```

The setup also needs `git-lfs` and `tmux`. `prepare.py` downloads only the 245
raw Parquet LFS objects, verifies their original OID hashes and sizes, copies
them to persistent storage, and validates all files/modalities. A full Git
bundle (all refs/history), current worktree archive, recovery manifest and
environment report are saved persistently. Large LFS contents are stored
separately as the verified original Parquet files; unrelated LFS objects from
historical branches are not downloaded.

## Split provenance and comparison limits

Prefer the original `benchmark/preprocessing_training/shared_splits_600/*.npz`
artifacts. No such files are versioned in the reference repository. When they
are unavailable, the unchanged `create_split_artifact` function can reconstruct
the algorithm deterministically. **Reconstruction does not establish identity
with historical runs.** A separate provenance file records this distinction,
the label digest, ordered molecule/row digest, index hashes, counts and ratios.
Identical ordered labels and molecule rows can share a reconstructed seed
artifact across modalities. Every target file is still checked with the
unchanged strict split validator before training.

Reconstruction requires explicit `ACCEPT_RECONSTRUCTED_SPLITS=1`. The user
accepted continuation after this limitation was disclosed on 2026-11-14.
Do not label reconstructed runs as historically verified. The split is
multilabel-stratified, not molecule-grouped. Canonical SMILES overlap between
train, validation and test is measured and saved; this does not prove absence
of scaffold or stereochemistry-insensitive overlap.

## Start, status and resume

The observation wrapper defers Python cyclic garbage collection during data
preparation, while ordinary reference counting stays active. It restores and
runs collection before the first estimator fits. This changes memory-management
timing only; the original loader, arrays, split validation and fitting arguments
are unchanged. A traceback timer reports the data-preparation stack every two
minutes and stops before fitting.

Replace `/workspace` only with the independently verified persistent mount.

```bash
export PERSISTENT_ROOT=/workspace
export ACCEPT_RECONSTRUCTED_SPLITS=1  # only when this limitation is accepted
bash /root/spectroscopy-qml/runpod/run_xgboost_raw_native_runpod.sh start
tmux list-sessions
tail -f /root/runpod-work/master.log
bash /root/spectroscopy-qml/runpod/run_xgboost_raw_native_runpod.sh status
```

The tmux session is `xgb-raw-native`. The user may close Jupyter, their browser
or their laptop. The pod itself must stay running and funded. No automatic
credit purchase or automatic GPU shutdown is configured.

After a pod restart, recreate the local environment and checkout (or restore
the saved bundle), attach the **same** persistent volume, and use the same
start command. Preparation revalidates persisted data. Completed experiments
are skipped only after their configuration and artifact hashes validate.
Each incomplete retry gets a new attempt directory; old outputs are retained.
When all 37 saved fits and their original arguments match a retry, the runner
verifies their hashes and metadata and restores them before scoring.
The first failed experiment stops the batch to avoid paying for repeated
systematic failures. Fix the reported problem, then restart the launcher.

On the same already validated pod, when the dataset and scientific files have
not changed, `XGB_REUSE_READY=1` can reuse the completed preflight. The runner
still checks source, audit, manifest and split hashes. Do not use that shortcut
for a new or modified data environment. Each restart archives the previous
master log before opening a new one.

```text
<mount>/data/raw/                            Original 245 Parquet files
<mount>/data/cache/                          Closed native-width cache copies
<mount>/shared_splits/                       25 split files and provenance
<mount>/repository-backup/<timestamp>/       Git bundle, source archive, recovery
<mount>/experiments/xgboost_raw_native/
    configurations/                         Environment, audit and readiness
    <modality>/seed_<seed>/attempts/<id>/     Immutable attempt directories
        artifacts/                          Original model/results/curves
        label_models/                       Per-label UBJ, loss history and GPU config
        config.json, console.log, status.json, completed.json
    logs/master.log                         Last closed setup/runner snapshot
    summaries/experiments.csv                All 25 rows; missing values stay empty
    summaries/status.json                   Current counts and attempts
```

Status/log snapshots refresh about every 30 seconds. A stale running heartbeat
is reported as interrupted after two minutes. Whole-object replacement of the
latest status files is not a claim of atomic transactions on Global Volumes;
the completion marker is written last, with hashes of all completed artifacts.

Per-label fitted models and validation loss histories are saved immediately
after each label finishes. After all 37 fits, evaluation reuses each exact CUDA
DMatrix for all models and predicts with an increasing tree limit. It aggregates
per-label counts over the full train and validation splits. Scores close to
0.5 are checked with the original classifier to protect the strict threshold
against rounding. Final predictions use each model's own best iteration.
The console reports evaluation progress and elapsed time for each split.

A retry from 37 verified fits records their source and original measured fit
times in `fit_recovery.json`. Partial fits are retained for inspection. The
`best_iteration` column lists one value per functional group. For restored fits,
`training_seconds` sums the saved per-label timings. For new fits it uses the
original trainer's fit timer. Both exclude retrospective F1 curve computation.

## Restoration without GitHub

Copy `repository.bundle` from the persistent backup to local storage and verify
its SHA-256 against `recovery.json`. Then:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone /root/repository.bundle /root/spectroscopy-qml
cd /root/spectroscopy-qml
git switch runpod/xgboost-raw-native
```

Use the commit/branch recorded in the selected recovery manifest. The bundle
contains all fetched repository refs/history; `worktree.tar.gz` is an additional
source snapshot. The original LFS OID values remain verifiable against
`data/dataset_manifest.json`. Never replace verified data with pointer files.
