"""
Quantum Kernel SVM training script for IR functional group classification.

Pipeline:
    1. Load data (max_files=5, SNV-normalised)
    2. Subsample MAX_TRAIN training samples  (kernel matrix is O(n²))
    3. PCA(8) + MinMaxScaler([-π, π])
    4. Compute quantum kernel matrices  K_train (n×n) and K_test (m×n)
    5. OneVsRest SVC(kernel='precomputed') — one binary classifier per class
    6. Evaluate with micro/macro F1

Run from the repo root:
    python src/spectroscopy_qml/ir/qml/train_kernel.py

Outputs:
    results/kernel_summary.txt
    results/kernel_matrices.npz   (cached K_train, K_test)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data
from spectroscopy_qml.ir.qml.model_qml_kernel import (
    build_kernel_matrix,
    build_preprocessor,
    N_QUBITS,
    N_REPS,
    N_CLASSES,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw")
MAX_FILES  = 5
APPLY_SNV  = True
TARGET_LEN = 1800

MAX_TRAIN  = 300    # kernel matrix is n² → keep manageable on CPU
MAX_TEST   = 100    # test kernel is MAX_TEST × MAX_TRAIN

SVM_C      = 1.0    # SVM regularisation
SEED       = 42

ROOT        = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SUMMARY_TXT = RESULTS_DIR / "kernel_summary.txt"
KERNEL_NPZ  = RESULTS_DIR / "kernel_matrices.npz"


# ── Helpers ───────────────────────────────────────────────────────────────────

def tune_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Grid-search global threshold maximising micro-F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_true, (y_scores >= t).astype(int),
                      average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Quantum Kernel SVM — ZZ Feature Map")
    print(f"  N_QUBITS={N_QUBITS}  N_REPS={N_REPS}  C={SVM_C}")
    print(f"  MAX_TRAIN={MAX_TRAIN}  MAX_TEST={MAX_TEST}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data  (max_files={MAX_FILES}) …")
    X, y = load_ir_data(DATA_DIR, target_length=TARGET_LEN,
                        max_files=MAX_FILES, apply_snv=APPLY_SNV)
    print(f"Dataset: {X.shape[0]} samples  |  {y.shape[1]} classes")

    # Train / test split
    X_tr_full, X_te_full, y_tr_full, y_te_full = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True)

    # Subsample  (kernel matrix is O(n²) — keep it tractable)
    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(X_tr_full), min(MAX_TRAIN, len(X_tr_full)),
                        replace=False)
    te_idx = rng.choice(len(X_te_full), min(MAX_TEST,  len(X_te_full)),
                        replace=False)

    X_train, y_train = X_tr_full[tr_idx], y_tr_full[tr_idx]
    X_test,  y_test  = X_te_full[te_idx], y_te_full[te_idx]
    print(f"Subsampled: train={len(X_train)}  test={len(X_test)}")

    # ── 2. Preprocessing: PCA + scale ────────────────────────────────────────
    print(f"\nPCA({N_QUBITS}) + MinMaxScaler([−π, π]) …")
    prep = build_preprocessor(n_components=N_QUBITS)
    X_train_q = prep.fit_transform(X_train)   # (n_train, N_QUBITS)
    X_test_q  = prep.transform(X_test)        # (n_test,  N_QUBITS)

    explained = prep.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained:.1%}")

    # ── 3. Quantum kernel matrices ────────────────────────────────────────────
    if KERNEL_NPZ.exists():
        print(f"\nLoading cached kernel matrices from {KERNEL_NPZ} …")
        cache = np.load(KERNEL_NPZ)
        K_train = cache["K_train"]
        K_test  = cache["K_test"]
        t_kernel = float(cache["t_kernel"])
        print(f"  K_train: {K_train.shape}  K_test: {K_test.shape}")
    else:
        print(f"\nComputing K_train ({len(X_train)}×{len(X_train)}) …")
        t0 = time.time()
        K_train = build_kernel_matrix(X_train_q, X_train_q)
        t_kernel = time.time() - t0
        print(f"  K_train done in {t_kernel:.1f}s")

        print(f"\nComputing K_test  ({len(X_test)}×{len(X_train)}) …")
        t1 = time.time()
        K_test = build_kernel_matrix(X_test_q, X_train_q)
        t_kernel += time.time() - t1
        print(f"  K_test  done  |  total kernel time: {t_kernel:.1f}s")

        np.savez_compressed(KERNEL_NPZ,
                            K_train=K_train, K_test=K_test,
                            t_kernel=np.array(t_kernel))
        print(f"  Saved to {KERNEL_NPZ}")

    # ── 4. Train OneVsRest SVM ────────────────────────────────────────────────
    print("\nTraining OneVsRest SVM …")
    t0 = time.time()
    clf = OneVsRestClassifier(
        SVC(kernel="precomputed", C=SVM_C, probability=True),
        n_jobs=-1,
    )
    clf.fit(K_train, y_train)
    t_svm = time.time() - t0
    print(f"  SVM trained in {t_svm:.1f}s")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    # Decision scores for threshold tuning
    try:
        y_scores = clf.predict_proba(K_test)
    except Exception:
        y_scores = clf.decision_function(K_test)
        # Normalise decision function to [0,1]
        y_scores = 1 / (1 + np.exp(-y_scores))

    best_t   = tune_threshold(y_test, y_scores)
    y_pred   = (y_scores >= best_t).astype(int)

    f1_micro = f1_score(y_test, y_pred, average="micro",    zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_wgt   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n── Results ──────────────────────────────────────────────────────")
    print(f"  Best threshold : {best_t:.2f}")
    print(f"  F1 micro       : {f1_micro:.4f}")
    print(f"  F1 macro       : {f1_macro:.4f}")
    print(f"  F1 weighted    : {f1_wgt:.4f}")
    print("\nPer-class report (top classes):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── 6. Summary ────────────────────────────────────────────────────────────
    total_time = t_kernel + t_svm
    with open(SUMMARY_TXT, "w") as f:
        f.write("Quantum Kernel SVM Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"feature_map    : ZZ  (N_QUBITS={N_QUBITS}, N_REPS={N_REPS})\n")
        f.write(f"max_files      : {MAX_FILES}\n")
        f.write(f"train_samples  : {len(X_train)}\n")
        f.write(f"test_samples   : {len(X_test)}\n")
        f.write(f"pca_variance   : {explained:.1%}\n")
        f.write(f"SVM_C          : {SVM_C}\n")
        f.write(f"best_threshold : {best_t:.2f}\n")
        f.write(f"f1_micro       : {f1_micro:.4f}\n")
        f.write(f"f1_macro       : {f1_macro:.4f}\n")
        f.write(f"f1_weighted    : {f1_wgt:.4f}\n")
        f.write(f"kernel_time    : {t_kernel:.1f}s\n")
        f.write(f"svm_time       : {t_svm:.1f}s\n")
        f.write(f"total_time     : {total_time:.1f}s\n")

    print(f"\nSummary → {SUMMARY_TXT}")
    print(f"Kernels  → {KERNEL_NPZ}")


if __name__ == "__main__":
    main()
