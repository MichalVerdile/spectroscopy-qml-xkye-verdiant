#!/usr/bin/env python3
"""Ensemble Inference: CNN + TTN 10.2 (No Specialist)"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import tensorflow as tf

print("\n" + "="*80)
print("ENSEMBLE INFERENCE: CNN + TTN 10.2")
print("="*80)

# ============================================================================
# 1. LOAD TEST DATA
# ============================================================================
print("\n1️⃣ Loading test data...")
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (
    load_ir_data,
    load_or_create_split_indices,
)

X, y = load_ir_data(data_dir='data/raw', target_length=1800, apply_snv=True)
print(f"   Loaded: X={X.shape}, y={y.shape}")

# Use cached split if available
split_path = Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260420_083510_focal_g2_percentile/data_split_seed42_test.npz")

if split_path.exists():
    print(f"   Loading test split from cache...")
    data = np.load(split_path)
    test_indices = data['test_indices']
    X_test, y_test = X[test_indices], y[test_indices]
    print(f"   ✅ Test set: X_test={X_test.shape}, y_test={y_test.shape}")
else:
    # Create split (80/10/10)
    split_path = Path("data_split_seed42_test.npz")
    split_indices = load_or_create_split_indices(
        labels=y,
        split_path=split_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        stratify_multilabel=True,
    )
    test_indices = split_indices['test']
    X_test, y_test = X[test_indices], y[test_indices]
    print(f"   ✅ Created test split: X_test={X_test.shape}, y_test={y_test.shape}")

# ============================================================================
# 2. LOAD CNN MODEL
# ============================================================================
print("\n2️⃣ Loading CNN model...")
import pickle

cnn_pickle_path = Path("benchmark/cnn/models/ir/original/results.pickle")
if cnn_pickle_path.exists():
    try:
        with open(cnn_pickle_path, 'rb') as f:
            cnn_results = pickle.load(f)
        
        # Check if predictions are cached
        if 'test_predictions' in cnn_results:
            y_pred_cnn = cnn_results['test_predictions']
            print(f"   ✅ CNN predictions loaded from pickle: {y_pred_cnn.shape}")
        else:
            print(f"   ⚠️  Pickle has: {list(cnn_results.keys())}")
            cnn_results = None
    except Exception as e:
        print(f"   ❌ Failed to load CNN pickle: {e}")
        cnn_results = None
        
    # Fallback: try keras
    if cnn_results is None:
        cnn_keras_path = Path("benchmark/cnn/models/ir/original/ir_model.keras")
        try:
            import tensorflow as tf
            cnn_model = tf.keras.models.load_model(cnn_keras_path, compile=False)
            print(f"   ✅ CNN Keras model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load CNN Keras: {e}")
            cnn_model = None
else:
    print(f"   ❌ CNN not found")
    cnn_model = None
    cnn_results = None

# ============================================================================
# 3. LOAD TTN 10.2 MODEL
# ============================================================================
print("\n3️⃣ Loading TTN 10.2 model...")
ttn_path = Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260420_083510_focal_g2_percentile/ttn_ir_best.pt")

if ttn_path.exists():
    try:
        from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (
            TTNIRClassifier10_2,
            DEFAULT_SEGMENT_WINDOW_SIZE,
            DEFAULT_SEGMENT_STRIDE,
        )
        ttn_model = TTNIRClassifier10_2(
            num_labels=37,
            chi=64,
            input_dim=1800,
            segment_window_size=DEFAULT_SEGMENT_WINDOW_SIZE,
            segment_stride=DEFAULT_SEGMENT_STRIDE,
        )
        ttn_model.load_state_dict(torch.load(ttn_path, map_location='cpu'))
        ttn_model.eval()
        print(f"   ✅ TTN 10.2 loaded from {ttn_path}")
    except Exception as e:
        print(f"   ❌ Failed to load TTN: {e}")
        ttn_model = None
else:
    print(f"   ❌ TTN not found at {ttn_path}")
    ttn_model = None

# ============================================================================
# 4. INFERENCE
# ============================================================================
print("\n4️⃣ Running inference on test set...")

y_pred_ensemble = None

# CNN Inference (from pickle or model)
if cnn_results is not None and 'test_predictions' in cnn_results:
    y_pred_cnn = cnn_results['test_predictions']
    print(f"   ✅ Using CNN predictions from cache: {y_pred_cnn.shape}")
    y_pred_ensemble = y_pred_cnn
elif cnn_model is not None:
    print("   Running CNN inference...")
    try:
        y_pred_cnn = cnn_model.predict(X_test, batch_size=128, verbose=0)
        print(f"   ✅ CNN predictions: {y_pred_cnn.shape}")
        y_pred_ensemble = y_pred_cnn
    except Exception as e:
        print(f"   ❌ CNN inference failed: {e}")

# TTN Inference
if ttn_model is not None:
    print("   Running TTN 10.2 inference...")
    try:
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float()
            y_pred_ttn = ttn_model(X_test_t).numpy()
        print(f"   ✅ TTN predictions: {y_pred_ttn.shape}")
        
        # Blend
        if y_pred_ensemble is not None:
            print("   Blending CNN (0.5) + TTN (0.5)...")
            y_pred_ensemble = 0.5 * y_pred_ensemble + 0.5 * y_pred_ttn
        else:
            y_pred_ensemble = y_pred_ttn
    except Exception as e:
        print(f"   ❌ TTN inference failed: {e}")

# ============================================================================
# 5. EVALUATION
# ============================================================================
print("\n5️⃣ Evaluation...")
if y_pred_ensemble is not None:
    y_pred_binary = (y_pred_ensemble > 0.5).astype(int)
    
    # Metrics
    f1_micro = f1_score(y_test, y_pred_binary, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    hamming = hamming_loss(y_test, y_pred_binary)
    exact_match = accuracy_score(y_test, y_pred_binary)
    
    print(f"\n📊 ENSEMBLE RESULTS:")
    print(f"{'='*60}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"F1-Score (Micro):    {f1_micro:.4f}")
    print(f"F1-Score (Macro):    {f1_macro:.4f}")
    print(f"Hamming Loss:        {hamming:.4f}")
    print(f"Exact Match:         {exact_match:.4f}")
    print(f"{'='*60}")
    
    print(f"\n✅ ENSEMBLE COMPLETE: F1 = {f1_weighted:.4f}")
    if cnn_model is not None or cnn_results is not None:
        print(f"   CNN: Available")
    if ttn_model is not None:
        print(f"   TTN 10.2: Available")
    print(f"\n🎯 Expected with Specialist: F1 > 90%")
else:
    print("❌ No predictions generated!")

print("\n" + "="*80)
