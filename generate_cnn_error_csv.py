#!/usr/bin/env python3
"""Generate CNN baseline error analysis CSV"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("CNN BASELINE ERROR ANALYSIS - CSV GENERATION")
print("="*80)

# Functional groups mapping  
FUNCTIONAL_GROUPS = {
    0: 'Alkane', 1: 'Alkene', 2: 'Alkyne', 3: 'Aromatic', 4: 'Alcohol', 5: 'Ether', 
    6: 'Thiol', 7: 'Sulfide', 8: 'Disulfide', 9: 'Amine', 10: 'Azo', 11: 'Imine', 
    12: 'Nitrile', 13: 'Isocyanate', 14: 'Carboxylic acid', 15: 'Ester', 16: 'Ketone', 
    17: 'Aldehyde', 18: 'Amide', 19: 'Hydrazone', 20: 'Ozone', 21: 'Peroxide', 
    22: 'Peroxy compound', 23: 'Acid anhydride', 24: 'Imine', 25: 'Enamine', 
    26: 'Acetal', 27: 'Phosphine', 28: 'Thioketone', 29: 'Acyl halide', 
    30: 'Sulfoxide', 31: 'Sulfonamide', 32: 'Phosphite', 33: 'Sulfoxide', 
    34: 'Thial', 35: 'Phosphide', 36: 'Sulfide'
}

print("\n1️⃣ Loading data...")
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (
    load_ir_data
)

X, y = load_ir_data(data_dir='data/raw', target_length=1800, apply_snv=True)
n_total = len(X)
n_train = int(n_total * 0.8)
n_val = int((n_total - n_train) * 0.5) + n_train
X_test, y_test = X[n_val:], y[n_val:]
print(f"   ✅ Test set: {X_test.shape}, {y_test.shape}")

print("\n2️⃣ Loading CNN model...")
import tensorflow as tf
from tensorflow import keras

cnn_model_path = Path("benchmark/cnn/models/ir/original/ir_model.keras")
if not cnn_model_path.exists():
    print(f"   ❌ CNN model not found: {cnn_model_path}")
    sys.exit(1)

try:
    cnn_model = keras.models.load_model(str(cnn_model_path))
    print(f"   ✅ CNN model loaded from Keras")
except Exception as e:
    print(f"   ⚠️ Keras loading failed: {e}")
    print("   Trying pickle fallback...")
    import pickle
    pickle_path = Path("benchmark/cnn/models/ir/original/ir_model.pkl")
    with open(pickle_path, 'rb') as f:
        cnn_model = pickle.load(f)
        print(f"   ✅ CNN model loaded from pickle")

print("\n3️⃣ Running CNN inference...")
batch_size = 1024
y_pred_cnn = []

for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i+batch_size]
    y_batch = cnn_model.predict(X_batch, verbose=0)
    y_pred_cnn.append(y_batch)
    if (i // batch_size + 1) % 20 == 0:
        print(f"   [...{i+batch_size}/{len(X_test)}...]")

y_pred_cnn = np.vstack(y_pred_cnn)
y_pred_cnn_binary = (y_pred_cnn > 0.5).astype(int)
print(f"   ✅ Predictions: {y_pred_cnn.shape}")

print("\n4️⃣ Computing per-label metrics...")
from sklearn.metrics import f1_score, precision_score, recall_score

cnn_results = []
for label_idx in range(37):
    y_true_label = y_test[:, label_idx]
    y_pred_label = y_pred_cnn_binary[:, label_idx]
    
    f1 = f1_score(y_true_label, y_pred_label, zero_division=0)
    precision = precision_score(y_true_label, y_pred_label, zero_division=0)
    recall = recall_score(y_true_label, y_pred_label, zero_division=0)
    
    error_rate = (np.sum((y_true_label != y_pred_label)) / len(y_true_label)) * 100
    n_samples = np.sum(y_true_label)
    
    cnn_results.append({
        'label_idx': label_idx,
        'label_name': FUNCTIONAL_GROUPS.get(label_idx, f"Label{label_idx}"),
        'n_samples': n_samples,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'error_rate': error_rate,
    })

cnn_df = pd.DataFrame(cnn_results)
print(f"   ✅ Metrics computed for {len(cnn_df)} labels")

print("\n5️⃣ Saving CSV...")
cnn_df.to_csv('cnn_baseline_error_analysis.csv', index=False)
print(f"   ✅ Saved: cnn_baseline_error_analysis.csv")

print("\n6️⃣ Summary:")
print(cnn_df.sort_values('error_rate', ascending=False).head(10))

print("\n" + "="*80)
