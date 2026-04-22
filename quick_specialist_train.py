#!/usr/bin/env python3
"""Quick Specialist Training - No BS, Just Train"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("🚀 QUICK SPECIALIST TRAINING START")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1️⃣ Loading data (≤70 files)...")
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (
    load_ir_data,
)

X, y = load_ir_data(data_dir='data/raw', target_length=1800, max_files=70, apply_snv=True)
print(f"   ✅ Loaded: X={X.shape}, y={y.shape}")

# Filter to 10 specialist labels
specialist_indices = [34, 10, 19, 27, 33, 23, 24, 25, 29, 36]
y = y[:, specialist_indices]
print(f"   ✅ Specialist labels: y={y.shape}")

# Simple train/val split (80/20)
n_train = int(len(X) * 0.8)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]
print(f"   ✅ Train: {X_train.shape}, Val: {X_val.shape}")

# ============================================================================
# 2. BUILD MODEL
# ============================================================================
print("\n2️⃣ Building TTN Model (10 labels)...")
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (
    TTNIRClassifier10_2,
)

model = TTNIRClassifier10_2(
    num_labels=10,
    chi=64,
    input_dim=1800,
    segment_window_size=60,
    segment_stride=60,
)
print(f"   ✅ Model created: parameters={sum(p.numel() for p in model.parameters())}")

# ============================================================================
# 3. SETUP TRAINING
# ============================================================================
print("\n3️⃣ Setup training...")
device = "cpu"
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.BCEWithLogitsLoss()
print(f"   ✅ Device: {device}")

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
print("\n4️⃣ Training (20 epochs)...")
print("=" * 60)

batch_size = 256
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    # Train
    model.train()
    train_loss = 0
    n_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        X_batch = torch.from_numpy(X_train[i:i+batch_size]).float().to(device)
        y_batch = torch.from_numpy(y_train[i:i+batch_size]).float().to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        n_batches += 1
    
    train_loss /= n_batches
    
    # Validate
    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().to(device)
        val_logits = model(X_val_t)
        val_loss = criterion(val_logits, y_val_t).item()
    
    # Print
    print(f"Epoch {epoch:2d}/20 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("=" * 60)
print("✅ TRAINING COMPLETE!")
print("\n🎯 Specialist Model: Ready for Ensemble!")
