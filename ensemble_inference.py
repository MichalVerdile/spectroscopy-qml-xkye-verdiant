#!/usr/bin/env python3
"""
ENSEMBLE INFERENCE: Baseline (37 Labels) + Specialist (10 Rare Labels)
Combines predictions with weighted averaging for rare groups
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Add repo to path
REPO_DIR = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant")
sys.path.insert(0, str(REPO_DIR / "src"))

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import TTNIRClassifier10_2
from spectroscopy_qml.utils.data_loader import load_processed_data

# ============================================================================
# CONFIGURATION
# ============================================================================

BASELINE_MODEL_PATH = REPO_DIR / "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/ttn_ir_best.pt"
SPECIALIST_MODEL_PATH = REPO_DIR / "src/spectroscopy_qml/ir/tree_tensor_network/experiment/specialist/results/ttn_ir_best.pt"

# 10 Rare Functional Group indices in specialist
SPECIALIST_INDICES = [34, 10, 19, 27, 33, 23, 24, 25, 29, 36]

# Ensemble weights
BASELINE_WEIGHT = 0.3
SPECIALIST_WEIGHT = 0.7

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_baseline_model(device="cpu"):
    """Load baseline model (37 labels)"""
    print(f"📦 Loading baseline model from {BASELINE_MODEL_PATH}...")
    
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")
    
    model = TTNIRClassifier10_2(num_labels=37)
    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
    model.eval()
    model = model.to(device)
    print(f"✅ Baseline model loaded (37 labels)")
    return model

def load_specialist_model(device="cpu"):
    """Load specialist model (10 labels)"""
    print(f"📦 Loading specialist model from {SPECIALIST_MODEL_PATH}...")
    
    if not SPECIALIST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Specialist model not found: {SPECIALIST_MODEL_PATH}")
    
    model = TTNIRClassifier10_2(num_labels=10)
    model.load_state_dict(torch.load(SPECIALIST_MODEL_PATH, map_location=device))
    model.eval()
    model = model.to(device)
    print(f"✅ Specialist model loaded (10 labels)")
    return model

def ensemble_inference(baseline_model, specialist_model, X_test, batch_size=32, device="cpu"):
    """
    Run ensemble inference
    
    Args:
        baseline_model: TTN model with 37 outputs
        specialist_model: TTN model with 10 outputs
        X_test: Test features (N, 3, 1800)
        batch_size: Batch size for inference
        device: 'cpu' or 'cuda'
    
    Returns:
        y_ensemble: Ensemble predictions (N, 37)
    """
    n_samples = X_test.shape[0]
    y_ensemble = np.zeros((n_samples, 37))
    
    print(f"\n🚀 Running ensemble inference on {n_samples} samples...")
    print(f"   Baseline weight: {BASELINE_WEIGHT}, Specialist weight: {SPECIALIST_WEIGHT}")
    
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = torch.from_numpy(X_test[start_idx:end_idx]).float().to(device)
            
            # Baseline predictions (37 labels)
            y_baseline = baseline_model(X_batch).cpu().numpy()
            
            # Specialist predictions (10 labels)
            y_specialist = specialist_model(X_batch).cpu().numpy()
            
            # Blend predictions for rare groups
            batch_ensemble = y_baseline.copy()
            
            for specialist_idx, rare_idx in enumerate(SPECIALIST_INDICES):
                # Weighted average for rare groups
                batch_ensemble[:, rare_idx] = (
                    BASELINE_WEIGHT * y_baseline[:, rare_idx] +
                    SPECIALIST_WEIGHT * y_specialist[:, specialist_idx]
                )
            
            y_ensemble[start_idx:end_idx] = batch_ensemble
            
            # Progress
            pct = (batch_idx + 1) / num_batches * 100
            print(f"   [{pct:5.1f}%] Batch {batch_idx+1}/{num_batches}")
    
    print(f"✅ Ensemble inference complete")
    return y_ensemble

def evaluate_ensemble(y_pred, y_true, label_names=None):
    """Evaluate ensemble predictions"""
    from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
    
    print(f"\n📊 ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    # Threshold to 0/1
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Global metrics
    f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    
    print(f"F1-Score (Micro):   {f1_micro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    print(f"F1-Score (Weighted):{f1_weighted:.4f}")
    print(f"Hamming Loss:       {hamming_loss(y_true, y_pred_binary):.4f}")
    print(f"Exact Match:        {np.mean(np.all(y_true == y_pred_binary, axis=1)):.4f}")
    
    # Per-label on rare groups
    print(f"\n📍 RARE GROUPS (Specialist Focus):")
    print(f"{'-'*60}")
    
    specialist_f1s = []
    for specialist_idx, rare_idx in enumerate(SPECIALIST_INDICES):
        f1 = f1_score(y_true[:, rare_idx], y_pred_binary[:, rare_idx], zero_division=0)
        specialist_f1s.append(f1)
        
        if label_names:
            print(f"  {label_names[rare_idx]:25s} F1: {f1:.4f}")
        else:
            print(f"  Label {rare_idx:2d}: F1 {f1:.4f}")
    
    print(f"\nAverage F1 (Rare Groups): {np.mean(specialist_f1s):.4f}")
    print(f"{'='*60}")
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_rare_mean': np.mean(specialist_f1s),
        'exact_match': np.mean(np.all(y_true == y_pred_binary, axis=1))
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cpu"  # Use CPU (MPS doesn't support QR)
    
    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              ENSEMBLE INFERENCE: Baseline + Specialist                     ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    # Load models
    try:
        baseline_model = load_baseline_model(device)
        specialist_model = load_specialist_model(device)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Load test data
    print(f"\n📂 Loading test data...")
    try:
        X_test, y_test = load_processed_data(
            data_dir=REPO_DIR / "data",
            split='test',
            feature_type='lorentzian'
        )
        print(f"✅ Test data loaded: {X_test.shape} features, {y_test.shape} labels")
    except Exception as e:
        print(f"⚠️  Could not load test data: {e}")
        print(f"   Skipping evaluation")
        return None
    
    # Run ensemble inference
    y_pred = ensemble_inference(baseline_model, specialist_model, X_test, device=device)
    
    # Evaluate
    results = evaluate_ensemble(y_pred, y_test)
    
    # Save results
    results_file = REPO_DIR / "ensemble_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'baseline_weight': BASELINE_WEIGHT,
            'specialist_weight': SPECIALIST_WEIGHT,
            'specialist_indices': SPECIALIST_INDICES,
            'metrics': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n✨ ENSEMBLE COMPLETE ✨")
    print(f"   Target F1: 90-92%")
    print(f"   Actual F1 (Weighted): {results['f1_weighted']:.4f}")
    print(f"   Improvement: {results['f1_weighted'] - 0.862:.4f} (over 86.2% baseline)")
    
    return results

if __name__ == "__main__":
    main()
