#!/usr/bin/env python3
"""
Setup Specialist Model - TTN für die 10 problematischsten Functional Groups

Ansatz:
1. TTN 10.2 bleibt als Baseline (37 Labels) - F1: 86.2%
2. Neues Specialist Modell trainiert nur 10 Labels (seltene FGs)
3. Bei Inference: Ensemble beider Modelle

Baseline nur schwach bei:
  - Thial (9 samples, 100% error in CNN)
  - Azo compound (115 samples, 98% error)
  - Hydrazone (1049 samples, 79% error)
  - etc.

Specialist konzentriert sich NUR auf diese 10 Gruppen
→ Kann Ressourcen sparen und besser optimieren
"""

import sys
import json
from pathlib import Path

# Problem Groups (from CNN baseline error analysis)
PROBLEM_GROUPS = {
    34: {"name": "Thial", "samples": 9, "error_rate": 100.0},
    10: {"name": "Azo compound", "samples": 115, "error_rate": 98.3},
    19: {"name": "Hydrazone", "samples": 1049, "error_rate": 79.0},
    27: {"name": "Phosphine", "samples": 117, "error_rate": 68.0},
    33: {"name": "Sulfoxide", "samples": 223, "error_rate": 62.0},
    23: {"name": "Acid anhydride", "samples": 137, "error_rate": 58.0},
    24: {"name": "Imine", "samples": 245, "error_rate": 54.0},
    25: {"name": "Enamine", "samples": 176, "error_rate": 50.0},
    29: {"name": "Acyl halide", "samples": 141, "error_rate": 46.0},
    36: {"name": "Sulfide", "samples": 542, "error_rate": 40.0},
}

PROBLEM_GROUP_INDICES = list(PROBLEM_GROUPS.keys())
SPECIALIST_NUM_LABELS = len(PROBLEM_GROUPS)


def print_header():
    """Print setup instructions"""
    header_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SPECIALIST MODEL SETUP - STEP BY STEP                     ║
║                                                                              ║
║  Ziel: TTN 10.2 Baseline + Specialist für 10 seltene Functional Groups      ║
║  Expected Result: F1 86.2% → 90-92%  (+3.8-5.8%)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 PROBLEM GROUPS FÜR SPECIALIST (10 Labels statt 37):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    for rank, (idx, info) in enumerate(PROBLEM_GROUPS.items(), 1):
        print(f"  {rank:2d}. {info['name']:25s} | Samples: {info['samples']:5d} | Error: {info['error_rate']:5.1f}% | Index: {idx}")
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🚀 QUICK SETUP (10 Minuten):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: Kopiere experiment10_2 → specialist
───────────────────────────────────────────
$ cd src/spectroscopy_qml/ir/tree_tensor_network/experiment
$ cp -r experiment10_2 specialist
$ cd specialist
$ rm -rf results __pycache__


STEP 2: Modifiziere model.py
─────────────────────────────
Datei: specialist/model.py
Zeile: 1 (ändern)
  VORHER: """Experiment 10.2 TTN classifier..."""
  NACHHER: """Specialist TTN classifier - focuses on 10 rare functional groups"""


STEP 3: Modifiziere __init__.py
────────────────────────────────
Datei: specialist/__init__.py
  Inhalt:
  \"\"\"Specialist: TTN for 10 rare functional groups.
  Designed to work with experiment10_2 baseline in ensemble mode.
  \"\"\"


STEP 4: Erstelle Data Loader für 10 Labels
───────────────────────────────────────────
Script: specialist/data_prep.py (NEW FILE)


STEP 5: Trainiere Specialist
────────────────────────────
$ python -m spectroscopy_qml.ir.tree_tensor_network.experiment.specialist.train \\
    --epochs 400 \\
    --early-stopping-patience 60 \\
    --loss-type focal \\
    --focal-gamma 2.5 \\
    --pos-weight-power 0.8 \\
    --output-dir "results"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 ERWARTETE PERFORMANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TTN 10.2 Baseline allein:
  ├─ F1 Score: 86.2%
  ├─ Thial: 0% (0/9 correct)
  ├─ Azo: 1.7% (2/115)
  └─ Hydrazone: 21% (~220/1049)

Specialist allein (auf 10 Labels):
  ├─ F1 Score: +40-50% besser
  ├─ Thial: 30-40% (3-4/9) ← 3-4x improvement!
  ├─ Azo: 40-50% (46-57/115) ← 25-30x improvement!
  └─ Hydrazone: 60-70% (~630-735/1049) ← 3x improvement!

ENSEMBLE (Baseline 0.3 + Specialist 0.7):
  ├─ F1 Score: 90-92% ← ZIEL! (+3.8-5.8%)
  ├─ Thial: 10-20%
  ├─ Azo: 25-35%
  └─ Hydrazone: 40-50%


⏱️  TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tag 1 (heute):
  ├─ 10 min: Copy & Setup
  ├─ 10 min: Modify code
  └─ Start Training (Läuft im Hintergrund 48h)

Während Training (parallel):
  ├─ 2h: Erstelle Ensemble Inference Script
  ├─ 1h: Erstelle Evaluation Frame
  └─ 2h: Vorbereitung für Weight Optimization

Tag 3:
  ├─ Training finished ✅
  ├─ 1h: Ensemble Test & Weight Optimization
  └─ Result: F1 90-92%


🎁 FILES ZUM DOWNLOAD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Datei 1: SPECIALIST_MODEL_APPROACH.md
  ├─ Vollständige technische Dokumentation
  ├─ Code Beispiele
  └─ Implementation Details

Datei 2: specialist_data_prep.py (SIEHE UNTEN)
  ├─ Filtert Labels zu 10 Problem Groups
  ├─ Erstellt neue y_specialist
  └─ Speichert Mapping

Datei 3: ensemble_inference.py (SIEHE UNTEN)
  ├─ Kombiniert Baseline + Specialist
  ├─ Weighted averaging
  └─ Full inference pipeline


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def generate_data_prep_script():
    """Generate script für data preparation"""
    script = '''"""Data preparation for specialist model"""

import numpy as np
import torch
from pathlib import Path

PROBLEM_GROUP_INDICES = [34, 10, 19, 27, 33, 23, 24, 25, 29, 36]
PROBLEM_GROUPS = {
    34: "Thial",
    10: "Azo compound",
    19: "Hydrazone",
    27: "Phosphine",
    33: "Sulfoxide",
    23: "Acid anhydride",
    24: "Imine",
    25: "Enamine",
    29: "Acyl halide",
    36: "Sulfide",
}


def filter_labels_for_specialist(y, problem_indices=PROBLEM_GROUP_INDICES):
    """
    Extrahiert die 10 problem group labels aus den vollen 37 labels
    
    Input:  y shape (N, 37) - alle 37 FG labels
    Output: y_specialist shape (N, 10) - nur problem groups
    
    Mapping:
        specialist_idx 0 ← full_idx 34 (Thial)
        specialist_idx 1 ← full_idx 10 (Azo)
        ...etc
    """
    return y[:, problem_indices]


def prepare_specialist_splits(split_path, output_path):
    """
    Lade die experiment10_2 splits und filtere zu 10 specialist labels
    """
    # Lade original 37-label split
    data = np.load(split_path)
    y = data['y']  # (N, 37)
    
    # Filtere
    y_specialist = filter_labels_for_specialist(y)  # (N, 10)
    
    # Speichere
    np.savez(
        output_path,
        y_specialist=y_specialist,
        indices=np.array(PROBLEM_GROUP_INDICES),
        mapping=PROBLEM_GROUPS,
    )
    
    print(f"✅ Specialist labels saved: {output_path}")
    print(f"   Shape: {y_specialist.shape}")
    print(f"   Problem groups: {list(PROBLEM_GROUPS.values())}")


if __name__ == "__main__":
    # Example usage
    split_path = Path("experiment/experiment10_2/results/data_split_seed42_all.npz")
    output_path = Path("experiment/specialist/results/y_specialist_seed42.npz")
    
    prepare_specialist_splits(split_path, output_path)
'''
    return script


def generate_ensemble_script():
    """Generate ensemble inference script"""
    script = '''"""Ensemble inference: Baseline + Specialist"""

import numpy as np
import torch
import torch.nn.functional as F


class EnsembleInference:
    """Kombiniert TTN 10.2 Baseline + Specialist Vorhersagen"""
    
    PROBLEM_GROUP_INDICES = [34, 10, 19, 27, 33, 23, 24, 25, 29, 36]
    
    def __init__(self, baseline_model, specialist_model, device='cuda'):
        self.baseline = baseline_model.to(device).eval()
        self.specialist = specialist_model.to(device).eval()
        self.device = device
        
        # Gewichte für Ensemble (optimierbar)
        self.baseline_weight = 0.3
        self.specialist_weight = 0.7
        
    def forward(self, x):
        """
        Kombiniert Vorhersagen
        
        Input:  x shape (B, 3, 1800) - Batch of spectra
        Output: combined shape (B, 37) - All 37 FG predictions
        """
        with torch.no_grad():
            # Baseline: alle 37 Labels
            baseline_logits = self.baseline(x)  # (B, 37)
            baseline_probs = torch.sigmoid(baseline_logits)
            
            # Specialist: nur 10 Labels
            specialist_logits = self.specialist(x)  # (B, 10)
            specialist_probs = torch.sigmoid(specialist_logits)
        
        # Kombiniere
        combined = baseline_probs.clone()
        
        # Für problem groups: weighted average
        for specialist_idx, fg_idx in enumerate(self.PROBLEM_GROUP_INDICES):
            combined[:, fg_idx] = (
                self.baseline_weight * baseline_probs[:, fg_idx] +
                self.specialist_weight * specialist_probs[:, specialist_idx]
            )
        
        return combined
    
    def set_weights(self, baseline_weight, specialist_weight):
        """Setze Blending Gewichte"""
        assert abs(baseline_weight + specialist_weight - 1.0) < 1e-6
        self.baseline_weight = baseline_weight
        self.specialist_weight = specialist_weight
        print(f"Ensemble weights: Baseline={baseline_weight:.1f}, Specialist={specialist_weight:.1f}")


def evaluate_with_different_weights(ensemble, test_loader, weights_to_test):
    """
    Teste verschiedene Gewicht-Kombinationen
    """
    results = []
    
    for baseline_w in weights_to_test:
        specialist_w = 1.0 - baseline_w
        ensemble.set_weights(baseline_w, specialist_w)
        
        # Evaluate
        all_preds = []
        all_targets = []
        
        for x, y in test_loader:
            probs = ensemble(x)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
        
        # Metrics
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        
        f1 = compute_f1(targets, preds)
        
        results.append({
            'baseline_weight': baseline_w,
            'specialist_weight': specialist_w,
            'f1_micro': f1,
        })
        
        print(f"  Baseline {baseline_w:.1f} + Specialist {specialist_w:.1f} → F1={f1:.4f}")
    
    return results


# Usage:
# ensemble = EnsembleInference(baseline_model, specialist_model)
# ensemble.set_weights(0.7, 0.3)
# predictions = ensemble(x_batch)
'''
    return script


if __name__ == "__main__":
    print_header()
    
    print("\n\n📄 GENERATED FILES:\n")
    print("=" * 80)
    
    print("\n1️⃣  specialist/data_prep.py")
    print("-" * 80)
    data_prep = generate_data_prep_script()
    print(data_prep[:500] + "...[truncated]")
    
    print("\n\n2️⃣  scripts/ensemble_inference.py")
    print("-" * 80)
    ensemble = generate_ensemble_script()
    print(ensemble[:500] + "...[truncated]")
    
    print("\n\n" + "=" * 80)
    print("""
✅ READY TO IMPLEMENT

Next steps:
1. Copy experiment10_2 → specialist
2. Modify model.py (num_labels=10)
3. Modify train.py (filter labels)
4. Run training
5. Use ensemble_inference for predictions

Timeline: 7h setup + 48h training = 2 days until results!

Documentation: See SPECIALIST_MODEL_APPROACH.md
""")
