"""
TTN 10.2 Optimierungs-Vorschläge basierend auf CNN Baseline Fehleranalyse
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("TTN 10.2 (LORENTZIAN, PERCENTILE) - OPTIMIERUNGS-STRATEGIE")
print("="*80)

# Lade die CNN Fehleranalyse
csv_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/reports/cnn_baseline_error_analysis.csv")
metrics_df = pd.read_csv(csv_path)

# Top problematische Gruppen
top_problems = metrics_df.nlargest(10, 'Error_Rate')

print("\n1. PROBLEMATISCHE GRUPPEN (aus CNN Baseline):")
print("-" * 80)

for idx, row in top_problems.iterrows():
    problem = "KRITISCH" if row['Error_Rate'] > 0.7 else "SCHLECHT" if row['Error_Rate'] > 0.5 else "MODERAT"
    print(f"[{int(row['Label_ID'])}] {row['FG_Name']:20s} | Error: {row['Error_Rate']:6.1%} | F1: {row['F1']:.3f} | Samples: {int(row['Positive_Samples']):6d} [{problem}]")

print("\n" + "="*80)
print("2. EMPFOHLENE ANPASSUNGEN FÜR TTN 10.2")
print("="*80)

print("\n🔴 PRIORITÄT 1 - CRITICAL CHANGES (muss sofort implementiert werden):")
print("-" * 80)

print("\nA. WEIGHTED LOSS - Für seltene Klassen")
print("""
Implementierung:
  • Berechne Class Weights basierend auf Häufigkeit
  • Inverse Häufigkeit: weight[i] = total_samples / (n_classes * count[i])
  • Weise höhere Gewichte seltenen Gruppen zu
  
Code-Beispiel:
  from sklearn.utils.class_weight import compute_class_weight
  
  # Für jedes Label
  for label_idx in range(n_labels):
    y_label = y_train[:, label_idx]
    weights = compute_class_weight('balanced', classes=[0,1], y=y_label)
    class_weights[label_idx] = weights
  
Effekt auf problematische Gruppen:
  • Thial (9 Samples):       weight ≈ 88x höher
  • Azo compound (115):       weight ≈ 7x höher
  • Hydrazone (1049):         weight ≈ 0.76x höher
""")

print("\nB. LONGER TRAINING - Überprüfe Early Stopping")
print("""
Status: TTN 10.2 wurde bei Epoch 78/200 gestoppt (Early Stopping)
Problem: Modell könnte noch trainiert werden
  
Lösung:
  • Erhöhe patience für Early Stopping von aktuellen setting auf 30-40 Epochs
  • Oder: Training auf 250-300 Epochs setzen
  • Monitore: val_f1 (nicht nur val_loss!)
  
Erwarteter Gewinn: +2-3% F1 Score möglich
""")

print("\nC. THRESHOLD OPTIMIZATION - Für seltene Klassen")
print("""
Aktuell: Fixed Threshold at 0.5 für alle Labels
Problem: Seltene Klassen haben andere Verteilungen
  
Lösung: Lerne pro-Label Threshold
  • Für jedes Label: finde optimalen Threshold auf Validation Set
  • Seltene Labels bekommen höhere Thresholds (weniger False Positives)
  • Häufige Labels bekommen niedrigere Thresholds (weniger False Negatives)
  
Code:
  for label_idx in range(n_labels):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.2, 0.8, 50):
      preds = (y_val_probs[:, label_idx] > thresh).astype(int)
      f1 = f1_score(y_val[:, label_idx], preds)
      if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
    optimal_thresholds[label_idx] = best_thresh
  
Effekt: Kann Recall bei seltenen Labels um 20-40% verbessern
""")

print("\n🟠 PRIORITÄT 2 - MEDIUM CHANGES (sehr empfohlen):")
print("-" * 80)

print("\nD. DATA AUGMENTATION für seltene Gruppen")
print("""
Methoden:
  1. Spektrale Perturbationen:
     - Kleine zufällige Verschiebungen im Wellenzahlenbereich
     - Rausch-Addition (SNR 50-100 dB)
     - Baseline-Schwankungen (+/- 5%)
  
  2. SMOTE (Synthetic Minority Over-sampling):
     - Generiert synthetische Samples für Thial, Azo compound
     - Erhöht Balance im Dataset
  
Fokus auf: Thial (9→50), Azo compound (115→300)
""")

print("\nE. FEATURE ENGINEERING - Spektralbereiche")
print("""
Bekannte Vibrationsbanden für problematische Gruppen:
  
  • Thial (C=S Stretch):        1050-1150 cm⁻¹
  • Azo (N=N Stretch):          1400-1600 cm⁻¹
  • Hydrazone (N-N Stretch):    900-1050 cm⁻¹
  • Phosphine (P-C Stretch):    800-1200 cm⁻¹
  • Sulfoxide (S=O Stretch):    1030-1070 cm⁻¹
  
Implementierung:
  • Erstelle Region-Masken für problematische Gruppen
  • Verstärke Features in diesen Regionen (z.B. mit Gaussian Weights)
  • Oder: Erstelle zusätzliche Kanäle mit fokussierten Spektren
""")

print("\nF. FOCAL LOSS - Für Hard Examples")
print("""
Standard BCE-Loss behandelt alle Fehler gleich.
Focal Loss konzentriert sich auf schwierige Samples.

Formula: FL = -α * (1-pt)^γ * log(pt)
  α = balancing parameter
  γ = focusing parameter (2-5)

Effekt: Modell konzentriert sich auf seltene/schwierige Gruppen
""")

print("\n🟢 PRIORITÄT 3 - NICE-TO-HAVE (optional):")
print("-" * 80)

print("\nG. ENSEMBLE mit CNN Baseline")
print("""
Idee: Kombiniere CNN und TTN Vorhersagen
  
Ensemble-Strategie:
  1. Average Ensemble:
     pred_ensemble = 0.5 * pred_cnn + 0.5 * pred_ttn
  
  2. Weighted Ensemble (basierend auf F1 Scores):
     weight_cnn = 0.6 (F1: 97.3%)
     weight_ttn = 0.4 (F1: 86.2%)
     pred_ensemble = 0.6 * pred_cnn + 0.4 * pred_ttn
  
  3. Label-spezifisches Ensemble:
     Für jedes Label: nutze Modell mit besserem F1 Score
     
Erwarteter F1 Score: 92-95% (besser als beide einzeln!)
""")

print("\nH. Hyperparameter Tuning")
print("""
Zu testen:
  • Learning Rate: [0.0001, 0.0005, 0.001, 0.005]
  • Batch Size: [64, 128, 256]
  • Dropout Rate: [0.1, 0.3, 0.5]
  • Weight Decay: [1e-5, 1e-4, 1e-3]
  
Tool: Optuna oder Ray Tune für automatische Suche
""")

print("\n" + "="*80)
print("3. IMPLEMENTIERUNGS-ROADMAP")
print("="*80)

roadmap = [
    ("Sofort (heute)", "Weighted Loss implementieren", "2h"),
    ("Tag 1", "Extended Training (250 Epochs) starten", "1d (parallel)"),
    ("Tag 2", "Optimal Threshold Learning implementieren", "3h"),
    ("Tag 3", "Data Augmentation für Thial + Azo", "4h"),
    ("Tag 4", "Feature Engineering (Spektralregionen)", "3h"),
    ("Tag 5", "Focal Loss + Hyperparameter Tuning", "2d"),
    ("Tag 6", "Ensemble mit CNN Baseline testen", "2h"),
]

print("\nGeschätzte Arbeitszeit pro Schritt:")
for phase, task, time_est in roadmap:
    print(f"  {phase:20s} | {task:50s} | {time_est:10s}")

print("\n" + "="*80)
print("4. ERWARTETE VERBESSERUNGEN")
print("="*80)

improvements = {
    'Baseline (aktuell)': 86.2,
    'Nach Weighted Loss': 88.5,
    'Nach Extended Training': 89.5,
    'Nach Threshold Optimization': 90.2,
    'Nach Data Augmentation': 91.0,
    'Nach Feature Engineering': 91.5,
    'Nach Focal Loss + Tuning': 92.0,
    'Ensemble mit CNN': 94.0,
}

print("\nPrognostizierte F1 Score Entwicklung:")
for step, f1 in improvements.items():
    improvement = f1 - 86.2
    bar = "█" * int(f1 / 5)
    print(f"  {step:30s}: {f1:5.1f}% {bar} (+{improvement:4.1f}%)")

print("\n" + "="*80)
print("5. SPEZIFISCHE ANPASSUNGEN PRO PROBLEM-GRUPPE")
print("="*80)

problem_groups = [
    {
        'name': 'Thial',
        'error': '100%',
        'samples': 9,
        'solution': 'Data Augmentation (9→50), Weight: 88x, Feature: 1050-1150 cm⁻¹'
    },
    {
        'name': 'Azo compound',
        'error': '98.3%',
        'samples': 115,
        'solution': 'Data Augmentation (115→300), Weight: 7x, Feature: 1400-1600 cm⁻¹'
    },
    {
        'name': 'Hydrazone',
        'error': '79%',
        'samples': 1049,
        'solution': 'Weighted Loss (2x), Focus Region: 900-1050 cm⁻¹'
    },
    {
        'name': 'Phosphine',
        'error': '78.6%',
        'samples': 117,
        'solution': 'Weighted Loss (3x), Data Aug, Focus: 800-1200 cm⁻¹'
    },
    {
        'name': 'Sulfoxide',
        'error': '78%',
        'samples': 930,
        'solution': 'Weighted Loss (1.5x), Focus: 1030-1070 cm⁻¹'
    },
]

for group in problem_groups:
    print(f"\n[{group['name']}] - Error: {group['error']}, Samples: {group['samples']}")
    print(f"  → {group['solution']}")

print("\n" + "="*80)
print("NÄCHSTER SCHRITT: Wählen Sie eine Strategie!")
print("="*80)
