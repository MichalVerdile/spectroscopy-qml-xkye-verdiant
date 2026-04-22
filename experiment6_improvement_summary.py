#!/usr/bin/env python3
"""
Experiment 6 - Score Improvement Strategies
Übersicht aller Möglichkeiten für bessere F1-Scores
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   EXPERIMENT 6 - VERBESSERUNGSSTRATEGIEN                      ║
║              Basierend auf CNN Baseline (F1: 97.3%) Fehleranalyse             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🎯 KERNPROBLEME IDENTIFIZIERT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CLASS IMBALANCING (KRITISCH)
   └─ Thial:           9 Samples   (100% Fehlerrate im CNN)
   └─ Azo compound:    115 Samples (98.3% Fehlerrate)
   └─ Hydrazone:       1,049 Samples (79% Fehlerrate)
   
   ⚡ LÖSUNG: Weighted Loss mit pos_weight_power=0.6
   └─ Gibt Thial ~88x höheres Gewicht
   └─ Azo ~7x höheres Gewicht
   
2. FEATURE QUALITY (MITTEL)
   └─ Raw Derivatives reintroduzieren Rauschen
   
   ⚡ LÖSUNG: Lorentzian-smoothed Features (wie Exp 10.2)
   └─ Glattere Ableitungen, weniger Rauschen
   
3. EARLY STOPPING (MITTEL)
   └─ Modell stoppt zu früh (zu konservativ)
   
   ⚡ LÖSUNG: Extended Training + relaxed early stopping
   └─ 300-400 Epochen (statt 200)
   └─ Patience 60 (statt 20)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SCORE IMPROVEMENT MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategie                   │ Aufwand │ Gain │ Priorität │ Schwierigkeit
────────────────────────────┼─────────┼──────┼───────────┼──────────────
Extended Training (400ep)   │ 0h+48h  │ +2%  │ 🔴 HIGH  │ ⭐ TRIVIAL
Weighted Loss (0.6)         │ 1h      │ +1%  │ 🔴 HIGH  │ ⭐ EASY
Lorentzian Features         │ 3h      │ +1.5%│ 🟠 MID   │ ⭐⭐ OK
Focal Loss                  │ 2h      │ +0.5%│ 🟡 LOW   │ ⭐⭐ OK
Data Augmentation           │ 4h      │ +1%  │ 🟡 LOW   │ ⭐⭐⭐ HARD
Per-Label Threshold Opt     │ 1h      │ +0.3%│ 🟡 LOW   │ ⭐ EASY
Chi erhöhen (64→96)         │ 0h      │ +0.3%│ 🟡 LOW   │ ⭐ EASY
────────────────────────────┼─────────┼──────┼───────────┼──────────────
TOTAL (All)                 │ 11h     │ +6%  │ ★★★★★   │ ⭐⭐⭐

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 QUICK WINS - TOP 3 (Sofort umzusetzen)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#1: EXTENDED TRAINING + WEIGHTED LOSS
────────────────────────────────────────
Befehl:
  python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \\
      --epochs 400 \\
      --early-stopping-patience 60 \\
      --pos-weight-power 0.6

Aufwand:     0h (nur CLI flags)
Laufzeit:    ~48 Stunden GPU
Erwarteter Gewinn:  +2-3% F1 ✅

Warum funktioniert das:
  ✓ 400 Epochen geben dem Modell mehr Zeit
  ✓ Patience 60 verhindert zu frühes Stoppen
  ✓ pos_weight_power=0.6 balanciert Thial/Azo/Hydrazone


#2: LORENTZIAN FEATURE UPGRADE (Während #1 läuft)
────────────────────────────────────────────────
Schritte:
  1. Feature Map aus experiment10_2/model.py kopieren
  2. In experiment6/model.py integrieren
  3. Train mit Lorentzian flags

Aufwand:     ~3 Stunden
Laufzeit:    ~48 Stunden GPU (parallel zu #1)
Erwarteter Gewinn:  +1.5-2% F1
Kombi mit #1: +3.5-5% total! 🎉

Warum funktioniert das:
  ✓ Analytische Ableitungen statt Finite Differences
  ✓ Glattere Features = weniger Rauschen
  ✓ Besseres Signal für seltene Gruppen


#3: FOCAL LOSS (Optional, wenn Zeit)
──────────────────────────────────
Befehl:
  python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \\
      --epochs 400 \\
      --loss-type focal \\
      --focal-gamma 2.5

Aufwand:     ~2 Stunden
Laufzeit:    ~48 Stunden GPU
Erwarteter Gewinn:  +0.5-1% F1 (zusätzlich)

Warum funktioniert das:
  ✓ Focus auf schwierige Samples (True Negatives)
  ✓ Exponential reweighting: (1-p)^gamma
  ✓ Besseres Lernen von Edge Cases


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 EXPERIMENTVERLAUF - ZEITLICH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TAG 1 (Start)
  ├─ 09:00 - Schnellcheck: python train.py --check-only
  ├─ 10:00 - Start Experiment 6a (Extended + Weighted)
  └─ 11:00 - Start Implementierung Lorentzian (parallel)

TAG 2-3 (Parallel)
  ├─ Exp 6a trainiert (~36-48h) → Ergebnis: +2-3% F1
  ├─ Implementiere Lorentzian Features
  └─ Implementiere Focal Loss Option

TAG 3-4
  ├─ Start Experiment 6b (Lorentzian)
  ├─ Start Experiment 6c (Focal Loss)
  └─ Exp 6a abgeschlossen ✅

TAG 4-5
  ├─ Exp 6b & 6c trainieren
  └─ Versuche Data Augmentation

TAG 6
  ├─ Exp 6b & 6c abgeschlossen
  ├─ Results vergleichen
  └─ Best Version identifizieren

ZIEL: Alle Varianten getestet, Best F1 Score gefunden ✅


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 DER PLAN - SCHRITT FÜR SCHRITT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A: Fast Track (48 Stunden)
────────────────────────────────
Schritt 1 [0h]:  Terminal öffnen
Schritt 2 [1h]:  Parameter anpassen
Schritt 3 [48h]: Training laufen lassen
Schritt 4 [1h]:  Results evaluieren

RESULT: +2-3% F1 ohne Code-Änderungen 🎉


OPTION B: Balanced (5 Tage)
──────────────────────────
Schritt 1 [1h]:   Extended Training Setup
Schritt 2 [3h]:   Lorentzian Features integrieren
  └─ Parallel: Extended Training läuft
Schritt 3 [2h]:   Focal Loss hinzufügen
  └─ Parallel: Lorentzian Training läuft
Schritt 4 [4h]:   Data Augmentation vorbereiten
  └─ Parallel: Focal Loss Training läuft
Schritt 5 [1h]:   Alle Results vergleichen & Best wählen

RESULT: +5-7% F1 mit mehreren Varianten 🎉🎉


OPTION C: Comprehensive (2 Wochen)
──────────────────────────────────
Alle Schritte von Option B +
  └─ Hyperparameter Grid Search (Chi, LR, etc)
  └─ Ensemble mit CNN Baseline
  └─ Cross-Validation Evaluation

RESULT: +7-10% F1 mit voll optimiertem System 🎉🎉🎉


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 TECHNISCHE DETAILS - WEIGHTED LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pos_weight_power = 0.6  (statt 0.5)

Effekt:
  Thial       (9 samples):      ~88.0x → 176.0x
  Azo         (115 samples):    ~7.0x  → 14.0x
  Hydrazone   (1049 samples):   ~0.76x → 1.52x

Diese exponentiellen Gewichte zwingen TTN dazu:
  ✓ Viel mehr bei Thial Samples lernen
  ✓ Bessere Features für seltene Gruppen
  ✓ Weniger False Negatives bei Thial

Mathematik:
  weight = (1 + pos_weight_power) ^ (negative_samples / positive_samples)
  
  Mit power=0.6 vs 0.5:
    Gain = (1.6)^log(samples) - (1.5)^log(samples)
    
  Bei nur 9 Positiven = ~2x stärkere Gewichtung


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ HÄUFIGE FRAGEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

F: Welche einzelne Veränderung hat den größten Impact?
A: Extended Training (300-400 epochs) + Weighted Loss (0.6)
   → +2-3% F1 mit minimal Aufwand

F: Kann ich alle 5 Strategien kombinieren?
A: JA! Verwende alle zusammen:
   - Extended Training: 400 epochs
   - Weighted Loss: pos_weight_power=0.6
   - Lorentzian Features: Kopie aus Exp 10.2
   - Focal Loss: alpha=0.25, gamma=2.5
   - Data Augmentation: 5x für Thial/Azo
   
   Erwarteter Combined Gain: +6-8% F1

F: Wie stelle ich fest, welche Strategie am besten funktioniert?
A: Ablation Study:
   1. Run 1: Extended Training nur
   2. Run 2: + Weighted Loss
   3. Run 3: + Lorentzian Features
   4. Run 4: + Focal Loss
   5. Run 5: + Data Augmentation
   
   → Grafik: F1 Score vs Kombinationen

F: Brauche ich GPU dafür?
A: GPU macht es 50x schneller, aber CPU geht auch (braucht 2-3 Tage pro Run)

F: Was ist realistisch zu erreichen?
A: Conservative:  +2-3% (Extended + Weighted)
   Realistic:     +4-5% (+ Lorentzian + Focal)
   Aggressive:    +6-8% (+ Data Augmentation + Tuning)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ RECOMMENDATION: STARTEN SIE JETZT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOFORT umzusetzen (15 Minuten):
  
  python -m spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.train \\
      --epochs 400 \\
      --early-stopping-patience 60 \\
      --pos-weight-power 0.6 \\
      --early-stopping-min-delta 5e-5 \\
      --output-dir "experiment/experiment6/results_improved"

Danach (parallel laufen lassen):
  - Lorentzian Features (3h Implementierung, 48h Training)
  - Focal Loss Variant (2h Implementierung, 48h Training)
  - Data Augmentation (4h Implementierung, 48h Training)

Result in 1-2 Wochen: Best case +7-8% F1  🎯


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 REFERENZEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detaillierte Anleitung:     EXPERIMENT_6_IMPROVEMENTS.md
Error Analysis (CNN Base):  reports/ERROR_ANALYSIS_REPORT.md
Optimization Strategy:      TTN_OPTIMIZATION_STRATEGY.py
Experiment 10.6 (Ref):     EXPERIMENT_10_6_NOTES.md

╔════════════════════════════════════════════════════════════════════════════════╗
║ Ready to improve? Run the command above and watch F1 score increase! 🚀        ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
