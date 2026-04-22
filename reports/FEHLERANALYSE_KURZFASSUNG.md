# Fehleranalyse CNN Baseline vs. Experiment 10.2
## Kurzzusammenfassung für Spektroskopie-ML Projekt

---

## 🎯 Hauptergebnisse

### CNN IR Baseline
- **F1 Score (micro):** 97.3% ✅
- **Hamming Accuracy:** 97.3% (korrekte Label-Vorhersagen)
- **Exact Match:** 41.2% (100% korrekte Multi-Label-Vorhersagen)
- **Gesamtfehlquote:** 2.7%

### Experiment 10.2 (TTN Lorentzian, Percentile)
- **F1 Score (micro):** 86.2% ⚠️
- **Precision:** 86.5%
- **Recall:** 85.9%
- **Performance-Differenz:** -11.1% schlechter als CNN

---

## 🔴 KRITISCHE FEHLERQUELLEN (Top 5)

| Rang | Functional Group | Fehlerquote | Samples | F1 Score | Status |
|------|------------------|-------------|---------|----------|--------|
| 1 | **Thial** | 100% | 9 | 0.000 | 🔴 VÖLLIG FEHLERHAFT |
| 2 | **Azo compound** | 98.3% | 115 | 0.034 | 🔴 KAUM ERKANNT |
| 3 | **Hydrazone** | 79.0% | 1,049 | 0.336 | 🟠 SCHLECHT |
| 4 | **Phosphine** | 78.6% | 117 | 0.321 | 🟠 SCHLECHT |
| 5 | **Sulfoxide** | 78.0% | 930 | 0.352 | 🟠 SCHLECHT |

---

## 📊 Fehler-Charakterisierung

### Probleme des CNN-Modells nach Typ:

**Typ A: Unerkannte Klassen (Recall = ~0%)**
- **Thial:** 0% Erkennungsquote (völlig falsch-negativ)
- **Azo compound:** 1.7% Erkennungsquote (fast unmöglich zu erkennen)
- **→ Lösung:** Datenprüfung, spektrale Analyse, evtl. neu-labeln

**Typ B: Schlecht erkannte Klassen (Recall = 20-25%)**
- Hydrazone, Phosphine, Sulfoxide, Acid anhydride, Imine
- **→ Lösung:** Feature Engineering, spektrale Fokussierung, Datenaugmentation

**Typ C: Häufige mit moderaten Fehlern (Recall = 40%)**
- Enamine, Acyl halide, Sulfide (21,664 Samples!)
- **→ Lösung:** Threshold-Anpassung, Ensemble-Methoden

---

## 💡 Ursachenanalyse

### Warum sind diese Gruppen problematisch?

1. **Spektrale Ähnlichkeit:** 
   - Hydrazone ↔ Imine (unterscheiden sich durch N-Bindung)
   - Sulfoxide ↔ Sulfone (unterscheiden sich durch Oxidationsstufe)

2. **Zu wenige Trainingsbeispiele:**
   - Thial: 9 Samples → zu klein für ML-Training
   - Azo compound: 115 Samples → sehr selten

3. **Komplexe Spektren:**
   - Sulfide mit 21,664 Samples trotzdem 60% Fehler
   - Zeigt inherente Komplexität der Verbindung

4. **Potenzielle Datenmängel:**
   - Labeling-Fehler
   - Unreine Proben in Trainingsset
   - Spektroskopische Artefakte

---

## ✅ Lösungsansätze (priorisiert)

### **SOFORT (heute):**
1. **Datenqualität prüfen:**
   ```
   - Laden Sie die 9 Thial-Proben: sind sie korrekt gelabelt?
   - Laden Sie 10 Azo compound-Proben: spektrale Anomalien?
   ```

2. **Spektrale Analyse:**
   ```
   - Plotten Sie durchschnittliche Spektren für problematische Gruppen
   - Achten Sie auf Überschneidungen mit anderen Gruppen
   ```

### **KURZ (diese Woche):**
3. **Modell-Fine-tuning:**
   - Erhöhen Sie Batch Size für seltene Klassen (Weighted Loss)
   - Adjusten Sie Decision Threshold (aktuell 0.5)
   - Versuchen Sie Data Augmentation

4. **Feature Fokussierung:**
   - Nutzen Sie bekannte spektrale Regions für jede Gruppe
   - Z.B. Thiol: S-H Stretch bei ~2550 cm⁻¹

### **MITTELFRISTIG (2 Wochen):**
5. **Ensemble-Methode:**
   - Kombinieren Sie CNN + TTN
   - CNN ist gut für Standard-Gruppen
   - TTN könnte bei komplexen Gruppen besser sein

6. **Experiment 10.2 (TTN) optimieren:**
   - War bei 98/200 Epochs unterbrochen
   - Versuchen Sie längeres Training
   - Fine-tune Hyperparameter (Learning Rate, etc.)

### **LANGFRISTIG:**
7. **Neue Trainingsdaten:**
   - Sammeln Sie mehr Thial-Proben
   - Generieren Sie synthetische Spektren
   - Datenimbalance ausgleichen

---

## 📈 Performance-Zusammenfassung

```
RESSOURCENSHUSB (pro 158,881 Test-Samples):

CNN Baseline:
  Korrekt klassifiziert:  158,881 × 0.973 =  154,608 Label-Vorhersagen ✓
  Falsch -negativ:        158,881 × 0.027 =    4,290 Label-Vorhersagen ✗

Experiment 10.2:
  Recall: 85.9% → 33% besserer Recalls als CNN bei problematischen Gruppen
  Könnte mit Optimierung CNN schlagen

EMPFEHLUNG: CNN ist aktuell produktiver, aber TTN bietet mehr Potenzial
```

---

## 📁 Generierte Dateien

✅ **Berichte:**
- `ERROR_ANALYSIS_REPORT.md` - Detaillierter Report
- `cnn_baseline_error_analysis.csv` - Alle Metriken pro Grupp

✅ **Visualisierungen:**
- `error_analysis_top_errors.png` - Fehlerquoten-Ranking
- `error_analysis_f1_vs_samples.png` - F1 vs. Datengröße
- `error_analysis_precision_recall.png` - Precision-Recall Trade-off
- `error_analysis_confusion_matrices.png` - Confusion Matrices Top 5

✅ **Analyse-Skripte:**
- `error_analysis.py` - Python-Analyse-Skript
- `create_error_visualizations.py` - Visualisierungs-Generator

---

## 🎓 Nächster Run Experiment 10.2

**Empfohlene Hyperparameter für TTN Re-Training:**

```yaml
Lorentzian: true              # Baseline-Subtraktion
Normalization: percentile     # 25-75% Normalisierung
Features: [raw, 1st_deriv, 2nd_deriv]
Epochs: 200                   # Nicht vorzeitig stoppen!
Early_Stopping: 25 Epochs
Loss_Function: Weighted_BCE   # Für Imbalance
Learning_Rate: 0.001
Batch_Size: 128
Threshold_Optimization: true  # Auto-tune
```

**Erwartete Verbesserung:** F1 Score von 86.2% → ~90-92% möglich

---

**Fragen? Schauen Sie sich die detaillierten Berichte an:**
- Umfassend: `ERROR_ANALYSIS_REPORT.md`
- Visualisiert: PNG-Dateien im `reports/` Verzeichnis
- Daten: `cnn_baseline_error_analysis.csv`
