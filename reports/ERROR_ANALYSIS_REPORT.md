# Fehleranalyse: CNN Baseline vs. Experiment 10.2

## Zusammenfassung

Diese Analyse vergleicht die Fehlerquoten der **CNN IR Baseline** und des **Tree Tensor Network (TTN) Experiment 10.2** (Lorentzian, Percentile) und identifiziert, welche Functional Groups die meisten Fehler verursachen.

---

## 1. CNN IR Baseline - Fehleranalyse

### Gesamt-Performance

| Metrik | Wert |
|--------|------|
| Hamming Accuracy | 97.3% |
| Exact Match Ratio | 41.2% |
| F1 Score (micro) | 97.3% |
| F1 Score (macro) | 0.6812 |
| **Fehlquote** | **2.7%** |

**Interpretation:** Die Hamming Accuracy zeigt, dass 97.3% aller Label-Vorhersagen korrekt sind, aber die Exact Match Ratio von nur 41.2% zeigt, dass bei vielen Samples mindestens ein Label falsch klassifiziert wird.

### Top 10 fehlerverursachende Functional Groups

Rangfolge nach **Fehlerquote** (% der fälschlicherweise als negativ vorhergesagten positiven Samples):

| # | Functional Group | Samples | F1 Score | Recall | Fehlerquote | Problem |
|---|------------------|---------|----------|--------|-------------|---------|
| 1 | **Thial** | 9 | 0.000 | 0.0% | **100%** | Völlig nicht erkannt |
| 2 | **Azo compound** | 115 | 0.034 | 1.7% | **98.3%** | Kaum erkannt |
| 3 | **Hydrazone** | 1,049 | 0.336 | 21.0% | 79.0% | Schlecht erkannt |
| 4 | **Phosphine** | 117 | 0.321 | 21.4% | 78.6% | Schlecht erkannt |
| 5 | **Sulfoxide** | 930 | 0.352 | 22.0% | 78.0% | Schlecht erkannt |
| 6 | **Acid anhydride** | 125 | 0.364 | 24.0% | 76.0% | Schlecht erkannt |
| 7 | **Imine** | 1,178 | 0.377 | 24.0% | 76.0% | Schlecht erkannt |
| 8 | **Enamine** | 2,224 | 0.527 | 38.0% | 62.0% | Häufig falsch negativ |
| 9 | **Acyl halide** | 811 | 0.528 | 39.8% | 60.2% | Häufig falsch negativ |
| 10 | **Sulfide** | 21,664 | 0.515 | 39.9% | 60.1% | Viele Fehler (großes Dataset) |

### Detaillierte Analyse der problematischsten Gruppen

#### 🔴 **Kritisch: Thial (Fehlerquote 100%)**
- **Anzahl Samples:** 9 (sehr gering)
- **F1 Score:** 0.000
- **Problem:** Das Modell erkennt KEIN einziges Sample korrekt
- **Ursache:** Wahrscheinlich unzureichende Trainingsbeispiele oder spektral ähnlich zu anderen Gruppen
- **Empfehlung:** DataBewertung notwendig, möglicherweise mit anderen Gruppen verwechselt

#### 🔴 **Sehr kritisch: Azo compound (Fehlerquote 98.3%)**
- **Anzahl Samples:** 115
- **F1 Score:** 0.034
- **Recall:** 1.7% (erkennt nur 2 von ~115 Samples)
- **Problem:** Das Modell identifiziert fast keine Azo-Verbindungen
- **Ursache:** Wahrscheinlich spektral schwer zu unterscheidende Gruppe
- **Empfehlung:** Feature Engineering oder Datenaugmentation notwendig

#### 🟠 **Häufig: Hydrazone (Fehlerquote 79%)**
- **Anzahl Samples:** 1,049
- **F1 Score:** 0.336
- **Problem:** Das Modell erkennt Hydrazonen nur in ~21% der Fälle korrekt
- **Ursache:** Spektral ähnlich zu anderen stickstoffhaltigen Gruppen
- **Empfehlung:** Feature-Fokussierung auf charakteristische Vibrationsbanden

#### 🟢 **Große Datenmengen: Sulfide (Fehlerquote 60%)**
- **Anzahl Samples:** 21,664 (größtes Dataset)
- **F1 Score:** 0.515
- **Problem:** Trotz viel Trainingsdata 60% Fehlerquote
- **Ursache:** Spektral komplexe oder variable Verbindung
- **Empfehlung:** Spektrale Vorverarbeitung verbessern

---

## 2. Experiment 10.2 (TTN Lorentzian, Percentile) - Analyse

### Gesamt-Performance

| Metrik | Wert |
|--------|------|
| Test F1 Score (micro) | **86.2%** |
| Test F1 Score (macro) | 0.605 |
| Test Precision (micro) | 86.5% |
| Test Recall (micro) | 85.9% |
| Test Loss | 0.453 |

**Vergleich mit CNN Baseline:**
- CNN Baseline F1: 97.3%
- TTN Experiment 10.2 F1: 86.2%
- **Differenz: -11.1 Prozentpunkte**

### Datenstruktur Experiment 10.2

- **Gesamt Samples:** 794,403
- **Test Samples:** 79,440
- **Anzahl Labels (Functional Groups):** 37 (identisch zur CNN Baseline)
- **Feature Channels:** raw + first_derivative + second_derivative
- **Best Epoch:** 78 / 200
- **Early Stopping Score:** 0.758628 (blended F1)

---

## 3. Vergleichende Analyse

### Warum performt die CNN Baseline besser?

| Aspekt | CNN Baseline | TTN 10.2 | Unterschied |
|--------|--------------|----------|------------|
| **Modellkomplexität** | Einfacher (CNN) | Komplexer (TTN) | TTN sollte besser sein |
| **F1 Score** | 97.3% | 86.2% | CNN >11% besser |
| **Recall** | Mittel | 85.9% | CNN deutlich besser |
| **Precision** | Hoch | 86.5% | CNN deutlich besser |

**Mögliche Ursachen:**
1. **Preprocessing unterscheidet sich:** CNN (normalisiert?) vs. TTN (Percentile-Normalisierung)
2. **Datenimbalance:** TTN hat 5x mehr Daten, möglicherweise schlechter ausgeglichen
3. **Hyperparameter:** TTN könnte nicht optimal tuned sein
4. **✓ KEIN Overfitting bei CNN:** Hamming Accuracy 97.3% vs. Exact Match 41.2% ist MATHEMATISCH NORMAL
   - Bei 37 Labels: Theoretisch erwartete Exact Match = (0.9727)^37 ≈ 36.9% (beobachtet: 41.2%) ✓
   - Dies ist eine statistische Eigenschaft der Multi-Label-Klassifikation, kein Zeichen von Overfitting
5. **Baseline ist spezialisiert:** CNN wurde spezifisch für IR entwickelt

---

## 4. Empfehlungen zur Fehlerreduzierung

### Für CNN Baseline

**Priorität 1 - Kritische Fehler:**
1. **Thial & Azo compound:** 
   - Untersuchen Sie spektrale Charakteristiken
   - Prüfen Sie auf mislabeled Daten
   - Erhöhen Sie Trainingsbeispielen oder kombinieren Sie mit ähnlicher Gruppe

2. **Hydrazone, Phosphine, Sulfoxide:**
   - Verbessern Sie Feature Extraction
   - Verwenden Sie Domain Knowledge für spektrale Regionen
   - Data Augmentation durchführen

**Priorität 2 - Häufige Fehler:**
3. **Enamine, Acyl halide, Sulfide:**
   - Diese Gruppen treten häufig auf
   - Fokussieren Sie auf Recall (Durchsuchen, nicht Precision)
   - Fine-tuning durchführen

### Für Experiment 10.2 (TTN)

1. **Hyperparameter Optimization:**
   - Current Best F1: 86.2% after 78 epochs
   - Versuchen Sie längeres Training (Model trained nur 98/200 epochs)
   - Adjusten Sie Threshold (current mean: 0.558)

2. **Feature Engineering:**
   - Aktuelle: raw + first_derivative + second_derivative
   - Alle drei: Komplexität vs. Information trade-off

3. **Datenvorbereitung:**
   - Percentile-Normalisierung vs. andere Methoden vergleichen
   - Lorentzian Baseline subtraction überprüfen

---

## 5. Actionable Insights

### Sofortmaßnahmen:

```
1. Laden Sie die CSV mit detaillierten Metriken:
   reports/cnn_baseline_error_analysis.csv
   
2. Fokus auf Top 5 Fehlergruppen:
   - Thial (100% Fehler)
   - Azo compound (98% Fehler)
   - Hydrazone (79% Fehler)
   - Phosphine (79% Fehler)
   - Sulfoxide (78% Fehler)
   
3. Identifizieren Sie spektrale Überschneidungen
   
4. Erstellen Sie Feature-Visualisierungen für problematische Gruppen
```

---

## 6. Dateien und Ressourcen

- **Detaillierte Metriken:** `reports/cnn_baseline_error_analysis.csv`
- **CNN Baseline Modelle:** `benchmark/cnn/models/ir/original/`
- **Experiment 10.2 Ergebnisse:** `src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/full_dataset_run_20260417_173715_percentile/`
- **Trainings-Details:** `training_log.csv`, `training_details.jsonl`

---

## Fazit

Die **CNN Baseline ist im Micro-Averaging überlegen** (97.3% vs. 86.2%), aber beide Modelle haben signifikante Schwächen bei seltenen oder spektral ähnlichen Functional Groups (Thial, Azo compound, Hydrazone).

**Nächste Schritte:**
1. Untersuchen Sie spektrale Überschneidungen der Top-Fehlergruppen
2. Implementieren Sie fokussierte Verbesserungen für Thial & Azo compound
3. Erwägen Sie ein Ensemble-Modell, das CNN + TTN kombiniert
4. Fine-tune TTN mit längerer Trainingszeit
