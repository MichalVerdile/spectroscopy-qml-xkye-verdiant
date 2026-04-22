"""
Fehleranalyse für CNN Baseline und Experiment 10.2
Analysiert, welche Functional Groups zu Fehlern führen
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)
from rdkit import Chem
import sys

try:
    import torch
except ImportError:
    torch = None

# ============================================================================
# CNN Baseline Analyse
# ============================================================================

def get_functional_groups_names():
    """Gebe eine geordnete Liste der Functional Group Namen in der gleichen Reihenfolge wie im CNN Script"""
    return [
        "Acid anhydride",      # 0
        "Acyl halide",         # 1
        "Alcohol",             # 2
        "Aldehyde",            # 3
        "Alkane",              # 4
        "Alkene",              # 5
        "Alkyne",              # 6
        "Amide",               # 7
        "Amine",               # 8
        "Arene",               # 9
        "Azo compound",        # 10
        "Carbamate",           # 11
        "Carboxylic acid",     # 12
        "Enamine",             # 13
        "Enol",                # 14
        "Ester",               # 15
        "Ether",               # 16
        "Haloalkane",          # 17
        "Hydrazine",           # 18
        "Hydrazone",           # 19
        "Imide",               # 20
        "Imine",               # 21
        "Isocyanate",          # 22
        "Isothiocyanate",      # 23
        "Ketone",              # 24
        "Nitrile",             # 25
        "Phenol",              # 26
        "Phosphine",           # 27
        "Sulfide",             # 28
        "Sulfonamide",         # 29
        "Sulfonate",           # 30
        "Sulfone",             # 31
        "Sulfonic acid",       # 32
        "Sulfoxide",           # 33
        "Thial",               # 34
        "Thioamide",           # 35
        "Thiol",               # 36
    ]

def get_functional_groups_for_smiles(smiles: str) -> dict:
    """Bestimme Functional Groups für ein SMILES String"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    functional_groups_dict = {
        "Acid anhydride": Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])"),
        "Acyl halide": Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"),
        "Alcohol": Chem.MolFromSmarts("[#6][OX2H]"),
        "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
        "Alkane": Chem.MolFromSmarts("[CX4;H3,H2]"),
        "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
        "Alkyne": Chem.MolFromSmarts("[CX2]#[CX2]"),
        "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
        "Amine": Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"),
        "Arene": Chem.MolFromSmarts("[cX3]1[cX3][cX3][cX3][cX3][cX3]1"),
        "Azo compound": Chem.MolFromSmarts("[#6][NX2]=[NX2][#6]"),
        "Carbamate": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2H0]"),
        "Carboxylic acid": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
        "Enamine": Chem.MolFromSmarts("[NX3][CX3]=[CX3]"),
        "Enol": Chem.MolFromSmarts("[OX2H][#6X3]=[#6]"),
        "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
        "Ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
        "Haloalkane": Chem.MolFromSmarts("[#6][F,Cl,Br,I]"),
        "Hydrazine": Chem.MolFromSmarts("[NX3][NX3]"),
        "Hydrazone": Chem.MolFromSmarts("[NX3][NX2]=[#6]"),
        "Imide": Chem.MolFromSmarts("[CX3](=[OX1])[NX3][CX3](=[OX1])"),
        "Imine": Chem.MolFromSmarts("[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"),
        "Isocyanate": Chem.MolFromSmarts("[NX2]=[C]=[O]"),
        "Isothiocyanate": Chem.MolFromSmarts("[NX2]=[C]=[S]"),
        "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
        "Nitrile": Chem.MolFromSmarts("[NX1]#[CX2]"),
        "Phenol": Chem.MolFromSmarts("[OX2H][cX3]:[c]"),
        "Phosphine": Chem.MolFromSmarts("[PX3]"),
        "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
        "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
        "Sulfonate": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]"),
        "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
        "Sulfonic acid": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H]"),
        "Sulfoxide": Chem.MolFromSmarts("[#16X3]=[OX1]"),
        "Thial": Chem.MolFromSmarts("[CX3H1](=S)[#6,H]"),
        "Thioamide": Chem.MolFromSmarts("[NX3][CX3]=[SX1]"),
        "Thiol": Chem.MolFromSmarts("[#16X2H]"),
    }
    
    result = {}
    for fg_name, smarts in functional_groups_dict.items():
        if smarts is not None:
            matches = mol.HasSubstructMatch(smarts)
            result[fg_name] = matches
    
    return result

def analyze_cnn_baseline():
    """Analysiere CNN IR Baseline Fehler (Multi-Label Klassifikation)"""
    
    print("\n" + "="*80)
    print("CNN IR BASELINE FEHLERANALYSE")
    print("="*80)
    
    # Lade die CNN Baseline Ergebnisse
    pickle_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/benchmark/cnn/models/ir/original/results.pickle")
    
    if not pickle_path.exists():
        print(f"Fehler: {pickle_path} nicht gefunden!")
        return
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    
    # Extrahiere Vorhersagen und Labels (Multi-Label Format)
    y_true = results['tgt']  # Shape: (n_samples, n_labels)
    y_pred = results['pred']  # Shape: (n_samples, n_labels)
    
    # Binarisiere Vorhersagen (use threshold 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print(f"\nAnzahl Test-Samples: {y_true.shape[0]}")
    print(f"Anzahl Labels (Functional Groups): {y_true.shape[1]}")
    print(f"Shape y_true: {y_true.shape}, Shape y_pred: {y_pred.shape}")
    
    # Hole Functional Group Namen
    fg_names = get_functional_groups_names()
    
    # Berechne Metriken pro Label/Klasse
    print("\n" + "-"*80)
    print("METRIKEN PRO LABEL (Functional Group)")
    print("-"*80)
    
    class_metrics = []
    
    for label_id in range(y_true.shape[1]):
        y_true_label = y_true[:, label_id]
        y_pred_label = y_pred_binary[:, label_id]
        
        # Zähle echte positive Samples
        n_positive = np.sum(y_true_label == 1)
        
        if n_positive == 0:
            continue  # Überspringe Labels ohne positive Samples
        
        # Berechne Metriken
        tp = np.sum((y_true_label == 1) & (y_pred_label == 1))
        fn = np.sum((y_true_label == 1) & (y_pred_label == 0))
        fp = np.sum((y_true_label == 0) & (y_pred_label == 1))
        tn = np.sum((y_true_label == 0) & (y_pred_label == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        error_rate = fn / n_positive if n_positive > 0 else 0
        
        fg_name = fg_names[label_id] if label_id < len(fg_names) else f"Label_{label_id}"
        
        class_metrics.append({
            'Label_ID': label_id,
            'FG_Name': fg_name,
            'Positive_Samples': int(n_positive),
            'TP': int(tp),
            'FN': int(fn),
            'FP': int(fp),
            'TN': int(tn),
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Error_Rate': error_rate
        })
    
    metriken_df = pd.DataFrame(class_metrics)
    
    # Sortiere nach Error Rate (absteigend)
    metriken_df_sorted = metriken_df.sort_values('Error_Rate', ascending=False)
    
    print("\nTop 10 Functional Groups mit höchsten Fehlerquoten:")
    top_errors = metriken_df_sorted[['Label_ID', 'FG_Name', 'Positive_Samples', 'F1', 'Precision', 'Recall', 'Error_Rate']].head(10)
    for idx, row in top_errors.iterrows():
        print(f"  [{row['Label_ID']}] {row['FG_Name']:25s} | Samples: {row['Positive_Samples']:6d} | F1: {row['F1']:.3f} | Recall: {row['Recall']:.3f} | Error: {row['Error_Rate']:.1%}")
    
    print("\nTop 10 Functional Groups mit niedrigsten F1 Scores:")
    metriken_df_f1 = metriken_df.sort_values('F1', ascending=True)
    top_f1 = metriken_df_f1[['Label_ID', 'FG_Name', 'Positive_Samples', 'F1', 'Precision', 'Recall']].head(10)
    for idx, row in top_f1.iterrows():
        print(f"  [{row['Label_ID']}] {row['FG_Name']:25s} | Samples: {row['Positive_Samples']:6d} | F1: {row['F1']:.3f} | Prec: {row['Precision']:.3f} | Rec: {row['Recall']:.3f}")
    
    # Speichere detaillierte Metriken
    output_file = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/reports/cnn_baseline_error_analysis.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metriken_df_sorted.to_csv(output_file, index=False)
    print(f"\nDetaillierte Metriken gespeichert: {output_file}")
    
    # Berechne Gesamt-Metriken (Multi-Label Metriken)
    print("\n" + "-"*80)
    print("GESAMT-METRIKEN CNN IR BASELINE")
    print("-"*80)
    
    # Hamming Accuracy (wie in der summary.txt - Anteil korrekt klassifizierter Labels über alle Samples)
    hamming_accuracy = np.mean(y_true == y_pred_binary)
    
    # Sample-weise Genauigkeit (exakte Übereinstimmung)
    exact_match = np.mean(np.all(y_true == y_pred_binary, axis=1))
    
    # Micro-averaged F1
    f1_micro = f1_score(y_true.flatten(), y_pred_binary.flatten(), average='micro')
    
    # Macro-averaged F1
    f1_macro = f1_score(y_true, y_pred_binary, average='macro')
    
    print(f"Hamming Accuracy:    {hamming_accuracy:.4f} ({hamming_accuracy*100:.1f}%)")
    print(f"Exact Match Ratio:   {exact_match:.4f} ({exact_match*100:.1f}%)")
    print(f"F1 Score (micro):    {f1_micro:.4f} ({f1_micro*100:.1f}%)")
    print(f"F1 Score (macro):    {f1_macro:.4f}")
    print(f"\nFehlquote (1 - Hamming Acc): {(1-hamming_accuracy)*100:.1f}%")
    
    return metriken_df_sorted


def analyze_experiment_10_2():
    """Analysiere Experiment 10.2 Fehler"""
    from pathlib import Path
    print("\n\n" + "="*80)
    print("EXPERIMENT 10.2 (LORENTZIAN, PERCENTILE) FEHLERANALYSE")
    print("="*80)
    
    # Finde den latest percentile run
    results_dir = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results")
    
    # Finde alle percentile Runs
    percentile_runs = sorted([d for d in results_dir.iterdir() if d.is_dir() and 'percentile' in d.name])
    
    if not percentile_runs:
        print("Keine Experiment 10.2 percentile Runs gefunden!")
        return
    
    # Finde einen Run mit summary
    latest_run = None
    for run in reversed(percentile_runs):
        summary_path = run / "summary.txt"
        if summary_path.exists():
            latest_run = run
            break
    
    if latest_run is None:
        print("Keine Experiment 10.2 percentile Runs mit Summary gefunden!")
        return
    
    print(f"\nAnalysiere: {latest_run.name}")
    
    # Lese summary.txt
    summary_path = latest_run / "summary.txt"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary_content = f.read()
            print("\n" + summary_content)
    
    # Lese training_log.csv um weitere Informationen zu bekommen
    training_log_path = latest_run / "training_log.csv"
    if training_log_path.exists():
        try:
            df_log = pd.read_csv(training_log_path)
            print("\n" + "-"*80)
            print("TRAININGS-STATISTIKEN")
            print("-"*80)
            
            # Zeige letzte Zeile (bestes Epoch)
            if len(df_log) > 0:
                last_row = df_log.iloc[-1]
                print(f"Letztes Epoch:")
                for col in ['epoch', 'train_loss', 'val_loss', 'val_f1', 'val_recall', 'val_precision']:
                    if col in df_log.columns:
                        print(f"  {col}: {last_row[col]}")
                        
                # Finde bestes F1 Score
                if 'val_f1' in df_log.columns:
                    best_idx = df_log['val_f1'].idxmax()
                    best_f1 = df_log.loc[best_idx, 'val_f1']
                    best_epoch = df_log.loc[best_idx, 'epoch']
                    print(f"\nBestes Validation F1: {best_f1:.4f} (Epoch {best_epoch})")
        except Exception as e:
            print(f"Error reading training log: {e}")
    
    # Versuche, auf die Daten zuzugreifen
    try:
        data_path = latest_run / "data_split_seed42_all.npz"
        if data_path.exists():
            data = np.load(data_path, allow_pickle=True)
            
            num_samples = int(data['num_samples'])
            num_labels = int(data['num_labels'])
            
            print(f"\n" + "-"*80)
            print("DATEN-STRUKTUR")
            print("-"*80)
            print(f"Gesamt Samples: {num_samples}")
            print(f"Gesamt Labels: {num_labels}")
            print(f"Test Samples: {len(data['test_indices'])}")
            
            # Zeige Label-Nummern
            fg_names = get_functional_groups_names()
            print(f"\nAnzahl definierter Functional Groups: {len(fg_names)}")
    except Exception as e:
        print(f"Error reading data: {e}")


if __name__ == "__main__":
    cnn_analysis = analyze_cnn_baseline()
    analyze_experiment_10_2()
    
    print("\n" + "="*80)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*80)
