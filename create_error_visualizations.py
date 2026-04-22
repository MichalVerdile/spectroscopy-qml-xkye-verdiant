"""
Erstelle Visualisierungen für die Fehleranalyse
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setze Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Lade die Daten
pickle_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/benchmark/cnn/models/ir/original/results.pickle")
csv_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/reports/cnn_baseline_error_analysis.csv")

print("Lade Daten...")
with open(pickle_path, 'rb') as f:
    results = pickle.load(f)

y_true = results['tgt']
y_pred = results['pred']
y_pred_binary = (y_pred > 0.5).astype(int)

metrics_df = pd.read_csv(csv_path)

# Sortiere und nimm Top 15
metrics_top = metrics_df.sort_values('Error_Rate', ascending=False).head(15)

print(f"Erstelle Visualisierungen...")

# ============================================
# Abbildung 1: Top Fehlerverursacher
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#d62728' if err > 0.7 else '#ff7f0e' if err > 0.5 else '#2ca02c' 
          for err in metrics_top['Error_Rate']]

bars = ax.barh(range(len(metrics_top)), metrics_top['Error_Rate'] * 100, color=colors, alpha=0.7, edgecolor='black')

ax.set_yticks(range(len(metrics_top)))
ax.set_yticklabels([f"[{int(row['Label_ID'])}] {row['FG_Name']}" for _, row in metrics_top.iterrows()],
                     fontsize=11)
ax.set_xlabel('Fehlerquote (%)', fontsize=12, fontweight='bold')
ax.set_title('CNN IR Baseline: Top 15 Fehlerverursacher (Functional Groups)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 105)

# Füge Werte auf den Balken hinzu
for i, (bar, row) in enumerate(zip(bars, metrics_top.itertuples())):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}% ({int(row.Positive_Samples)} samples)',
            ha='left', va='center', fontsize=9, fontweight='bold')

ax.axvline(x=60, color='red', linestyle='--', linewidth=2, alpha=0.5, label='60% Fehler-Schwelle')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
output_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/reports/error_analysis_top_errors.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================
# Abbildung 2: F1 Score vs. Fehlerquote
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))

# Färbe nach Fehlerquote
scatter = ax.scatter(metrics_df['Positive_Samples'], metrics_df['F1'], 
                     c=metrics_df['Error_Rate']*100, s=metrics_df['Positive_Samples']/10,
                     cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)

# Markiere Top Fehler
top_errors = metrics_df.sort_values('Error_Rate', ascending=False).head(5)
for _, row in top_errors.iterrows():
    ax.annotate(f"[{int(row['Label_ID'])}] {row['FG_Name'][:15]}", 
                xy=(row['Positive_Samples'], row['F1']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Anzahl positive Samples (im Test-Set)', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('CNN IR Baseline: F1 Score vs. Datenmenge\n(Größe = Anzahl Samples, Farbe = Fehlerquote)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xscale('log')
ax.grid(alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Fehlerquote (%)', fontsize=11, fontweight='bold')

plt.tight_layout()
output_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/reports/error_analysis_f1_vs_samples.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================
# Abbildung 3: Precision vs. Recall Trade-off
# ============================================
fig, ax = plt.subplots(figsize=(12, 8))

# Plot für alle Gruppen
scatter = ax.scatter(metrics_df['Recall'], metrics_df['Precision'],
                     c=metrics_df['F1'], s=metrics_df['Positive_Samples']/10,
                     cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)

# Markiere Top 5 Fehler
top_errors = metrics_df.sort_values('Error_Rate', ascending=False).head(5)
for _, row in top_errors.iterrows():
    ax.plot(row['Recall'], row['Precision'], 'r*', markersize=20, markeredgecolor='black', markeredgewidth=1)
    ax.annotate(f"{row['FG_Name'][:12]}", xy=(row['Recall'], row['Precision']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# Diagonale (perfakte Balance)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect Balance')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('CNN IR Baseline: Precision vs. Recall Trade-off\n(Größe = Anzahl Samples, Farbe = F1 Score)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('F1 Score', fontsize=11, fontweight='bold')

plt.tight_layout()
output_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/reports/error_analysis_precision_recall.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================
# Abbildung 4: Confusion Matrix für Top Fehler
# ============================================
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

top_5_errors = metrics_df.sort_values('Error_Rate', ascending=False).head(5)

for idx, (ax, (_, row)) in enumerate(zip(axes[:5], top_5_errors.iterrows())):
    label_id = int(row['Label_ID'])
    y_true_label = y_true[:, label_id]
    y_pred_label = y_pred_binary[:, label_id]
    
    cm = confusion_matrix(y_true_label, y_pred_label)
    
    # Normalisiere für bessere Visualisierung
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Zeige Werte
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Neg (Pred)', 'Pos (Pred)'])
    ax.set_yticklabels(['Neg (True)', 'Pos (True)'])
    ax.set_title(f'[{label_id}] {row["FG_Name"]}\n(Fehler: {row["Error_Rate"]:.1%}, F1: {row["F1"]:.3f})',
                fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Ratio')

# Verstecke den letzten Plot
axes[5].axis('off')

plt.suptitle('CNN IR Baseline: Confusion Matrices der Top 5 Fehlergruppen', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
output_path = Path("/Users/leonakryeziu/ZHAW_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/reports/error_analysis_confusion_matrices.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================
# Summary Statistics
# ============================================
print("\n" + "="*80)
print("ZUSAMMENFASSUNG DER FEHLERANALYSE")
print("="*80)
print(f"\nAnzahl Functional Groups: {len(metrics_df)}")
print(f"Gesamt Test Samples (Label-Vorhersagen): {y_true.shape[0] * y_true.shape[1]:,}")
print(f"\nFehler-Statistiken:")
print(f"  Durchschnittliche Fehlerquote: {metrics_df['Error_Rate'].mean():.1%}")
print(f"  Median Fehlerquote: {metrics_df['Error_Rate'].median():.1%}")
print(f"  Max Fehlerquote: {metrics_df['Error_Rate'].max():.1%} ({metrics_df.loc[metrics_df['Error_Rate'].idxmax(), 'FG_Name']})")
print(f"\nF1 Score Statistiken:")
print(f"  Durchschnittlicher F1 Score: {metrics_df['F1'].mean():.3f}")
print(f"  Median F1 Score: {metrics_df['F1'].median():.3f}")
print(f"  Minimum F1 Score: {metrics_df['F1'].min():.3f} ({metrics_df.loc[metrics_df['F1'].idxmin(), 'FG_Name']})")

print("\nVisualisierungen erstellt:")
print("  ✓ error_analysis_top_errors.png")
print("  ✓ error_analysis_f1_vs_samples.png")
print("  ✓ error_analysis_precision_recall.png")
print("  ✓ error_analysis_confusion_matrices.png")

print("\n" + "="*80)
