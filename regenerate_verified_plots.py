#!/usr/bin/env python3
"""
Regenerate individual plots with CORRECT data from actual CSV files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

print("\n" + "="*80)
print("REGENERATING PLOTS WITH CORRECT DATA")
print("="*80)

# Load the REAL CSV files with actual data
cnn_df = pd.read_csv('CNN_error_analysis.csv').sort_values('error_rate', ascending=False)
ttn_df = pd.read_csv('TTN_error_analysis.csv').sort_values('error_rate', ascending=False)

print(f"\n✅ CNN Data: {len(cnn_df)} groups")
print(f"✅ TTN Data: {len(ttn_df)} groups")

# ============================================================================
# VERIFICATION: Print Top 10 actual values
# ============================================================================
print("\n📊 VERIFICATION - TOP 10 ACTUAL VALUES:")
print("\n--- CNN TOP 10 ---")
for i, (idx, row) in enumerate(cnn_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['label_name']:25s} {row['error_rate']*100:6.2f}%")

print("\n--- TTN TOP 10 ---")
for i, (idx, row) in enumerate(ttn_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['label_name']:25s} {row['error_rate']*100:6.2f}%")

# ============================================================================
# CNN PLOT with CORRECT data
# ============================================================================
print("\n\nRegenerating CNN plot with correct data...")

fig = plt.figure(figsize=(16, 12))

# Plot 1
ax1 = plt.subplot(2, 2, 1)
colors_cnn = ['#CC0000' if x > 50 else '#FF6B6B' if x > 20 else '#FFB3B3' for x in cnn_df['error_rate']]
bars = ax1.barh(range(len(cnn_df)), cnn_df['error_rate']*100, color=colors_cnn, edgecolor='black', linewidth=0.8)
ax1.set_yticks(range(len(cnn_df)))
ax1.set_yticklabels(cnn_df['label_name'], fontsize=9)
ax1.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('CNN Baseline - All 37 Functional Groups (Ranked by Error)', fontsize=12, fontweight='bold')
ax1.set_xlim([0, 105])
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(cnn_df.iterrows()):
    ax1.text(row['error_rate']*100 + 1.5, i, f"{row['error_rate']*100:.1f}%", 
            va='center', fontsize=8)

# Plot 2
ax2 = plt.subplot(2, 2, 2)
error_pct = cnn_df['error_rate'] * 100
ax2.hist(error_pct, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axvline(error_pct.mean(), color='#CC0000', linestyle='--', linewidth=3,
           label=f'Mean: {error_pct.mean():.1f}%')
ax2.axvline(error_pct.median(), color='#990000', linestyle=':', linewidth=3,
           label=f'Median: {error_pct.median():.1f}%')
ax2.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('# Functional Groups', fontsize=11, fontweight='bold')
ax2.set_title('CNN Error Rate Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3
ax3 = plt.subplot(2, 2, 3)
top_10 = cnn_df.head(10)
bars = ax3.bar(range(len(top_10)), top_10['error_rate']*100, 
               color='#CC0000', alpha=0.8, edgecolor='black', linewidth=1.2)
ax3.set_xticks(range(len(top_10)))
ax3.set_xticklabels(top_10['label_name'], rotation=45, ha='right', fontsize=10, fontweight='bold')
ax3.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('CNN - Top 10 Problem Groups', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 105])
ax3.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(top_10.iterrows()):
    ax3.text(i, row['error_rate']*100 + 2, f"{row['error_rate']*100:.1f}%", 
            ha='center', fontsize=10, fontweight='bold')

# Plot 4
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

error_pct_for_stats = cnn_df['error_rate'] * 100
summary_text = f"""CNN BASELINE ERROR ANALYSIS (VERIFIED)

Performance Metrics:
  Mean Error Rate:     {error_pct_for_stats.mean():.2f}%
  Median Error Rate:   {error_pct_for_stats.median():.2f}%
  Std Deviation:       {error_pct_for_stats.std():.2f}%
  
  Max Error:           {error_pct_for_stats.max():.1f}% ({cnn_df.loc[cnn_df['error_rate'].idxmax(), 'label_name']})
  Min Error:           {error_pct_for_stats.min():.1f}% ({cnn_df.loc[cnn_df['error_rate'].idxmin(), 'label_name']})

Top 5 Problem Groups:
  1. {cnn_df.iloc[0]['label_name']}: {cnn_df.iloc[0]['error_rate']*100:.1f}%
  2. {cnn_df.iloc[1]['label_name']}: {cnn_df.iloc[1]['error_rate']*100:.1f}%
  3. {cnn_df.iloc[2]['label_name']}: {cnn_df.iloc[2]['error_rate']*100:.1f}%
  4. {cnn_df.iloc[3]['label_name']}: {cnn_df.iloc[3]['error_rate']*100:.1f}%
  5. {cnn_df.iloc[4]['label_name']}: {cnn_df.iloc[4]['error_rate']*100:.1f}%

Root Cause: Class Imbalance (rare groups)
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#FFE6E6', alpha=0.6, edgecolor='#CC0000', linewidth=2))

plt.tight_layout()
plt.savefig('CNN_error_analysis_VERIFIED.png', dpi=300, bbox_inches='tight')
print("✅ Saved: CNN_error_analysis_VERIFIED.png")

# ============================================================================
# TTN PLOT with CORRECT data
# ============================================================================
print("Regenerating TTN plot with correct data...")

fig = plt.figure(figsize=(16, 12))

# Plot 1
ax1 = plt.subplot(2, 2, 1)
colors_ttn = ['#006666' if x > 30 else '#4ECDC4' if x > 10 else '#A3E9E5' for x in ttn_df['error_rate']]
bars = ax1.barh(range(len(ttn_df)), ttn_df['error_rate']*100, color=colors_ttn, edgecolor='black', linewidth=0.8)
ax1.set_yticks(range(len(ttn_df)))
ax1.set_yticklabels(ttn_df['label_name'], fontsize=9)
ax1.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('TTN 10.2 - All 37 Functional Groups (Ranked by Error)', fontsize=12, fontweight='bold')
ax1.set_xlim([0, max(ttn_df['error_rate']*100) + 10])
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(ttn_df.iterrows()):
    if row['error_rate']*100 > 0.5:
        ax1.text(row['error_rate']*100 + 0.5, i, f"{row['error_rate']*100:.1f}%", 
                va='center', fontsize=8)

# Plot 2
ax2 = plt.subplot(2, 2, 2)
error_pct_ttn = ttn_df['error_rate'] * 100
ax2.hist(error_pct_ttn, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axvline(error_pct_ttn.mean(), color='#008080', linestyle='--', linewidth=3,
           label=f'Mean: {error_pct_ttn.mean():.1f}%')
ax2.axvline(error_pct_ttn.median(), color='#006666', linestyle=':', linewidth=3,
           label=f'Median: {error_pct_ttn.median():.1f}%')
ax2.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('# Functional Groups', fontsize=11, fontweight='bold')
ax2.set_title('TTN 10.2 Error Rate Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3
ax3 = plt.subplot(2, 2, 3)
top_10_ttn = ttn_df.head(10)
bars = ax3.bar(range(len(top_10_ttn)), top_10_ttn['error_rate']*100, 
               color='#008080', alpha=0.8, edgecolor='black', linewidth=1.2)
ax3.set_xticks(range(len(top_10_ttn)))
ax3.set_xticklabels(top_10_ttn['label_name'], rotation=45, ha='right', fontsize=10, fontweight='bold')
ax3.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('TTN 10.2 - Top 10 Problem Groups', fontsize=12, fontweight='bold')
ax3.set_ylim([0, max(top_10_ttn['error_rate']*100) + 10])
ax3.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(top_10_ttn.iterrows()):
    ax3.text(i, row['error_rate']*100 + 1, f"{row['error_rate']*100:.1f}%", 
            ha='center', fontsize=10, fontweight='bold')

# Plot 4
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

summary_text = f"""TTN 10.2 ERROR ANALYSIS (VERIFIED)

Performance Metrics:
  Mean Error Rate:     {error_pct_ttn.mean():.2f}%
  Median Error Rate:   {error_pct_ttn.median():.2f}%
  Std Deviation:       {error_pct_ttn.std():.2f}%
  
  Max Error:           {error_pct_ttn.max():.1f}% ({ttn_df.loc[ttn_df['error_rate'].idxmax(), 'label_name']})
  Min Error:           {error_pct_ttn.min():.1f}% ({ttn_df.loc[ttn_df['error_rate'].idxmin(), 'label_name']})

Top 5 Problem Groups:
  1. {ttn_df.iloc[0]['label_name']}: {ttn_df.iloc[0]['error_rate']*100:.1f}%
  2. {ttn_df.iloc[1]['label_name']}: {ttn_df.iloc[1]['error_rate']*100:.1f}%
  3. {ttn_df.iloc[2]['label_name']}: {ttn_df.iloc[2]['error_rate']*100:.1f}%
  4. {ttn_df.iloc[3]['label_name']}: {ttn_df.iloc[3]['error_rate']*100:.1f}%
  5. {ttn_df.iloc[4]['label_name']}: {ttn_df.iloc[4]['error_rate']*100:.1f}%

Root Cause: High-variance common groups
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#E0F7F6', alpha=0.6, edgecolor='#008080', linewidth=2))

plt.tight_layout()
plt.savefig('TTN_error_analysis_VERIFIED.png', dpi=300, bbox_inches='tight')
print("✅ Saved: TTN_error_analysis_VERIFIED.png")

print("\n" + "="*80)
print("✅ VERIFIED PLOTS REGENERATED WITH CORRECT DATA")
print("="*80)
