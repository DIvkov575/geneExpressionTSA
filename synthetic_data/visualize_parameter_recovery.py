#!/usr/bin/env python3
"""
Visualize parameter recovery results for custom ARIMA models.
Creates comparison plots showing how well each model recovers true parameters.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# True parameters
TRUE_AR = 0.7
TRUE_MA = 0.3

# Estimated parameters from evaluation
results = {
    'Model': ['MultiSeriesARIMA', 'StatsModels ARIMA', 'True Values'],
    'AR': [0.6716, 0.6932, TRUE_AR],
    'MA': [0.3128, 0.2994, TRUE_MA],
    'AR_Error': [0.0284, 0.0068, 0.0],
    'MA_Error': [0.0128, 0.0006, 0.0],
    'Total_Error': [0.0412, 0.0075, 0.0]
}

df = pd.DataFrame(results)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ARIMA Parameter Recovery Evaluation\nSynthetic Data: ARIMA(1,1,1) with AR=0.7, MA=0.3', 
             fontsize=16, fontweight='bold')

# Color scheme
colors = ['#3498db', '#2ecc71', '#e74c3c']
error_color = '#e67e22'

# Plot 1: AR Parameter Comparison
ax1 = axes[0, 0]
models = df['Model'][:2]  # Exclude "True Values"
ar_values = df['AR'][:2]
x_pos = np.arange(len(models))

bars1 = ax1.bar(x_pos, ar_values, color=colors[:2], alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axhline(y=TRUE_AR, color='red', linestyle='--', linewidth=2, label=f'True AR = {TRUE_AR}')
ax1.set_ylabel('AR Coefficient', fontsize=12, fontweight='bold')
ax1.set_title('AR Parameter Recovery', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, ar_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontweight='bold')

# Plot 2: MA Parameter Comparison
ax2 = axes[0, 1]
ma_values = df['MA'][:2]

bars2 = ax2.bar(x_pos, ma_values, color=colors[:2], alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(y=TRUE_MA, color='red', linestyle='--', linewidth=2, label=f'True MA = {TRUE_MA}')
ax2.set_ylabel('MA Coefficient', fontsize=12, fontweight='bold')
ax2.set_title('MA Parameter Recovery', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, ma_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontweight='bold')

# Plot 3: Error Comparison
ax3 = axes[1, 0]
ar_errors = df['AR_Error'][:2]
ma_errors = df['MA_Error'][:2]

x = np.arange(len(models))
width = 0.35

bars3a = ax3.bar(x - width/2, ar_errors, width, label='AR Error', 
                 color=error_color, alpha=0.7, edgecolor='black')
bars3b = ax3.bar(x + width/2, ma_errors, width, label='MA Error', 
                 color='#9b59b6', alpha=0.7, edgecolor='black')

ax3.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
ax3.set_title('Parameter Estimation Errors', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=15, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars3a, bars3b]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)

# Plot 4: Total Error Comparison
ax4 = axes[1, 1]
total_errors = df['Total_Error'][:2]

bars4 = ax4.bar(x_pos, total_errors, color=['#e74c3c', '#27ae60'], 
                alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Total Absolute Error', fontsize=12, fontweight='bold')
ax4.set_title('Overall Parameter Recovery Performance', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models, rotation=15, ha='right')
ax4.grid(axis='y', alpha=0.3)

# Add value labels and percentage
for i, (bar, val) in enumerate(zip(bars4, total_errors)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}\n({val/(TRUE_AR+TRUE_MA)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# Add text box with summary
textstr = '\n'.join([
    'Summary:',
    f'• MultiSeriesARIMA: 4-5% error',
    f'• StatsModels: <1% error',
    f'• Both within statistical tolerance',
    f'• Custom implementation validated ✓'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax4.text(0.95, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('parameter_recovery_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: parameter_recovery_comparison.png")

# Create a second figure showing relative errors
fig2, ax = plt.subplots(figsize=(10, 6))

# Calculate relative errors as percentages
ar_rel_errors = (df['AR_Error'][:2] / TRUE_AR) * 100
ma_rel_errors = (df['MA_Error'][:2] / TRUE_MA) * 100

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, ar_rel_errors, width, label='AR Relative Error (%)', 
               color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, ma_rel_errors, width, label='MA Relative Error (%)', 
               color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
ax.set_title('Parameter Recovery: Relative Errors\nSynthetic Data: ARIMA(1,1,1)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold')

# Add threshold line at 10%
ax.axhline(y=10, color='green', linestyle='--', linewidth=2, 
           label='10% Threshold (Acceptable)', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig('parameter_recovery_relative_errors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: parameter_recovery_relative_errors.png")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  1. parameter_recovery_comparison.png")
print("  2. parameter_recovery_relative_errors.png")
print("\nKey Findings:")
print("  • MultiSeriesARIMA: Good recovery (4-5% error)")
print("  • StatsModels ARIMA: Excellent recovery (<1% error)")
print("  • Both models successfully distinguish original parameters")
print("="*70)
