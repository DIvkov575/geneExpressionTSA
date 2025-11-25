#!/usr/bin/env python3
"""
Save parameter recovery performance scores to model_comparison.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# Performance data from evaluation
performance_data = {
    'MultiSeriesARIMA': {
        'ar_true': 0.7000,
        'ar_est': 0.6716,
        'ar_error': 0.0284,
        'ar_rel_error': 4.05,
        'ma_true': 0.3000,
        'ma_est': 0.3128,
        'ma_error': 0.0128,
        'ma_rel_error': 4.26,
        'total_error': 0.0412,
        'status': 'SUCCESSFUL RECOVERY'
    },
    'StatsModels_ARIMA': {
        'ar_true': 0.7000,
        'ar_est': 0.6932,
        'ar_error': 0.0068,
        'ar_rel_error': 0.97,
        'ma_true': 0.3000,
        'ma_est': 0.2994,
        'ma_error': 0.0006,
        'ma_rel_error': 0.21,
        'total_error': 0.0075,
        'status': 'EXCELLENT RECOVERY',
        'aic': 270.02,
        'bic': 277.81
    },
    'MultiHorizonARIMA_v2': {
        'ar_true': 0.7000,
        'ar_est': None,
        'ar_error': None,
        'ar_rel_error': None,
        'ma_true': 0.3000,
        'ma_est': None,
        'ma_error': None,
        'ma_rel_error': None,
        'total_error': None,
        'status': 'NOT APPLICABLE (AR-only model)',
        'note': 'Uses 5 AR lags to approximate AR+MA dynamics'
    }
}

# Generate the performance scores section
output = []
output.append("\n" + "="*80)
output.append("PARAMETER RECOVERY PERFORMANCE SCORES")
output.append("="*80)
output.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output.append(f"Synthetic Data: ARIMA(1,1,1) with AR=0.7, MA=0.3, d=1, n=100")
output.append("")

# Summary table
output.append("SUMMARY TABLE:")
output.append("-" * 80)
output.append(f"{'Model':<25} {'AR Est':<10} {'MA Est':<10} {'AR Err':<10} {'MA Err':<10} {'Total Err':<10}")
output.append("-" * 80)

for model_name, data in performance_data.items():
    model_display = model_name.replace('_', ' ')
    ar_est = f"{data['ar_est']:.4f}" if data['ar_est'] is not None else "N/A"
    ma_est = f"{data['ma_est']:.4f}" if data['ma_est'] is not None else "N/A"
    ar_err = f"{data['ar_error']:.4f}" if data['ar_error'] is not None else "N/A"
    ma_err = f"{data['ma_error']:.4f}" if data['ma_error'] is not None else "N/A"
    total_err = f"{data['total_error']:.4f}" if data['total_error'] is not None else "N/A"
    
    output.append(f"{model_display:<25} {ar_est:<10} {ma_est:<10} {ar_err:<10} {ma_err:<10} {total_err:<10}")

output.append("-" * 80)
output.append("")

# Detailed scores for each model
output.append("DETAILED PERFORMANCE SCORES:")
output.append("")

for model_name, data in performance_data.items():
    model_display = model_name.replace('_', ' ')
    output.append(f"{model_display}:")
    output.append(f"  Status: {data['status']}")
    output.append("")
    
    if data['ar_est'] is not None:
        output.append(f"  AR Parameter:")
        output.append(f"    True value:      {data['ar_true']:.4f}")
        output.append(f"    Estimated:       {data['ar_est']:.4f}")
        output.append(f"    Absolute error:  {data['ar_error']:.4f}")
        output.append(f"    Relative error:  {data['ar_rel_error']:.2f}%")
        output.append("")
        
        output.append(f"  MA Parameter:")
        output.append(f"    True value:      {data['ma_true']:.4f}")
        output.append(f"    Estimated:       {data['ma_est']:.4f}")
        output.append(f"    Absolute error:  {data['ma_error']:.4f}")
        output.append(f"    Relative error:  {data['ma_rel_error']:.2f}%")
        output.append("")
        
        output.append(f"  Overall:")
        output.append(f"    Total error:     {data['total_error']:.4f}")
        
        if 'aic' in data:
            output.append(f"    AIC:             {data['aic']:.2f}")
            output.append(f"    BIC:             {data['bic']:.2f}")
    else:
        output.append(f"  Note: {data.get('note', 'No parameter estimates available')}")
    
    output.append("")

# Performance ranking
output.append("PERFORMANCE RANKING (by Total Error):")
output.append("-" * 80)

ranked = [(name, data['total_error']) for name, data in performance_data.items() 
          if data['total_error'] is not None]
ranked.sort(key=lambda x: x[1])

for i, (model_name, error) in enumerate(ranked, 1):
    model_display = model_name.replace('_', ' ')
    grade = "EXCELLENT" if error < 0.01 else "GOOD" if error < 0.05 else "ACCEPTABLE"
    output.append(f"  {i}. {model_display:<25} Error: {error:.4f}  [{grade}]")

output.append("")

# Accuracy percentages
output.append("PARAMETER RECOVERY ACCURACY:")
output.append("-" * 80)

for model_name, data in performance_data.items():
    if data['ar_est'] is not None:
        model_display = model_name.replace('_', ' ')
        ar_accuracy = 100 - data['ar_rel_error']
        ma_accuracy = 100 - data['ma_rel_error']
        avg_accuracy = (ar_accuracy + ma_accuracy) / 2
        
        output.append(f"{model_display}:")
        output.append(f"  AR accuracy: {ar_accuracy:.2f}%")
        output.append(f"  MA accuracy: {ma_accuracy:.2f}%")
        output.append(f"  Average:     {avg_accuracy:.2f}%")
        output.append("")

# Statistical significance
output.append("STATISTICAL SIGNIFICANCE ANALYSIS:")
output.append("-" * 80)
output.append("Standard Error (SE) ≈ 1/√n ≈ 0.10 for n=100")
output.append("")

for model_name, data in performance_data.items():
    if data['ar_error'] is not None:
        model_display = model_name.replace('_', ' ')
        ar_se_ratio = data['ar_error'] / 0.10
        ma_se_ratio = data['ma_error'] / 0.10
        
        ar_sig = "Not significant" if ar_se_ratio < 0.5 else "Marginally significant" if ar_se_ratio < 1.0 else "Significant"
        ma_sig = "Not significant" if ma_se_ratio < 0.5 else "Marginally significant" if ma_se_ratio < 1.0 else "Significant"
        
        output.append(f"{model_display}:")
        output.append(f"  AR error / SE: {ar_se_ratio:.2f} → {ar_sig}")
        output.append(f"  MA error / SE: {ma_se_ratio:.2f} → {ma_sig}")
        output.append("")

output.append("="*80)
output.append("")

# Write to file
output_text = "\n".join(output)
output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_comparison.txt')

with open(output_file, 'a') as f:
    f.write(output_text)

print("✓ Performance scores saved to model_comparison.txt")
print("\nSummary:")
print("  • MultiSeriesARIMA: 95.8% average accuracy")
print("  • StatsModels ARIMA: 99.4% average accuracy")
print("  • Both successfully distinguish original parameters")
