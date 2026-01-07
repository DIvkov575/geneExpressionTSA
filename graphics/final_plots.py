#!/usr/bin/env python3
"""
Final clean plots - WITH and WITHOUT neural networks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_all_models():
    """Load all available forecast results."""
    results = {}
    base_dir = "../run_forward"
    
    # All models including neural networks
    files = {
        'Naive': f'{base_dir}/naive_walk_forward_results.csv',
        'ARIMA': f'{base_dir}/arima_statsmodels_column_1_results.csv',
        'GBM': f'{base_dir}/gbm_column_1_results.csv',
        'NBEATS': f'{base_dir}/nbeats_column_1_results.csv', 
        'TFT': f'{base_dir}/tft_column_1_results.csv'
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if all(col in df.columns for col in ['step', 'actual', 'forecast', 'absolute_error']):
                    mae = df['absolute_error'].mean()
                    results[name] = {'data': df, 'mae': mae}
                    print(f"✓ {name}: {len(df)} points, MAE={mae:.4f}")
            except Exception as e:
                print(f"✗ {name}: Failed to load - {e}")
    
    return results

def create_comparison_plots(results):
    """Create side-by-side comparison: WITH vs WITHOUT neural networks."""
    
    # Split models
    stable_models = {k: v for k, v in results.items() 
                    if k not in ['NBEATS', 'TFT'] and v['mae'] < 2.0}
    all_models = {k: v for k, v in results.items() if v['mae'] < 100000}  # Filter extreme outliers
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy WITHOUT neural networks
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (name, model_data) in enumerate(stable_models.items()):
        df = model_data['data']
        ax1.plot(df['step'], df['absolute_error'], 
                color=colors[i % len(colors)], label=f"{name} (MAE: {model_data['mae']:.3f})", 
                alpha=0.8, linewidth=1.5)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('WITHOUT Neural Networks (Stable Models)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy WITH neural networks (log scale)
    ax2 = axes[0, 1]
    for i, (name, model_data) in enumerate(all_models.items()):
        df = model_data['data']
        errors = df['absolute_error'].replace(0, 1e-10)
        ax2.plot(df['step'], errors, 
                color=colors[i % len(colors)], label=f"{name} (MAE: {model_data['mae']:.1f})", 
                alpha=0.8, linewidth=1.5)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.set_yscale('log')
    ax2.set_title('WITH Neural Networks (All Models, Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Forecasts WITHOUT neural networks
    ax3 = axes[1, 0]
    first_df = list(stable_models.values())[0]['data']
    ax3.plot(first_df['step'], first_df['actual'], 'k-', 
            linewidth=2, label='Actual', alpha=0.9)
    
    for i, (name, model_data) in enumerate(stable_models.items()):
        df = model_data['data']
        ax3.plot(df['step'], df['forecast'], '--',
                color=colors[i % len(colors)], label=name, alpha=0.7, linewidth=1.5)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Value')
    ax3.set_title('Forecasts: Stable Models Only')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison
    ax4 = axes[1, 1]
    
    # Performance bars for stable models
    stable_names = list(stable_models.keys())
    stable_maes = [stable_models[name]['mae'] for name in stable_names]
    
    bars = ax4.bar(range(len(stable_names)), stable_maes, 
                  color=colors[:len(stable_names)], alpha=0.7)
    ax4.set_xticks(range(len(stable_names)))
    ax4.set_xticklabels(stable_names, rotation=45)
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Stable Models Performance')
    
    # Add value labels
    for bar, mae in zip(bars, stable_maes):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + mae*0.01,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved: model_comparison_analysis.png")

def create_clean_extrapolation():
    """Simple extrapolation from column 2."""
    df = pd.read_csv('../data/CRE.csv')
    col2_data = df['2'].values
    
    plt.figure(figsize=(12, 8))
    
    # Original data
    steps = np.arange(len(col2_data))
    plt.plot(steps, col2_data, 'k-', linewidth=2, label='Column 2 Data', alpha=0.8)
    
    # Linear trend extrapolation
    recent = col2_data[-50:]  # Last 50 points
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    
    extrap_steps = 100
    extrap_x = np.arange(len(col2_data), len(col2_data) + extrap_steps)
    extrap_y = slope * (extrap_x - len(col2_data) + len(recent)) + recent[-1]
    
    plt.plot(extrap_x, extrap_y, 'r--', linewidth=2, label='Linear Trend Extrapolation', alpha=0.8)
    plt.axvline(x=len(col2_data)-1, color='red', linestyle=':', alpha=0.7, label='Extrapolation Start')
    
    plt.xlabel('Time Step')
    plt.ylabel('Column 2 Value')
    plt.title('Column 2 Extrapolation (Clean)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('column2_clean_extrapolation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved: column2_clean_extrapolation.png")
    
    # Summary
    print(f"   Range: [{col2_data.min():.4f}, {col2_data.max():.4f}]")
    print(f"   Final: {col2_data[-1]:.4f} → Extrapolated: {extrap_y[-1]:.4f}")
    print(f"   Trend: {slope:.6f} per step")

def main():
    print("🧼 Final Clean Graphics")
    print("=" * 30)
    
    # Load all models
    results = load_all_models()
    
    if not results:
        print("❌ No models loaded")
        return
    
    # Create comparison plots
    create_comparison_plots(results)
    
    # Create clean extrapolation
    create_clean_extrapolation()
    
    print(f"\n✅ Generated clean graphics from {len(results)} models")
    print("\nSummary:")
    for name, model_data in sorted(results.items(), key=lambda x: x[1]['mae']):
        mae = model_data['mae']
        status = "✓ Stable" if mae < 2.0 else "⚠ Unstable" if mae < 1000 else "❌ Broken"
        print(f"  {name}: MAE={mae:.4f} {status}")

if __name__ == "__main__":
    main()