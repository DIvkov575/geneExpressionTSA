#!/usr/bin/env python3
"""
Simple, clean forecasting plots. No bullshit.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data():
    """Load available forecast results."""
    results = {}
    base_dir = "../run_forward"
    
    # Only load what actually exists and works
    files = {
        'ARIMA': f'{base_dir}/arima_statsmodels_column_1_results.csv',
        'Naive': f'{base_dir}/naive_walk_forward_results.csv',
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'absolute_error' in df.columns:
                results[name] = df
                print(f"✓ {name}: {len(df)} points, MAE={df['absolute_error'].mean():.4f}")
    
    return results

def plot_accuracy_simple(results):
    """Simple accuracy plot."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, df) in enumerate(results.items()):
        plt.plot(df['step'], df['absolute_error'], 
                color=colors[i % len(colors)], label=name, alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Forecast Accuracy vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('accuracy_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: accuracy_plot.png")

def plot_accuracy_log(results):
    """Simple accuracy plot with log scale."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, df) in enumerate(results.items()):
        # Filter out zeros for log scale
        errors = df['absolute_error'].replace(0, np.nan)
        plt.plot(df['step'], errors, 
                color=colors[i % len(colors)], label=name, alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error (log scale)')
    plt.yscale('log')
    plt.title('Forecast Accuracy vs Time (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('accuracy_plot_log.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: accuracy_plot_log.png")

def plot_forecasts_simple(results):
    """Simple forecast vs actual plot."""
    plt.figure(figsize=(14, 8))
    
    # Get first model for actual values
    first_df = list(results.values())[0]
    plt.plot(first_df['step'], first_df['actual'], 'k-', 
             linewidth=2, label='Actual', alpha=0.8)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, df) in enumerate(results.items()):
        plt.plot(df['step'], df['forecast'], '--', 
                color=colors[i % len(colors)], label=name, alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Forecasts vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('forecasts_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: forecasts_plot.png")

def plot_scatter_simple(results):
    """Simple scatter plots."""
    n_models = len(results)
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, df) in enumerate(results.items()):
        ax = axes[i]
        
        # Sample data if too many points
        if len(df) > 1000:
            sample = df.sample(500)
        else:
            sample = df
        
        ax.scatter(sample['actual'], sample['forecast'], 
                  color=colors[i % len(colors)], alpha=0.6, s=10)
        
        # Perfect prediction line
        min_val = min(sample['actual'].min(), sample['forecast'].min())
        max_val = max(sample['actual'].max(), sample['forecast'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Forecast')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: scatter_plots.png")

def main():
    print("📊 Simple Forecast Plots")
    print("=" * 30)
    
    # Load data
    results = load_data()
    
    if not results:
        print("❌ No valid data found")
        return
    
    # Generate plots
    plot_accuracy_simple(results)
    plot_accuracy_log(results)
    plot_forecasts_simple(results)
    plot_scatter_simple(results)
    
    print(f"\n✅ Generated 4 simple plots from {len(results)} models")

if __name__ == "__main__":
    main()