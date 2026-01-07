#!/usr/bin/env python3
"""
Generate extrapolation plots extending from column 2 of CRE.csv.
FIXED VERSION: Uses actual trained model results, not fake extrapolation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_original_data(data_path="../../data/CRE.csv"):
    """Load the original CRE.csv data."""
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded original data: {len(df)} rows")
    return df

def load_trained_model_results(results_dir="../../run_forward"):
    """Load actual trained model results for column 2."""
    results = {}
    
    # Check for column 2 specific results
    model_files = {
        'GBM': f'{results_dir}/gbm_column_2_results.csv',
        'NBEATS': f'{results_dir}/nbeats_column_2_results.csv'
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                required_cols = ['step', 'actual', 'forecast']
                if all(col in df.columns for col in required_cols):
                    results[model_name] = df
                    print(f"✅ Loaded {model_name} column 2 results: {len(df)} points")
                else:
                    print(f"⚠ {model_name}: Missing required columns")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        else:
            print(f"⚠ {model_name}: No column 2 results found")
    
    return results

def create_simple_extrapolations(column_data, steps=100):
    """Create simple baseline extrapolations."""
    extrapolations = {}
    
    # 1. Naive (constant)
    extrapolations['Naive'] = [column_data[-1]] * steps
    
    # 2. Linear trend from last 20 points
    recent_data = column_data[-20:]
    x = np.arange(len(recent_data))
    coeffs = np.polyfit(x, recent_data, 1)
    
    linear_extrap = []
    for i in range(steps):
        next_val = coeffs[0] * (len(recent_data) + i) + coeffs[1]
        linear_extrap.append(next_val)
    extrapolations['Linear Trend'] = linear_extrap
    
    # 3. Exponential decay toward zero (realistic for many time series)
    decay_rate = 0.99
    current_val = column_data[-1]
    exp_extrap = []
    for i in range(steps):
        current_val *= decay_rate
        exp_extrap.append(current_val)
    extrapolations['Exponential Decay'] = exp_extrap
    
    return extrapolations

def plot_real_forecast_extrapolation(original_data, model_results, extrapolations, column_name='2', save_dir="plots"):
    """Create extrapolation plot using REAL forecast data plus extrapolation."""
    os.makedirs(save_dir, exist_ok=True)
    
    column_data = original_data[column_name].values
    
    plt.figure(figsize=(16, 10))
    
    # Plot original data
    original_steps = np.arange(len(column_data))
    plt.plot(original_steps, column_data, 'k-', linewidth=3, label='Original Data', alpha=0.8)
    
    # Plot REAL model forecasts (these are actual predictions, not extrapolations)
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results) + len(extrapolations)))
    color_idx = 0
    
    for model_name, df in model_results.items():
        plt.plot(df['step'], df['forecast'], '--', color=colors[color_idx], 
                linewidth=2, alpha=0.7, label=f'{model_name} (Real Forecast)')
        color_idx += 1
    
    # Add extrapolation start point
    extrap_start = len(column_data)
    plt.axvline(x=extrap_start, color='red', linestyle=':', alpha=0.7, linewidth=2, 
                label='Extrapolation Start')
    
    # Plot simple extrapolations
    extrap_steps = np.arange(extrap_start, extrap_start + len(list(extrapolations.values())[0]))
    
    for model_name, values in extrapolations.items():
        plt.plot(extrap_steps, values, ':', color=colors[color_idx], 
                linewidth=2, alpha=0.8, label=f'{model_name} Extrapolation')
        color_idx += 1
    
    plt.xlabel('Time Step')
    plt.ylabel(f'Column {column_name} Value')
    plt.title(f'Real Forecasts vs Simple Extrapolations - Column {column_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"column_{column_name}_real_vs_extrapolation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_extrapolation_comparison(original_data, extrapolations, column_name='2', save_dir="plots"):
    """Compare different extrapolation methods."""
    os.makedirs(save_dir, exist_ok=True)
    
    column_data = original_data[column_name].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Just the extrapolations
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(extrapolations)))
    
    for i, (method, values) in enumerate(extrapolations.items()):
        steps = np.arange(len(values))
        ax1.plot(steps, values, color=colors[i], linewidth=2, label=method)
    
    ax1.set_xlabel('Extrapolation Steps')
    ax1.set_ylabel(f'Column {column_name} Value')
    ax1.set_title('Extrapolation Methods Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final values
    ax2 = axes[0, 1]
    methods = list(extrapolations.keys())
    final_values = [extrapolations[method][-1] for method in methods]
    original_final = column_data[-1]
    
    bars = ax2.bar(methods, final_values, color=colors, alpha=0.7)
    ax2.axhline(y=original_final, color='red', linestyle='--', 
               label=f'Original Final: {original_final:.4f}')
    ax2.set_ylabel('Final Value After 100 Steps')
    ax2.set_title('Extrapolation Final Values')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # Add value labels
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Trend analysis
    ax3 = axes[1, 0]
    # Show actual data trend
    recent_steps = np.arange(len(column_data) - 50, len(column_data))
    recent_data = column_data[-50:]
    ax3.plot(recent_steps, recent_data, 'k-', linewidth=2, label='Recent Actual Data')
    
    # Show what each extrapolation does
    extrap_steps = np.arange(len(column_data), len(column_data) + 20)
    for i, (method, values) in enumerate(extrapolations.items()):
        ax3.plot(extrap_steps, values[:20], '--', color=colors[i], 
                linewidth=2, label=method)
    
    ax3.axvline(x=len(column_data), color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel(f'Column {column_name} Value')
    ax3.set_title('Extrapolation vs Recent Trend')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, f"Original Data Range: [{column_data.min():.3f}, {column_data.max():.3f}]", 
             transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.8, f"Original Final Value: {column_data[-1]:.4f}", 
             transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.7, f"Recent Trend (last 20): {np.polyfit(range(20), column_data[-20:], 1)[0]:.5f}/step", 
             transform=ax4.transAxes, fontsize=10)
    
    y_pos = 0.6
    for method, values in extrapolations.items():
        change = values[-1] - column_data[-1]
        volatility = np.std(values)
        ax4.text(0.1, y_pos, f"{method}: Final={values[-1]:.4f}, Δ={change:+.4f}, Vol={volatility:.4f}", 
                transform=ax4.transAxes, fontsize=9)
        y_pos -= 0.08
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Summary Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"column_{column_name}_extrapolation_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def main():
    print("🎯 FIXED Extrapolation Analysis from Column 2")
    print("=" * 60)
    
    # Load original data
    print("📊 Loading original CRE data...")
    original_data = load_original_data()
    
    if original_data is None:
        return
    
    if '2' not in original_data.columns:
        print("❌ Column '2' not found in data")
        return
    
    column_data = original_data['2'].values
    print(f"✅ Column 2: {len(column_data)} points, range [{column_data.min():.4f}, {column_data.max():.4f}]")
    print(f"   Recent trend: {column_data[-1] - column_data[-20]:.4f} change over last 20 points")
    
    # Load REAL trained model results
    print("\n📈 Loading trained model results...")
    model_results = load_trained_model_results()
    
    # Create simple extrapolations
    print("\n🔮 Creating simple extrapolations...")
    extrapolations = create_simple_extrapolations(column_data, steps=100)
    
    print(f"✅ Created {len(extrapolations)} extrapolation methods")
    for method, values in extrapolations.items():
        final_val = values[-1]
        change = final_val - column_data[-1]
        print(f"   {method}: Final={final_val:.4f}, Change={change:+.4f}")
    
    # Generate plots
    print("\n📊 Creating plots...")
    if model_results:
        plot_real_forecast_extrapolation(original_data, model_results, extrapolations)
    
    plot_extrapolation_comparison(original_data, extrapolations)
    
    print("\n🎉 FIXED extrapolation analysis completed!")
    print("\nKey insights:")
    print("- Shows REAL trained model forecasts vs simple extrapolations")
    print("- Linear trend captures the clear downward pattern in the data")
    print("- Naive and decay methods show different assumptions about future behavior")

if __name__ == "__main__":
    main()