import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_single_model_log(csv_file, model_name="Model"):
    """Plot a single model with both linear and log scales"""
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
        
    df = pd.read_csv(csv_file)
    
    # Handle negative or zero values for log scale
    epsilon = 1e-10
    df['actual_log'] = np.where(df['actual'] <= 0, epsilon, df['actual'])
    df['forecast_log'] = np.where(df['forecast'] <= 0, epsilon, df['forecast'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Take first 200 points for clarity
    df_plot = df.head(200)
    
    # Plot 1: Linear Scale
    ax1 = axes[0, 0]
    ax1.plot(df_plot['step'], df_plot['actual'], 'b-', label='Actual', marker='o', markersize=2, alpha=0.8)
    ax1.plot(df_plot['step'], df_plot['forecast'], 'r--', label='Forecast', marker='s', markersize=2, alpha=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.set_title(f'{model_name} - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Logarithmic Scale (Y-axis)
    ax2 = axes[0, 1]
    ax2.plot(df_plot['step'], df_plot['actual_log'], 'b-', label='Actual', marker='o', markersize=2, alpha=0.8)
    ax2.plot(df_plot['step'], df_plot['forecast_log'], 'r--', label='Forecast', marker='s', markersize=2, alpha=0.8)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value (Log Scale)')
    ax2.set_yscale('log')
    ax2.set_title(f'{model_name} - Logarithmic Y-Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error Evolution - Linear
    ax3 = axes[1, 0]
    ax3.plot(df_plot['step'], df_plot['absolute_error'], 'g-', label='Absolute Error', marker='^', markersize=2, alpha=0.8)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title(f'{model_name} - Error Evolution (Linear)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error Evolution - Logarithmic
    ax4 = axes[1, 1]
    errors_log = np.where(df_plot['absolute_error'] <= 0, 1e-10, df_plot['absolute_error'])
    ax4.plot(df_plot['step'], errors_log, 'g-', label='Absolute Error', marker='^', markersize=2, alpha=0.8)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Absolute Error (Log Scale)')
    ax4.set_yscale('log')
    ax4.set_title(f'{model_name} - Error Evolution (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{model_name.lower().replace(' ', '_')}_log_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    
    # Statistics
    mae = df['absolute_error'].mean()
    max_error = df['absolute_error'].max()
    min_error = df['absolute_error'].min()
    
    print(f"\n=== {model_name} Statistics ===")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    print(f"Min Absolute Error: {min_error:.6f}")
    
    # Check if log scale is beneficial
    actual_ratio = df['actual'].max() / max(abs(df['actual'].min()), 1e-10)
    error_ratio = max_error / max(min_error, 1e-10) if min_error > 0 else float('inf')
    
    print(f"Data range ratio: {actual_ratio:.2f}")
    print(f"Error range ratio: {error_ratio:.2f}")
    
    if actual_ratio > 100 or error_ratio > 100:
        print("✓ Logarithmic scale is beneficial (high dynamic range)")
    elif actual_ratio > 10 or error_ratio > 10:
        print("~ Logarithmic scale may be helpful (moderate dynamic range)")
    else:
        print("✗ Linear scale is probably sufficient (low dynamic range)")
    
    return {'mae': mae, 'max_error': max_error, 'min_error': min_error, 'actual_ratio': actual_ratio, 'error_ratio': error_ratio}

if __name__ == "__main__":
    # Available files in run_forward directory
    base_dir = "run_forward"
    
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv',
        'TFT': 'tft_walk_forward_results.csv',
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Stats': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    print("Plotting forward walk results with logarithmic scale analysis...")
    print("="*60)
    
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            print(f"\nProcessing {model_name}...")
            result = plot_single_model_log(filepath, model_name)
        else:
            print(f"\nSkipping {model_name} - file not found: {filepath}")
    
    print("\n" + "="*60)
    print("Logarithmic scale analysis complete!")