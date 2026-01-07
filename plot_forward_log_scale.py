import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_forward_results_log(csv_file, model_name="Model", save_path=None, use_log_scale=True):
    """
    Plot forward walk results with logarithmic scale option
    
    Args:
        csv_file: Path to CSV file with columns: step, actual, forecast, absolute_error
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
        use_log_scale: Whether to use logarithmic scale for y-axis
    """
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
        
    df = pd.read_csv(csv_file)
    
    # Handle negative or zero values for log scale
    if use_log_scale:
        # Add small epsilon to avoid log(0) issues
        epsilon = 1e-10
        df['actual_log'] = np.where(df['actual'] <= 0, epsilon, df['actual'])
        df['forecast_log'] = np.where(df['forecast'] <= 0, epsilon, df['forecast'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Linear Scale
    ax1 = axes[0, 0]
    ax1.plot(df['step'], df['actual'], 'b-', label='Actual', marker='o', markersize=2, alpha=0.7)
    ax1.plot(df['step'], df['forecast'], 'r--', label='Forecast', marker='s', markersize=2, alpha=0.7)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.set_title(f'{model_name} - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Logarithmic Scale (Y-axis)
    if use_log_scale:
        ax2 = axes[0, 1]
        ax2.plot(df['step'], df['actual_log'], 'b-', label='Actual', marker='o', markersize=2, alpha=0.7)
        ax2.plot(df['step'], df['forecast_log'], 'r--', label='Forecast', marker='s', markersize=2, alpha=0.7)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Value (Log Scale)')
        ax2.set_yscale('log')
        ax2.set_title(f'{model_name} - Logarithmic Y-Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error Evolution - Linear
    ax3 = axes[1, 0]
    ax3.plot(df['step'], df['absolute_error'], 'g-', label='Absolute Error', marker='^', markersize=2, alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title(f'{model_name} - Error Evolution (Linear)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error Evolution - Logarithmic
    ax4 = axes[1, 1]
    # Handle zero errors for log scale
    errors_log = np.where(df['absolute_error'] <= 0, 1e-10, df['absolute_error'])
    ax4.plot(df['step'], errors_log, 'g-', label='Absolute Error', marker='^', markersize=2, alpha=0.7)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Absolute Error (Log Scale)')
    ax4.set_yscale('log')
    ax4.set_title(f'{model_name} - Error Evolution (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Log scale plot saved to: {save_path}")
    
    plt.show()
    
    # Statistics
    mae = df['absolute_error'].mean()
    max_error = df['absolute_error'].max()
    min_error = df['absolute_error'].min()
    median_error = df['absolute_error'].median()
    
    print(f"\n=== {model_name} Statistics ===")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Median Absolute Error: {median_error:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    print(f"Min Absolute Error: {min_error:.6f}")
    print(f"Error Range: {max_error - min_error:.6f}")
    
    # Check data range for log scale appropriateness
    actual_range = df['actual'].max() - df['actual'].min()
    forecast_range = df['forecast'].max() - df['forecast'].min()
    error_range = max_error - min_error
    
    print(f"\n=== Data Ranges ===")
    print(f"Actual values range: {actual_range:.6f}")
    print(f"Forecast values range: {forecast_range:.6f}")
    print(f"Error range: {error_range:.6f}")
    
    # Determine if log scale is beneficial
    actual_ratio = df['actual'].max() / max(df['actual'].min(), 1e-10) if df['actual'].min() > 0 else float('inf')
    error_ratio = max_error / max(min_error, 1e-10) if min_error > 0 else float('inf')
    
    print(f"\n=== Log Scale Analysis ===")
    print(f"Actual values ratio (max/min): {actual_ratio:.2f}")
    print(f"Error ratio (max/min): {error_ratio:.2f}")
    
    if actual_ratio > 100 or error_ratio > 100:
        print("✓ Logarithmic scale is beneficial for this data (high dynamic range)")
    elif actual_ratio > 10 or error_ratio > 10:
        print("~ Logarithmic scale may be helpful for this data (moderate dynamic range)")
    else:
        print("✗ Linear scale is probably sufficient for this data (low dynamic range)")
    
    return {
        'mae': mae, 
        'median_error': median_error,
        'max_error': max_error, 
        'min_error': min_error,
        'actual_ratio': actual_ratio,
        'error_ratio': error_ratio
    }

def plot_all_models_log_comparison():
    """Plot comparison of all models with logarithmic scaling"""
    base_dir = "run_forward"
    
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv',
        'TFT': 'tft_walk_forward_results.csv',
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Stats': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Colors for models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Linear Scale Forecasts
    ax = axes[0]
    actual_plotted = False
    for i, (model_name, filename) in enumerate(models.items()):
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df_subset = df.head(100)  # First 100 steps for clarity
            
            if not actual_plotted:
                ax.plot(df_subset['step'], df_subset['actual'], 'k-', 
                       label='Actual', linewidth=3, alpha=0.8)
                actual_plotted = True
            
            ax.plot(df_subset['step'], df_subset['forecast'], '--', 
                   label=f'{model_name}', alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('All Models: Linear Scale (First 100 Steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log Scale Forecasts
    ax = axes[1]
    actual_plotted = False
    for i, (model_name, filename) in enumerate(models.items()):
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df_subset = df.head(100)
            
            # Handle log scale
            epsilon = 1e-10
            actual_log = np.where(df_subset['actual'] <= 0, epsilon, df_subset['actual'])
            forecast_log = np.where(df_subset['forecast'] <= 0, epsilon, df_subset['forecast'])
            
            if not actual_plotted:
                ax.plot(df_subset['step'], actual_log, 'k-', 
                       label='Actual', linewidth=3, alpha=0.8)
                actual_plotted = True
            
            ax.plot(df_subset['step'], forecast_log, '--', 
                   label=f'{model_name}', alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value (Log Scale)')
    ax.set_yscale('log')
    ax.set_title('All Models: Log Y-Scale (First 100 Steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error Evolution - Linear
    ax = axes[2]
    for i, (model_name, filename) in enumerate(models.items()):
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df_subset = df.head(100)
            ax.plot(df_subset['step'], df_subset['absolute_error'], 
                   label=f'{model_name}', alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Evolution - Linear Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error Evolution - Log Scale
    ax = axes[3]
    for i, (model_name, filename) in enumerate(models.items()):
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df_subset = df.head(100)
            errors_log = np.where(df_subset['absolute_error'] <= 0, 1e-10, df_subset['absolute_error'])
            ax.plot(df_subset['step'], errors_log, 
                   label=f'{model_name}', alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Absolute Error (Log Scale)')
    ax.set_yscale('log')
    ax.set_title('Error Evolution - Log Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: MAE Comparison - Log Scale
    ax = axes[4]
    model_names = []
    mae_values = []
    
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            mae = df['absolute_error'].mean()
            model_names.append(model_name)
            mae_values.append(mae)
    
    bars = ax.bar(model_names, mae_values, alpha=0.7, color=colors[:len(mae_values)])
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Absolute Error (Log Scale)')
    ax.set_yscale('log')
    ax.set_title('MAE Comparison - Log Scale')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Error Distribution Box Plot - Log Scale
    ax = axes[5]
    error_data = []
    labels = []
    
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            errors = df.head(100)['absolute_error'].values
            errors_log = np.where(errors <= 0, 1e-10, errors)
            error_data.append(errors_log)
            labels.append(model_name)
    
    box_plot = ax.boxplot(error_data, tick_labels=labels)
    ax.set_xlabel('Model')
    ax.set_ylabel('Absolute Error (Log Scale)')
    ax.set_yscale('log')
    ax.set_title('Error Distribution - Log Scale')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('all_models_log_comparison.png', dpi=300, bbox_inches='tight')
    print("Logarithmic scale comparison plot saved to: all_models_log_comparison.png")
    plt.show()

if __name__ == "__main__":
    base_dir = "run_forward"
    
    # Individual model plots with logarithmic scaling
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv', 
        'TFT': 'tft_walk_forward_results.csv',
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Statsmodels': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    print("Plotting individual model results with logarithmic scaling...")
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        print(f"\nPlotting {model_name} forward walk results (log scale)...")
        result = plot_forward_results_log(
            filepath,
            model_name=model_name,
            save_path=f"{model_name.lower().replace(' ', '_')}_log_scale_plot.png",
            use_log_scale=True
        )
    
    # Comprehensive comparison with logarithmic scaling
    print("\nCreating comprehensive logarithmic scale comparison plot...")
    plot_all_models_log_comparison()