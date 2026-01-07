import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_forward_results(csv_file, model_name="Model", save_path=None):
    """
    Plot forward walk results showing actual vs forecasted values
    
    Args:
        csv_file: Path to CSV file with columns: step, actual, forecast, absolute_error
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
    """
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
        
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['step'], df['actual'], 'b-', label='Actual', marker='o', markersize=2, alpha=0.7)
    plt.plot(df['step'], df['forecast'], 'r--', label='Forecast', marker='s', markersize=2, alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'{model_name} Forward Walk Results: Actual vs Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    mae = df['absolute_error'].mean()
    max_error = df['absolute_error'].max()
    min_error = df['absolute_error'].min()
    
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    print(f"Min Absolute Error: {min_error:.6f}")
    
    return {'mae': mae, 'max_error': max_error, 'min_error': min_error}

def plot_all_models_comparison():
    """Plot comparison of all models on the same chart"""
    base_dir = "../run_forward"
    
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv',
        'TFT': 'tft_walk_forward_results.csv',
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Stats': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Forecasts (first 200 steps for clarity)
    plt.subplot(2, 2, 1)
    actual_plotted = False
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df_subset = df.head(200)  # First 200 steps for clarity
            
            if not actual_plotted:
                plt.plot(df_subset['step'], df_subset['actual'], 'k-', label='Actual', linewidth=2, alpha=0.8)
                actual_plotted = True
            
            plt.plot(df_subset['step'], df_subset['forecast'], '--', label=f'{model_name}', alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('All Models: Actual vs Forecasts (First 200 Steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error Evolution
    plt.subplot(2, 2, 2)
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df_subset = df.head(200)
            plt.plot(df_subset['step'], df_subset['absolute_error'], label=f'{model_name}', alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Error Evolution (First 200 Steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: MAE Comparison Bar Chart
    plt.subplot(2, 2, 3)
    model_names = []
    mae_values = []
    
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            mae = df['absolute_error'].mean()
            model_names.append(model_name)
            mae_values.append(mae)
    
    bars = plt.bar(model_names, mae_values, alpha=0.7, color=['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    plt.xlabel('Model')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error Comparison')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Error Distribution (Box Plot)
    plt.subplot(2, 2, 4)
    error_data = []
    labels = []
    
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Use first 200 steps and cap errors for visualization
            errors = df.head(200)['absolute_error'].values
            errors = np.clip(errors, 0, 1)  # Cap at 1 for visibility
            error_data.append(errors)
            labels.append(model_name)
    
    plt.boxplot(error_data, labels=labels)
    plt.xlabel('Model')
    plt.ylabel('Absolute Error (Capped at 1)')
    plt.title('Error Distribution (First 200 Steps)')
    plt.xticks(rotation=45)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
    print("Comprehensive comparison plot saved to: all_models_comparison.png")
    plt.show()

def print_summary_statistics():
    """Print summary statistics for all models"""
    base_dir = "../run_forward"
    
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv',
        'TFT': 'tft_walk_forward_results.csv', 
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Stats': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON - GENERATIVE FORECASTING RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'MAE':<15} {'Max Error':<15} {'Min Error':<15} {'Status':<15}")
    print("-"*80)
    
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            mae = df['absolute_error'].mean()
            max_error = df['absolute_error'].max()
            min_error = df['absolute_error'].min()
            
            # Determine status
            if mae < 0.01:
                status = "Good"
            elif mae < 0.1:
                status = "Moderate"
            elif mae < 1.0:
                status = "Poor"
            else:
                status = "Failed"
            
            print(f"{model_name:<15} {mae:<15.6f} {max_error:<15.6f} {min_error:<15.6f} {status:<15}")
        else:
            print(f"{model_name:<15} {'File Not Found':<45}")
    
    print("-"*80)
    print("Status: Good (<0.01), Moderate (<0.1), Poor (<1.0), Failed (≥1.0)")
    print("="*80)

if __name__ == "__main__":
    base_dir = "../run_forward"
    
    # Individual model plots
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv', 
        'TFT': 'tft_walk_forward_results.csv',
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Statsmodels': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    print("Plotting individual model results...")
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        print(f"\nPlotting {model_name} forward walk results...")
        plot_forward_results(
            filepath,
            model_name=model_name,
            save_path=f"{model_name.lower().replace(' ', '_')}_forward_walk_plot.png"
        )
    
    # Comprehensive comparison
    print("\nCreating comprehensive comparison plot...")
    plot_all_models_comparison()
    
    # Summary statistics
    print_summary_statistics()