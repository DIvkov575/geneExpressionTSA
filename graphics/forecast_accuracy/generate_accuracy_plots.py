#!/usr/bin/env python3
"""
Generate forecast accuracy vs time plots for column 1.
Shows how forecast error evolves over time for different models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_forecast_results(base_dir="../../run_forward"):
    """Load forecast results for all models focusing on column 1."""
    results = {}
    
    # Model files for column 1 - only include files that exist
    model_files = {
        'Naive': f'{base_dir}/naive_column_1_results.csv',
        'ARIMA Stats': f'{base_dir}/arima_statsmodels_column_1_results.csv', 
        'NBEATS': f'{base_dir}/nbeats_column_1_results.csv',
        'TFT': f'{base_dir}/tft_column_1_results.csv',
        'GBM': f'{base_dir}/gbm_column_1_results.csv',
        'ARIMA v3': f'{base_dir}/arima_v3_column_1_results.csv'
    }
    
    # Also check for alternative naming patterns
    alternative_files = {
        'NBEATS (alt)': f'{base_dir}/nbeats_walk_forward_results.csv',
        'TFT (alt)': f'{base_dir}/tft_walk_forward_results.csv',
        'GBM (alt)': f'{base_dir}/gbm_walk_forward_results.csv',
        'ARIMA Stats (alt)': f'{base_dir}/arima_statsmodels_walk_forward_results.csv',
        'Naive (alt)': f'{base_dir}/naive_walk_forward_results.csv',
        'ARIMA v3 (alt)': f'{base_dir}/arima_v3_walk_forward_results.csv'
    }
    
    # Combine both sets
    all_model_files = {**model_files, **alternative_files}
    
    for model_name, filepath in all_model_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if 'absolute_error' in df.columns:
                    results[model_name] = df
                    print(f"✓ Loaded {model_name}: {len(df)} data points")
                else:
                    print(f"⚠ {model_name}: Missing 'absolute_error' column")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        else:
            print(f"✗ {model_name}: File not found at {filepath}")
    
    return results

def plot_individual_accuracy(results, save_dir="individual"):
    """Create individual accuracy plots for each model."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, df in results.items():
        plt.figure(figsize=(12, 6))
        
        # Plot absolute error over time
        plt.plot(df['step'], df['absolute_error'], 'b-', alpha=0.7, linewidth=1)
        
        # Add moving average
        window = min(20, len(df) // 10)
        if window > 1:
            moving_avg = df['absolute_error'].rolling(window=window, center=True).mean()
            plt.plot(df['step'], moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window} steps)')
        
        plt.xlabel('Time Step')
        plt.ylabel('Absolute Error')
        plt.title(f'{model_name} - Forecast Accuracy vs Time (Column 1)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mae = df['absolute_error'].mean()
        max_error = df['absolute_error'].max()
        plt.axhline(y=mae, color='orange', linestyle='--', alpha=0.7, label=f'MAE: {mae:.4f}')
        
        if window > 1:
            plt.legend()
        
        # Use log scale if errors vary widely
        error_range = max_error / (df['absolute_error'].min() + 1e-10)
        if error_range > 100:
            plt.yscale('log')
            plt.ylabel('Absolute Error (log scale)')
        
        plt.tight_layout()
        
        # Save plot
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(save_dir, f"{safe_name}_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")

def plot_combined_accuracy(results, save_dir="combined"):
    """Create combined accuracy plots comparing all models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Combined plot - linear scale
    plt.figure(figsize=(15, 8))
    
    for i, (model_name, df) in enumerate(results.items()):
        # Subsample for clarity if too many points
        if len(df) > 1000:
            step = len(df) // 500
            df_plot = df.iloc[::step]
        else:
            df_plot = df
            
        plt.plot(df_plot['step'], df_plot['absolute_error'], 
                color=colors[i], alpha=0.7, linewidth=1.5, label=model_name)
    
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Forecast Accuracy Comparison - All Models (Column 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "all_models_accuracy_linear.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Combined plot - log scale
    plt.figure(figsize=(15, 8))
    
    for i, (model_name, df) in enumerate(results.items()):
        # Subsample for clarity
        if len(df) > 1000:
            step = len(df) // 500
            df_plot = df.iloc[::step]
        else:
            df_plot = df
            
        # Filter out zero errors for log scale
        df_nonzero = df_plot[df_plot['absolute_error'] > 0]
        if len(df_nonzero) > 0:
            plt.plot(df_nonzero['step'], df_nonzero['absolute_error'], 
                    color=colors[i], alpha=0.7, linewidth=1.5, label=model_name)
    
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Forecast Accuracy Comparison - All Models (Column 1, Log Scale)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "all_models_accuracy_log.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_error_evolution_analysis(results, save_dir="combined"):
    """Create detailed error evolution analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Error distribution (box plot)
    ax1 = axes[0, 0]
    error_data = []
    labels = []
    
    for model_name, df in results.items():
        # Cap extreme errors for visualization
        errors = np.clip(df['absolute_error'].values, 0, np.percentile(df['absolute_error'], 95))
        error_data.append(errors)
        labels.append(model_name)
    
    ax1.boxplot(error_data, labels=labels)
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Error Distribution (Capped at 95th Percentile)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative error
    ax2 = axes[0, 1]
    for model_name, df in results.items():
        cumulative_error = df['absolute_error'].cumsum()
        ax2.plot(df['step'], cumulative_error, label=model_name, alpha=0.7)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Absolute Error')
    ax2.set_title('Cumulative Error Accumulation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error rate (moving average of errors)
    ax3 = axes[1, 0]
    window = 50
    for model_name, df in results.items():
        if len(df) > window:
            moving_avg = df['absolute_error'].rolling(window=window).mean()
            ax3.plot(df['step'], moving_avg, label=model_name, alpha=0.7)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Moving Average Error')
    ax3.set_title(f'Error Trend ({window}-step Moving Average)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison (MAE by model)
    ax4 = axes[1, 1]
    model_names = list(results.keys())
    mae_values = [results[name]['absolute_error'].mean() for name in model_names]
    
    bars = ax4.bar(model_names, mae_values, alpha=0.7, color=plt.cm.tab10(np.linspace(0, 1, len(model_names))))
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Model Performance Comparison (MAE)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "error_evolution_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    print("🎯 Generating Forecast Accuracy vs Time Plots")
    print("=" * 50)
    
    # Load forecast results
    print("📊 Loading forecast results...")
    results = load_forecast_results()
    
    if not results:
        print("❌ No forecast results found")
        return
    
    print(f"✅ Loaded results for {len(results)} models")
    
    # Generate individual plots
    print("\\n📈 Creating individual accuracy plots...")
    plot_individual_accuracy(results)
    
    # Generate combined plots
    print("\\n📊 Creating combined accuracy plots...")
    plot_combined_accuracy(results)
    
    # Generate detailed analysis
    print("\\n🔍 Creating error evolution analysis...")
    plot_error_evolution_analysis(results)
    
    # Print summary statistics
    print("\\n📋 Summary Statistics:")
    print("-" * 50)
    for model_name, df in results.items():
        mae = df['absolute_error'].mean()
        max_error = df['absolute_error'].max()
        min_error = df['absolute_error'].min()
        std_error = df['absolute_error'].std()
        
        print(f"{model_name:<12}: MAE={mae:.4f}, Max={max_error:.4f}, Min={min_error:.4f}, Std={std_error:.4f}")
    
    print("\\n🎉 Forecast accuracy plots generated successfully!")

if __name__ == "__main__":
    main()