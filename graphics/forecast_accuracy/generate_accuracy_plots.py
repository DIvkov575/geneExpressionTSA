#!/usr/bin/env python3
"""
Generate forecast accuracy vs time plots for column 2.
Shows how forecast error (MAPE) evolves over time for different models.
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
    """Load forecast results for all models focusing on column 2."""
    results = {}
    
    # Model files for column 2 - only include files that exist
    model_files = {
        'GBM': f'{base_dir}/gbm_column_2_results.csv',
        'NBEATS': f'{base_dir}/nbeats_column_2_results.csv',
    }
    
    
    # Load direct column 2 files
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if 'absolute_error' in df.columns and 'actual' in df.columns:
                    # Calculate MAPE
                    df['mape'] = np.abs((df['actual'] - df['forecast']) / df['actual']) * 100
                    df = df.dropna()  # Remove any NaN values
                    results[model_name] = df
                    print(f"✓ Loaded {model_name}: {len(df)} data points")
                else:
                    print(f"⚠ {model_name}: Missing required columns")
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
        
        # Plot MAPE over time
        plt.plot(df['step'], df['mape'], 'b-', alpha=0.7, linewidth=1)
        
        # Add moving average
        window = min(20, len(df) // 10)
        if window > 1:
            moving_avg = df['mape'].rolling(window=window, center=True).mean()
            plt.plot(df['step'], moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window} steps)')
        
        plt.xlabel('Time Step')
        plt.ylabel('MAPE (%)')
        plt.title(f'{model_name} - Forecast Accuracy vs Time (Column 2)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_mape = df['mape'].mean()
        max_mape = df['mape'].max()
        plt.axhline(y=mean_mape, color='orange', linestyle='--', alpha=0.7, label=f'Mean MAPE: {mean_mape:.2f}%')
        
        if window > 1:
            plt.legend()
        
        # Use log scale if MAPE varies widely
        mape_range = max_mape / (df['mape'].min() + 1e-10)
        if mape_range > 100:
            plt.yscale('log')
            plt.ylabel('MAPE (%) - log scale')
        
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
    
    # Superimposed plot - all models on same plot
    plt.figure(figsize=(16, 10))
    
    # Plot all models superimposed with different styles
    for i, (model_name, df) in enumerate(results.items()):
        # Subsample for clarity
        if len(df) > 1000:
            step = len(df) // 500
            df_plot = df.iloc[::step]
        else:
            df_plot = df
            
        # Use different line styles for better distinction
        linestyle = ['-', '--', '-.', ':'][i % 4]
        plt.plot(df_plot['step'], df_plot['mape'], 
                color=colors[i], alpha=0.8, linewidth=2, label=model_name, linestyle=linestyle)
        
        # Add moving average for each model
        window = min(50, len(df_plot) // 10)
        if window > 2:
            moving_avg = df_plot['mape'].rolling(window=window, center=True).mean()
            plt.plot(df_plot['step'], moving_avg, 
                    color=colors[i], alpha=1.0, linewidth=3, linestyle='-')
    
    plt.xlabel('Time Step')
    plt.ylabel('MAPE (%)') 
    plt.title('Forecast Accuracy - All Models Superimposed (Column 2)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "all_models_superimposed.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Also create the original separate plots
    plt.figure(figsize=(15, 8))
    
    for i, (model_name, df) in enumerate(results.items()):
        if len(df) > 1000:
            step = len(df) // 500
            df_plot = df.iloc[::step]
        else:
            df_plot = df
            
        plt.plot(df_plot['step'], df_plot['mape'], 
                color=colors[i], alpha=0.7, linewidth=1.5, label=model_name)
    
    plt.xlabel('Time Step')
    plt.ylabel('MAPE (%)')
    plt.title('Forecast Accuracy Comparison - All Models (Column 2)')
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
            
        # Filter out zero MAPE for log scale
        df_nonzero = df_plot[df_plot['mape'] > 0]
        if len(df_nonzero) > 0:
            plt.plot(df_nonzero['step'], df_nonzero['mape'], 
                    color=colors[i], alpha=0.7, linewidth=1.5, label=model_name)
    
    plt.xlabel('Time Step')
    plt.ylabel('MAPE (%) - log scale')
    plt.title('Forecast Accuracy Comparison - All Models (Column 2, Log Scale)')
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
        # Cap extreme MAPE for visualization
        mape_values = np.clip(df['mape'].values, 0, np.percentile(df['mape'], 95))
        error_data.append(mape_values)
        labels.append(model_name)
    
    ax1.boxplot(error_data, labels=labels)
    ax1.set_ylabel('MAPE (%)')
    ax1.set_title('MAPE Distribution (Capped at 95th Percentile)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative error
    ax2 = axes[0, 1]
    for model_name, df in results.items():
        cumulative_mape = df['mape'].cumsum()
        ax2.plot(df['step'], cumulative_mape, label=model_name, alpha=0.7)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative MAPE (%)')
    ax2.set_title('Cumulative MAPE Accumulation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error rate (moving average of errors)
    ax3 = axes[1, 0]
    window = 50
    for model_name, df in results.items():
        if len(df) > window:
            moving_avg = df['mape'].rolling(window=window).mean()
            ax3.plot(df['step'], moving_avg, label=model_name, alpha=0.7)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Moving Average MAPE (%)')
    ax3.set_title(f'MAPE Trend ({window}-step Moving Average)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison (MAPE by model)
    ax4 = axes[1, 1]
    model_names = list(results.keys())
    mape_values = [results[name]['mape'].mean() for name in model_names]
    
    bars = ax4.bar(model_names, mape_values, alpha=0.7, color=plt.cm.tab10(np.linspace(0, 1, len(model_names))))
    ax4.set_ylabel('Mean MAPE (%)')
    ax4.set_title('Model Performance Comparison (MAPE)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mape_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=9)
    
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
        mean_mape = df['mape'].mean()
        max_mape = df['mape'].max()
        min_mape = df['mape'].min()
        std_mape = df['mape'].std()
        
        print(f"{model_name:<12}: MAPE={mean_mape:.2f}%, Max={max_mape:.2f}%, Min={min_mape:.2f}%, Std={std_mape:.2f}%")
    
    print("\\n🎉 Forecast accuracy plots (MAPE vs Time) generated successfully for Column 2!")

if __name__ == "__main__":
    main()