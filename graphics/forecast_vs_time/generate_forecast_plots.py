#!/usr/bin/env python3
"""
Generate forecast vs time plots for column 1.
Shows actual vs predicted time series for different models.
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
    
    # Model files for column 1
    model_files = {
        'Naive': f'{base_dir}/naive_column_1_results.csv',
        'ARIMA Stats': f'{base_dir}/arima_statsmodels_column_1_results.csv',
        'NBEATS': f'{base_dir}/nbeats_column_1_results.csv',
        'TFT': f'{base_dir}/tft_column_1_results.csv',
        'GBM': f'{base_dir}/gbm_column_1_results.csv',
        'ARIMA v3': f'{base_dir}/arima_v3_column_1_results.csv'
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                required_cols = ['step', 'actual', 'forecast', 'absolute_error']
                if all(col in df.columns for col in required_cols):
                    results[model_name] = df
                    print(f"✓ Loaded {model_name}: {len(df)} data points")
                else:
                    print(f"⚠ {model_name}: Missing required columns")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        else:
            print(f"✗ {model_name}: File not found at {filepath}")
    
    return results

def plot_individual_forecasts(results, save_dir="column_1"):
    """Create individual forecast vs time plots for each model."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, df in results.items():
        plt.figure(figsize=(14, 8))
        
        # Plot actual vs forecast
        plt.plot(df['step'], df['actual'], 'b-', label='Actual', linewidth=2, alpha=0.8)
        plt.plot(df['step'], df['forecast'], 'r--', label='Forecast', linewidth=1.5, alpha=0.7)
        
        # Add error shading
        error = df['absolute_error']
        plt.fill_between(df['step'], df['forecast'] - error, df['forecast'] + error, 
                        color='red', alpha=0.2, label='Error Band')
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'{model_name} - Forecast vs Time (Column 1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        mae = df['absolute_error'].mean()
        rmse = np.sqrt((df['absolute_error'] ** 2).mean())
        corr = np.corrcoef(df['actual'], df['forecast'])[0, 1]
        
        stats_text = f'MAE: {mae:.4f}\\nRMSE: {rmse:.4f}\\nCorr: {corr:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(save_dir, f"{safe_name}_forecast_vs_time.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")

def plot_combined_forecasts(results, save_dir="column_1"):
    """Create combined forecast plots showing all models together."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Combined plot - all forecasts
    plt.figure(figsize=(16, 10))
    
    # Get actual values from first model (should be same for all)
    first_model = list(results.values())[0]
    plt.plot(first_model['step'], first_model['actual'], 'k-', 
            label='Actual', linewidth=2, alpha=0.9, zorder=10)
    
    # Plot all forecasts
    for i, (model_name, df) in enumerate(results.items()):
        plt.plot(df['step'], df['forecast'], '--', color=colors[i], 
                label=f'{model_name}', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Forecast vs Time Comparison - All Models (Column 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "all_models_forecast_vs_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Combined plot - log scale for values
    plt.figure(figsize=(16, 10))
    
    # Get actual values from first model (should be same for all)
    first_model = list(results.values())[0]
    plt.plot(first_model['step'], np.abs(first_model['actual']) + 1e-10, 'k-', 
            label='Actual (abs)', linewidth=2, alpha=0.9, zorder=10)
    
    # Plot all forecasts with absolute values for log scale
    for i, (model_name, df) in enumerate(results.items()):
        plt.plot(df['step'], np.abs(df['forecast']) + 1e-10, '--', color=colors[i], 
                label=f'{model_name} (abs)', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('|Value| (log scale)')
    plt.yscale('log')
    plt.title('Forecast vs Time Comparison - All Models (Column 1, Log Scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "all_models_forecast_vs_time_log.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_forecast_phases(results, save_dir="column_1"):
    """Create plots showing different phases of forecasting."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Find common time range
    min_steps = min(len(df) for df in results.values())
    phases = [
        ("Early Phase", 0, min(200, min_steps // 3)),
        ("Middle Phase", min_steps // 3, 2 * min_steps // 3),
        ("Late Phase", 2 * min_steps // 3, min_steps)
    ]
    
    for phase_name, start_idx, end_idx in phases:
        plt.figure(figsize=(14, 8))
        
        # Get actual values from first model
        first_model = list(results.values())[0]
        phase_steps = first_model['step'].iloc[start_idx:end_idx]
        phase_actual = first_model['actual'].iloc[start_idx:end_idx]
        
        plt.plot(phase_steps, phase_actual, 'k-', 
                label='Actual', linewidth=2, alpha=0.9)
        
        # Plot forecasts for this phase
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for i, (model_name, df) in enumerate(results.items()):
            phase_forecast = df['forecast'].iloc[start_idx:end_idx]
            plt.plot(phase_steps, phase_forecast, '--', color=colors[i], 
                    label=f'{model_name}', linewidth=1.5, alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Forecast vs Time - {phase_name} (Steps {phase_steps.iloc[0]}-{phase_steps.iloc[-1]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        phase_safe = phase_name.lower().replace(' ', '_')
        save_path = os.path.join(save_dir, f"forecast_vs_time_{phase_safe}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")

def plot_residual_analysis(results, save_dir="column_1"):
    """Create residual analysis plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Plot 1: Residuals over time
    ax1 = axes[0, 0]
    for i, (model_name, df) in enumerate(results.items()):
        residuals = df['actual'] - df['forecast']
        # Subsample for clarity
        if len(df) > 500:
            step = len(df) // 250
            df_plot = df.iloc[::step]
            residuals_plot = residuals.iloc[::step]
        else:
            df_plot = df
            residuals_plot = residuals
            
        ax1.plot(df_plot['step'], residuals_plot, color=colors[i], 
                alpha=0.7, linewidth=1, label=model_name)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Residual (Actual - Forecast)')
    ax1.set_title('Residuals Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual distribution
    ax2 = axes[0, 1]
    residual_data = []
    labels = []
    
    for model_name, df in results.items():
        residuals = df['actual'] - df['forecast']
        # Cap extreme residuals for visualization
        residuals_capped = np.clip(residuals, np.percentile(residuals, 5), np.percentile(residuals, 95))
        residual_data.append(residuals_capped)
        labels.append(model_name)
    
    ax2.boxplot(residual_data, labels=labels)
    ax2.set_ylabel('Residual (Actual - Forecast)')
    ax2.set_title('Residual Distribution')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Actual vs Forecast scatter
    ax3 = axes[1, 0]
    for i, (model_name, df) in enumerate(results.items()):
        # Subsample for clarity
        if len(df) > 1000:
            df_sample = df.sample(n=500, random_state=42)
        else:
            df_sample = df
            
        ax3.scatter(df_sample['actual'], df_sample['forecast'], 
                   color=colors[i], alpha=0.6, s=20, label=model_name)
    
    # Perfect prediction line
    all_values = np.concatenate([df['actual'].values for df in results.values()])
    min_val, max_val = all_values.min(), all_values.max()
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
    
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Forecast')
    ax3.set_title('Actual vs Forecast')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Prediction interval coverage (simplified)
    ax4 = axes[1, 1]
    coverage_data = []
    model_names = list(results.keys())
    
    for model_name, df in results.items():
        # Simple coverage: percentage of predictions within 1 std of actual mean
        actual_mean = df['actual'].mean()
        actual_std = df['actual'].std()
        within_range = ((df['forecast'] >= actual_mean - actual_std) & 
                       (df['forecast'] <= actual_mean + actual_std)).mean()
        coverage_data.append(within_range * 100)
    
    bars = ax4.bar(model_names, coverage_data, color=colors, alpha=0.7)
    ax4.set_ylabel('Coverage (%)')
    ax4.set_title('Prediction Coverage (within 1σ of actual mean)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, coverage_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "residual_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    print("📊 Generating Forecast vs Time Plots")
    print("=" * 40)
    
    # Load forecast results
    print("📈 Loading forecast results...")
    results = load_forecast_results()
    
    if not results:
        print("❌ No forecast results found")
        return
    
    print(f"✅ Loaded results for {len(results)} models")
    
    # Generate individual plots
    print("\\n📊 Creating individual forecast plots...")
    plot_individual_forecasts(results)
    
    # Generate combined plots
    print("\\n🔄 Creating combined forecast plots...")
    plot_combined_forecasts(results)
    
    # Generate phase analysis
    print("\\n⏱ Creating forecast phase analysis...")
    plot_forecast_phases(results)
    
    # Generate residual analysis
    print("\\n🔍 Creating residual analysis...")
    plot_residual_analysis(results)
    
    print("\\n🎉 Forecast vs time plots generated successfully!")

if __name__ == "__main__":
    main()