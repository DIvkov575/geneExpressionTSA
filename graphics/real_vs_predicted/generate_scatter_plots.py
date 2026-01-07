#!/usr/bin/env python3
"""
Generate real vs predicted scatter plots for column 1.
Shows how well models predict actual values with perfect prediction reference.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

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
                    # Remove any NaN or infinite values
                    df = df.dropna()
                    df = df[np.isfinite(df['actual']) & np.isfinite(df['forecast'])]
                    
                    if len(df) > 0:
                        results[model_name] = df
                        print(f"✓ Loaded {model_name}: {len(df)} valid data points")
                    else:
                        print(f"⚠ {model_name}: No valid data after cleaning")
                else:
                    print(f"⚠ {model_name}: Missing required columns")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        else:
            print(f"✗ {model_name}: File not found at {filepath}")
    
    return results

def calculate_metrics(actual, predicted):
    """Calculate comprehensive prediction metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = np.mean(np.abs(actual - predicted))
    metrics['rmse'] = np.sqrt(np.mean((actual - predicted) ** 2))
    metrics['mape'] = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    
    # Correlation metrics
    try:
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(actual, predicted)
        metrics['r2'] = r2_score(actual, predicted)
    except:
        metrics['pearson_r'] = 0
        metrics['pearson_p'] = 1
        metrics['r2'] = -np.inf
    
    # Directional accuracy (percentage of correct directional predictions)
    actual_diff = np.diff(actual)
    predicted_diff = np.diff(predicted)
    if len(actual_diff) > 0:
        metrics['directional_accuracy'] = np.mean(np.sign(actual_diff) == np.sign(predicted_diff)) * 100
    else:
        metrics['directional_accuracy'] = 0
    
    return metrics

def plot_individual_scatter(results, save_dir="column_1"):
    """Create individual scatter plots for each model."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, df in results.items():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        actual = df['actual'].values
        forecast = df['forecast'].values
        
        # Calculate metrics
        metrics = calculate_metrics(actual, forecast)
        
        # Plot 1: Standard scatter plot
        ax1.scatter(actual, forecast, alpha=0.6, s=20, color='blue')
        
        # Perfect prediction line
        min_val = min(actual.min(), forecast.min())
        max_val = max(actual.max(), forecast.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Best fit line
        try:
            coeffs = np.polyfit(actual, forecast, 1)
            line_x = np.array([min_val, max_val])
            line_y = coeffs[0] * line_x + coeffs[1]
            ax1.plot(line_x, line_y, 'g-', alpha=0.7, linewidth=2, label=f'Best Fit (slope={coeffs[0]:.3f})')
        except:
            pass
        
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{model_name} - Actual vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Add metrics text
        metrics_text = f'R² = {metrics["r2"]:.3f}\\nMAE = {metrics["mae"]:.4f}\\nRMSE = {metrics["rmse"]:.4f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Residuals vs predicted
        residuals = actual - forecast
        ax2.scatter(forecast, residuals, alpha=0.6, s=20, color='purple')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals histogram
        ax3.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residuals Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add normality info
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax3.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\\nStd: {std_residual:.4f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 4: Time series comparison (sample)
        if len(df) > 500:
            sample_idx = np.linspace(0, len(df)-1, 200, dtype=int)
            steps_sample = df['step'].iloc[sample_idx]
            actual_sample = actual[sample_idx]
            forecast_sample = forecast[sample_idx]
        else:
            steps_sample = df['step']
            actual_sample = actual
            forecast_sample = forecast
        
        ax4.plot(steps_sample, actual_sample, 'b-', alpha=0.8, linewidth=1.5, label='Actual')
        ax4.plot(steps_sample, forecast_sample, 'r--', alpha=0.8, linewidth=1.5, label='Predicted')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Value')
        ax4.set_title('Time Series Comparison (Sample)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(save_dir, f"{safe_name}_scatter_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")

def plot_combined_scatter(results, save_dir="column_1"):
    """Create combined scatter plots comparing all models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Combined scatter plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Individual model subplots
    for i, (model_name, df) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        actual = df['actual'].values
        forecast = df['forecast'].values
        
        # Subsample for clarity
        if len(df) > 1000:
            sample_idx = np.random.choice(len(df), 500, replace=False)
            actual_sample = actual[sample_idx]
            forecast_sample = forecast[sample_idx]
        else:
            actual_sample = actual
            forecast_sample = forecast
        
        ax.scatter(actual_sample, forecast_sample, alpha=0.6, s=15, color=colors[i])
        
        # Perfect prediction line
        min_val = min(actual_sample.min(), forecast_sample.min())
        max_val = max(actual_sample.max(), forecast_sample.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1)
        
        # Calculate and display metrics
        metrics = calculate_metrics(actual, forecast)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_name}\\nR² = {metrics["r2"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Remove empty subplots
    for i in range(len(results), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "all_models_scatter_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Single combined plot
    plt.figure(figsize=(12, 10))
    
    for i, (model_name, df) in enumerate(results.items()):
        actual = df['actual'].values
        forecast = df['forecast'].values
        
        # Subsample for clarity
        if len(df) > 800:
            sample_idx = np.random.choice(len(df), 400, replace=False)
            actual_sample = actual[sample_idx]
            forecast_sample = forecast[sample_idx]
        else:
            actual_sample = actual
            forecast_sample = forecast
        
        plt.scatter(actual_sample, forecast_sample, alpha=0.6, s=20, 
                   color=colors[i], label=model_name)
    
    # Perfect prediction line
    all_actual = np.concatenate([df['actual'].values for df in results.values()])
    all_forecast = np.concatenate([df['forecast'].values for df in results.values()])
    min_val = min(all_actual.min(), all_forecast.min())
    max_val = max(all_actual.max(), all_forecast.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
             linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Real vs Predicted - All Models Comparison (Column 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "all_models_single_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_metrics_comparison(results, save_dir="column_1"):
    """Create comprehensive metrics comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics for all models
    all_metrics = {}
    for model_name, df in results.items():
        actual = df['actual'].values
        forecast = df['forecast'].values
        all_metrics[model_name] = calculate_metrics(actual, forecast)
    
    # Create metrics comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metric_names = ['r2', 'pearson_r', 'mae', 'rmse', 'mape', 'directional_accuracy']
    metric_labels = ['R² Score', 'Pearson Correlation', 'MAE', 'RMSE', 'MAPE (%)', 'Directional Accuracy (%)']
    
    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        if i >= 6:
            break
            
        ax = axes[i // 3, i % 3]
        
        model_names = list(all_metrics.keys())
        values = [all_metrics[name][metric] for name in model_names]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, values, color=colors, alpha=0.7)
        
        ax.set_ylabel(label)
        ax.set_title(f'Model Comparison - {label}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Special handling for certain metrics
        if metric in ['r2', 'pearson_r']:
            ax.set_ylim(-1, 1)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        elif metric in ['directional_accuracy', 'mape']:
            ax.set_ylim(0, max(100, max(values) * 1.1) if values else 100)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    return all_metrics

def main():
    print("🎯 Generating Real vs Predicted Scatter Plots")
    print("=" * 50)
    
    # Load forecast results
    print("📊 Loading forecast results...")
    results = load_forecast_results()
    
    if not results:
        print("❌ No forecast results found")
        return
    
    print(f"✅ Loaded results for {len(results)} models")
    
    # Generate individual scatter plots
    print("\\n📈 Creating individual scatter plots...")
    plot_individual_scatter(results)
    
    # Generate combined scatter plots
    print("\\n🔄 Creating combined scatter plots...")
    plot_combined_scatter(results)
    
    # Generate metrics comparison
    print("\\n📊 Creating metrics comparison...")
    all_metrics = plot_metrics_comparison(results)
    
    # Print summary table
    print("\\n📋 Model Performance Summary:")
    print("-" * 80)
    print(f"{'Model':<15} {'R²':<8} {'Corr':<8} {'MAE':<8} {'RMSE':<8} {'Dir%':<8}")
    print("-" * 80)
    
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<15} {metrics['r2']:<8.3f} {metrics['pearson_r']:<8.3f} "
              f"{metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} {metrics['directional_accuracy']:<8.1f}")
    
    print("\\n🎉 Real vs predicted scatter plots generated successfully!")

if __name__ == "__main__":
    main()