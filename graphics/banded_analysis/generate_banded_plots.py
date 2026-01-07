#!/usr/bin/env python3
"""
Generate banded real vs predicted plots for column 1.
Splits time axis into bands and stacks predictions to show temporal patterns.
Aggregates predictions from all models with color coding.
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

def create_time_bands(results, num_bands=10):
    """Split time axis into bands and organize data accordingly."""
    # Find common time range across all models
    min_step = min(df['step'].min() for df in results.values())
    max_step = max(df['step'].max() for df in results.values())
    total_steps = max_step - min_step + 1
    
    band_size = total_steps // num_bands
    bands = []
    
    print(f"📊 Creating {num_bands} time bands:")
    print(f"   Time range: {min_step} to {max_step} ({total_steps} steps)")
    print(f"   Band size: ~{band_size} steps per band")
    
    for band_idx in range(num_bands):
        band_start = min_step + band_idx * band_size
        band_end = min_step + (band_idx + 1) * band_size - 1
        if band_idx == num_bands - 1:  # Last band includes remainder
            band_end = max_step
        
        band_data = {}
        for model_name, df in results.items():
            # Filter data for this time band
            band_df = df[(df['step'] >= band_start) & (df['step'] <= band_end)]
            if len(band_df) > 0:
                band_data[model_name] = band_df
        
        if band_data:  # Only include bands with data
            bands.append({
                'band_idx': band_idx,
                'start_step': band_start,
                'end_step': band_end,
                'data': band_data
            })
            print(f"   Band {band_idx}: Steps {band_start}-{band_end} ({len(band_data)} models)")
    
    return bands

def plot_banded_scatter(bands, save_dir="column_1_bands"):
    """Create banded scatter plot with all models."""
    os.makedirs(save_dir, exist_ok=True)
    
    num_bands = len(bands)
    if num_bands == 0:
        print("❌ No bands with data found")
        return
    
    # Create figure with subplots for each band
    fig, axes = plt.subplots(2, (num_bands + 1) // 2, figsize=(20, 12))
    if num_bands == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_bands > 2 else axes
    
    # Color map for models
    all_models = set()
    for band in bands:
        all_models.update(band['data'].keys())
    all_models = sorted(list(all_models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
    model_colors = dict(zip(all_models, colors))
    
    for band_idx, band in enumerate(bands):
        if band_idx >= len(axes):
            break
            
        ax = axes[band_idx]
        
        # Plot each model in this band
        for model_name, df in band['data'].items():
            actual = df['actual'].values
            forecast = df['forecast'].values
            
            # Subsample if too many points
            if len(df) > 200:
                sample_idx = np.random.choice(len(df), 100, replace=False)
                actual_sample = actual[sample_idx]
                forecast_sample = forecast[sample_idx]
            else:
                actual_sample = actual
                forecast_sample = forecast
            
            ax.scatter(actual_sample, forecast_sample, 
                      color=model_colors[model_name], alpha=0.6, s=20, 
                      label=model_name if band_idx == 0 else "")
        
        # Perfect prediction line
        if band['data']:
            all_actual = np.concatenate([df['actual'].values for df in band['data'].values()])
            all_forecast = np.concatenate([df['forecast'].values for df in band['data'].values()])
            min_val = min(all_actual.min(), all_forecast.min())
            max_val = max(all_actual.max(), all_forecast.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Band {band_idx}: Steps {band["start_step"]}-{band["end_step"]}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add band statistics
        if band['data']:
            band_mae = np.mean([np.mean(df['absolute_error']) for df in band['data'].values()])
            ax.text(0.05, 0.95, f'Avg MAE: {band_mae:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(bands), len(axes)):
        if i < len(axes):
            fig.delaxes(axes[i])
    
    # Add legend
    if len(bands) > 0:
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "banded_scatter_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_stacked_bands(bands, save_dir="column_1_bands"):
    """Create vertically stacked bands plot."""
    os.makedirs(save_dir, exist_ok=True)
    
    if not bands:
        return
    
    # Create figure with vertical stacking
    fig, axes = plt.subplots(len(bands), 1, figsize=(16, 3 * len(bands)))
    if len(bands) == 1:
        axes = [axes]
    
    # Color map for models
    all_models = set()
    for band in bands:
        all_models.update(band['data'].keys())
    all_models = sorted(list(all_models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
    model_colors = dict(zip(all_models, colors))
    
    for band_idx, band in enumerate(bands):
        ax = axes[band_idx]
        
        # Normalize positions within band for stacking
        y_positions = {}
        y_offset = 0
        
        for model_idx, (model_name, df) in enumerate(band['data'].items()):
            actual = df['actual'].values
            forecast = df['forecast'].values
            
            # Create y-positions for stacking
            num_points = len(actual)
            y_pos = np.full(num_points, y_offset)
            
            # Plot as a scatter with fixed y-position
            ax.scatter(actual, y_pos, color=model_colors[model_name], 
                      alpha=0.4, s=15, marker='o', label=f'{model_name} Actual')
            ax.scatter(forecast, y_pos + 0.1, color=model_colors[model_name], 
                      alpha=0.4, s=15, marker='x', label=f'{model_name} Predicted' if band_idx == 0 else "")
            
            # Connect actual and predicted with lines
            for i in range(0, len(actual), max(1, len(actual) // 20)):  # Subsample lines
                ax.plot([actual[i], forecast[i]], [y_pos[i], y_pos[i] + 0.1], 
                       color=model_colors[model_name], alpha=0.2, linewidth=0.5)
            
            y_offset += 0.3
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Models (stacked)')
        ax.set_title(f'Band {band_idx}: Steps {band["start_step"]}-{band["end_step"]}')
        ax.set_ylim(-0.1, y_offset)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Remove y-tick labels (they're just model positions)
        ax.set_yticks([])
        
        # Add model labels on y-axis
        for model_idx, model_name in enumerate(band['data'].keys()):
            ax.text(-0.1, model_idx * 0.3 + 0.05, model_name, 
                   transform=ax.get_yaxis_transform(), 
                   verticalalignment='center', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "stacked_bands_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_band_performance_evolution(bands, save_dir="column_1_bands"):
    """Plot how model performance evolves across time bands."""
    os.makedirs(save_dir, exist_ok=True)
    
    if not bands:
        return
    
    # Calculate performance metrics for each band and model
    performance_data = {}
    band_centers = []
    
    for band in bands:
        band_center = (band['start_step'] + band['end_step']) / 2
        band_centers.append(band_center)
        
        for model_name, df in band['data'].items():
            if model_name not in performance_data:
                performance_data[model_name] = {'mae': [], 'r2': [], 'band_centers': []}
            
            mae = df['absolute_error'].mean()
            
            # Calculate R²
            actual = df['actual'].values
            forecast = df['forecast'].values
            ss_res = np.sum((actual - forecast) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            performance_data[model_name]['mae'].append(mae)
            performance_data[model_name]['r2'].append(r2)
            performance_data[model_name]['band_centers'].append(band_center)
    
    # Create performance evolution plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(performance_data)))
    
    # Plot 1: MAE evolution
    for i, (model_name, data) in enumerate(performance_data.items()):
        ax1.plot(data['band_centers'], data['mae'], 'o-', color=colors[i], 
                linewidth=2, markersize=6, label=model_name, alpha=0.8)
    
    ax1.set_xlabel('Time (Band Center)')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Model Performance Evolution - MAE Across Time Bands')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: R² evolution
    for i, (model_name, data) in enumerate(performance_data.items()):
        ax2.plot(data['band_centers'], data['r2'], 'o-', color=colors[i], 
                linewidth=2, markersize=6, label=model_name, alpha=0.8)
    
    ax2.set_xlabel('Time (Band Center)')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Performance Evolution - R² Across Time Bands')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "band_performance_evolution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    return performance_data

def main():
    print("🎭 Generating Banded Real vs Predicted Analysis")
    print("=" * 50)
    
    # Load forecast results
    print("📊 Loading forecast results...")
    results = load_forecast_results()
    
    if not results:
        print("❌ No forecast results found")
        return
    
    print(f"✅ Loaded results for {len(results)} models")
    
    # Create time bands
    print("\\n⏰ Creating time bands...")
    bands = create_time_bands(results, num_bands=8)
    
    if not bands:
        print("❌ No time bands created")
        return
    
    print(f"✅ Created {len(bands)} time bands")
    
    # Generate banded scatter plots
    print("\\n📈 Creating banded scatter plots...")
    plot_banded_scatter(bands)
    
    # Generate stacked bands plot
    print("\\n📊 Creating stacked bands analysis...")
    plot_stacked_bands(bands)
    
    # Generate performance evolution
    print("\\n📈 Creating performance evolution analysis...")
    performance_data = plot_band_performance_evolution(bands)
    
    # Print summary
    print("\\n📋 Band Analysis Summary:")
    print("-" * 60)
    for band_idx, band in enumerate(bands):
        print(f"Band {band_idx}: Steps {band['start_step']}-{band['end_step']} ({len(band['data'])} models)")
        
        # Calculate average MAE for this band
        if band['data']:
            band_maes = [np.mean(df['absolute_error']) for df in band['data'].values()]
            avg_mae = np.mean(band_maes)
            print(f"           Average MAE: {avg_mae:.4f}")
        
    print("\\n🎉 Banded analysis plots generated successfully!")

if __name__ == "__main__":
    main()