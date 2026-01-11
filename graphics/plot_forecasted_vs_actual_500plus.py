#!/usr/bin/env python3
"""
Plot forecasted vs actual values for 500+ extrapolation.
Creates scatter plots showing forecast accuracy with proper value pairing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def find_forecast_files():
    """Find the latest forecast CSV files that start from step 500."""
    pattern = "recursive_forecasts/*_from500_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ NO FORECAST FILES FOUND!")
        print(f"Looking for pattern: {pattern}")
        return {}
    
    # Group files by model and find the latest timestamp for each
    model_files = {}
    for file in files:
        filename = os.path.basename(file)
        if "_from500_" in filename:
            model_name = filename.split("_from500_")[0]
            # Only keep the latest file for each model
            if model_name not in model_files or file > model_files[model_name]:
                model_files[model_name] = file
    
    # Print found files
    for model_name, file in model_files.items():
        print(f"Found latest forecast file for {model_name}: {file}")
    
    return model_files

def load_forecast_data(model_files):
    """Load forecast data for all models."""
    all_data = {}
    failed_models = []
    
    for model_name, filepath in model_files.items():
        try:
            df = pd.read_csv(filepath)
            
            # Verify required columns exist
            required_cols = ['actual', 'forecasted']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ FAILED {model_name}: Missing columns {missing_cols}")
                failed_models.append(model_name)
                continue
            
            all_data[model_name] = df
            print(f"✅ Loaded {model_name}: {len(df)} data points")
            
        except Exception as e:
            print(f"❌ FAILED {model_name}: {str(e)}")
            failed_models.append(model_name)
    
    return all_data, failed_models

def create_forecasted_vs_actual_plot(all_data, failed_models):
    """Create superimposed forecasted vs actual scatter plot."""
    
    if not all_data:
        print("❌ NO DATA TO PLOT!")
        return
    
    # Set up single plot for all models with custom aspect ratio
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set custom aspect ratio: stretch x by 1.5, compress y by 0.75
    ax.set_aspect(0.75/1.5)  # This makes x-axis appear 1.5x wider and y-axis 0.75x shorter
    
    # Color scheme for different models
    colors = {
        'naive': '#1f77b4',           # Blue
        'arima_statsmodels': '#ff7f0e',  # Orange  
        'arima_v3': '#2ca02c',        # Green
        'gbm': '#d62728',             # Red
        'nbeats': '#9467bd',          # Purple
        'tft': '#8c564b',             # Brown
    }
    
    # Marker styles for variety
    markers = {
        'naive': 'o',
        'arima_statsmodels': 's',
        'arima_v3': '^',
        'gbm': 'D',
        'nbeats': 'v',
        'tft': 'X',
    }
    
    # Find global min/max for consistent axes
    all_actual = []
    all_forecasted = []
    
    for df in all_data.values():
        all_actual.extend(df['actual'].values)
        all_forecasted.extend(df['forecasted'].values)
    
    min_val = min(min(all_actual), min(all_forecasted))
    max_val = max(max(all_actual), max(all_forecasted))
    
    # Add some padding
    range_padding = (max_val - min_val) * 0.05
    plot_min = min_val - range_padding
    plot_max = max_val + range_padding
    
    # Plot each model on the same plot
    model_stats = {}
    
    for model_name, df in all_data.items():
        # Get the paired values
        actual_values = df['actual'].values
        forecasted_values = df['forecasted'].values
        
        # Create readable label
        readable_name = model_name.replace('_', ' ').title()
        if readable_name == 'Arima Statsmodels':
            readable_name = 'ARIMA (Statsmodels)'
        elif readable_name == 'Arima V3':
            readable_name = 'ARIMA v3'
        elif readable_name == 'Gbm':
            readable_name = 'GBM'
        elif readable_name == 'Nbeats':
            readable_name = 'N-BEATS'
        elif readable_name == 'Tft':
            readable_name = 'TFT'
        
        color = colors.get(model_name, '#666666')
        marker = markers.get(model_name, 'o')
        
        # Create scatter plot for this model
        ax.scatter(actual_values, forecasted_values, 
                   alpha=0.7, color=color, s=80, 
                   marker=marker, edgecolors='white', linewidth=0.5,
                   label=readable_name)
        
        # Calculate statistics
        mae = np.mean(np.abs(forecasted_values - actual_values))
        mse = np.mean((forecasted_values - actual_values)**2)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((actual_values - forecasted_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        model_stats[readable_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r_squared
        }
        
        print(f"✅ Plotted {readable_name}: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r_squared:.3f}")
    
    # Add perfect prediction line (y = x) - adjust for the new x-axis range
    x_center = (plot_min + plot_max) / 2
    x_range = (plot_max - plot_min) / 2  # Shrink by factor of 2
    x_min_new = x_center - x_range/2
    x_max_new = x_center + x_range/2
    
    # Only draw the perfect prediction line within the visible x-axis range
    line_start = max(x_min_new, plot_min)
    line_end = min(x_max_new, plot_max)
    ax.plot([line_start, line_end], [line_start, line_end], 
           'k--', alpha=0.8, linewidth=3, label='Perfect Prediction')
    
    # Set labels and title
    ax.set_xlabel('Actual Values', fontsize=14)
    ax.set_ylabel('Forecasted Values', fontsize=14)
    ax.set_title('Forecasted vs Actual Values (500+ Extrapolation)\nAll Models Superimposed', 
                fontsize=16, fontweight='bold')
    
    # Set axis ranges - shrink x-axis view by factor of 2
    ax.set_xlim(x_min_new, x_max_new)
    ax.set_ylim(plot_min, plot_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'graphics/forecasted_vs_actual_500plus_{timestamp}.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {plot_filename}")
    
    # Show summary
    print(f"\n{'='*60}")
    print("FORECASTED VS ACTUAL PLOT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully plotted: {len(all_data)} models")
    
    # Print model statistics table
    print("\nModel Performance Statistics:")
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<8}")
    print("-" * 55)
    for model, stats in model_stats.items():
        print(f"{model:<20} {stats['MAE']:<12.6f} {stats['RMSE']:<12.6f} {stats['R²']:<8.3f}")
    
    if failed_models:
        print(f"\n❌ Failed models: {len(failed_models)}")
        for model_name in failed_models:
            print(f"  ❌ {model_name}")
    
    print(f"\n📊 Plot dimensions: {fig.get_size_inches()}")
    print(f"💾 Saved as: {plot_filename}")
    
    # Display the plot
    plt.show()

def main():
    """Main execution function."""
    print("="*60)
    print("FORECASTED VS ACTUAL (500+ EXTRAPOLATION) PLOTTER")
    print("="*60)
    
    # Find forecast files
    print("1. Searching for forecast files...")
    model_files = find_forecast_files()
    
    if not model_files:
        print("❌ FAILED: No forecast files found!")
        return
    
    print(f"✅ Found {len(model_files)} model forecast files")
    
    # Load data
    print("\n2. Loading forecast data...")
    all_data, failed_models = load_forecast_data(model_files)
    
    if not all_data:
        print("❌ FAILED: No valid data loaded!")
        return
    
    # Create plot
    print("\n3. Creating forecasted vs actual plot...")
    create_forecasted_vs_actual_plot(all_data, failed_models)
    
    print(f"\n{'='*60}")
    print("PLOTTING COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()