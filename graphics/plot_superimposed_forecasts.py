#!/usr/bin/env python3
"""
Plot all model forecasts superimposed on a single plot.
Shows recursive forecasts from the beginning of column 2 for ALL models.
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
    """Find all full column 2 forecast CSV files."""
    pattern = "recursive_forecasts/*_full_column2_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ NO FORECAST FILES FOUND!")
        print(f"Looking for pattern: {pattern}")
        print("Available files in recursive_forecasts/:")
        if os.path.exists("recursive_forecasts/"):
            for f in os.listdir("recursive_forecasts/"):
                print(f"  {f}")
        else:
            print("  Directory doesn't exist!")
        return {}
    
    # Extract model names from filenames
    model_files = {}
    for file in files:
        filename = os.path.basename(file)
        # Extract model name (everything before "_full_column2_")
        model_name = filename.split("_full_column2_")[0]
        model_files[model_name] = file
        print(f"Found forecast file for {model_name}: {file}")
    
    return model_files

def load_forecast_data(model_files):
    """Load forecast data for all models."""
    all_data = {}
    failed_models = []
    
    for model_name, filepath in model_files.items():
        try:
            df = pd.read_csv(filepath)
            
            # Verify required columns exist
            required_cols = ['actual', 'forecasted', 'step', 'is_forecast']
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

def create_superimposed_plot(all_data, failed_models):
    """Create the superimposed forecast plot."""
    
    if not all_data:
        print("❌ NO DATA TO PLOT!")
        return
    
    # Set up the plot
    plt.figure(figsize=(16, 10))
    
    # Color scheme for different models
    colors = {
        'naive': '#1f77b4',           # Blue
        'arima_statsmodels': '#ff7f0e',  # Orange  
        'arima_v3': '#2ca02c',        # Green
        'gbm': '#d62728',             # Red
        'nbeats': '#9467bd',          # Purple
        'tft': '#8c564b',             # Brown
        'actual': '#000000'           # Black for actual data
    }
    
    # Line styles for variety
    line_styles = {
        'naive': '-',
        'arima_statsmodels': '--', 
        'arima_v3': '-.',
        'gbm': ':',
        'nbeats': '-',
        'tft': '--',
        'actual': '-'
    }
    
    # Plot actual data first (use any model's data since actual should be the same)
    first_model = list(all_data.keys())[0]
    actual_data = all_data[first_model]['actual'].values
    steps = all_data[first_model]['step'].values
    
    plt.plot(steps, actual_data, 
            color=colors.get('actual', '#000000'),
            linestyle=line_styles.get('actual', '-'),
            linewidth=2.5, 
            label='Actual Data', 
            alpha=0.8)
    
    # Plot each model's forecasts
    for model_name, df in all_data.items():
        forecasted = df['forecasted'].values
        steps = df['step'].values
        is_forecast = df['is_forecast'].values
        
        # Find where forecasting starts (first True in is_forecast)
        forecast_start_idx = np.where(is_forecast)[0]
        if len(forecast_start_idx) == 0:
            print(f"❌ WARNING: No forecast data found for {model_name}")
            continue
        
        forecast_start = forecast_start_idx[0]
        
        # Plot the forecasted portion
        forecast_steps = steps[forecast_start:]
        forecast_values = forecasted[forecast_start:]
        
        color = colors.get(model_name, np.random.rand(3,))  # Random color if not in dict
        linestyle = line_styles.get(model_name, '-')
        
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
        
        plt.plot(forecast_steps, forecast_values,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=readable_name,
                alpha=0.7)
        
        print(f"✅ Plotted {model_name}: {len(forecast_values)} forecast points starting from step {forecast_start}")
    
    
    # Customize plot
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Column 2 Value', fontsize=12) 
    plt.title('Superimposed Recursive Forecasts - All Models\n(Column 2 Full Series)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'graphics/recursive_forecasts_comparison_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {plot_filename}")
    
    # Show summary
    print(f"\n{'='*60}")
    print("SUPERIMPOSED FORECAST PLOT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully plotted models: {len(all_data)}")
    for model_name in all_data.keys():
        print(f"  ✅ {model_name}")
    
    if failed_models:
        print(f"❌ Failed models: {len(failed_models)}")
        for model_name in failed_models:
            print(f"  ❌ {model_name}")
    else:
        print("✅ All models plotted successfully!")
    
    print(f"📊 Plot dimensions: {plt.gcf().get_size_inches()}")
    print(f"💾 Saved as: {plot_filename}")
    
    # Display the plot
    plt.show()

def main():
    """Main execution function."""
    print("="*60)
    print("SUPERIMPOSED RECURSIVE FORECASTS PLOTTER")
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
    print("\n3. Creating superimposed plot...")
    create_superimposed_plot(all_data, failed_models)
    
    print(f"\n{'='*60}")
    print("PLOTTING COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()