#!/usr/bin/env python3
"""
Plot residuals for forecasts starting from step 500.
Shows forecast errors (forecasted - actual) for all models.
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
        print("Available files in recursive_forecasts/:")
        if os.path.exists("recursive_forecasts/"):
            for f in os.listdir("recursive_forecasts/"):
                if "from500" in f:
                    print(f"  {f}")
        else:
            print("  Directory doesn't exist!")
        return {}
    
    # Group files by model and find the latest timestamp for each
    model_files = {}
    for file in files:
        filename = os.path.basename(file)
        # Extract model name (everything before "_from500_")
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
            required_cols = ['actual', 'forecasted', 'step', 'start_point']
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

def create_residuals_plot(all_data, failed_models):
    """Create residuals plot for 500+ extrapolation."""
    
    if not all_data:
        print("❌ NO DATA TO PLOT!")
        return
    
    # Set up the plot
    plt.figure(figsize=(16, 8))
    
    # Color scheme for different models
    colors = {
        'naive': '#1f77b4',           # Blue
        'arima_statsmodels': '#ff7f0e',  # Orange  
        'arima_v3': '#2ca02c',        # Green
        'gbm': '#d62728',             # Red
        'nbeats': '#9467bd',          # Purple
        'tft': '#8c564b',             # Brown
    }
    
    # Line styles for variety
    line_styles = {
        'naive': '-',
        'arima_statsmodels': '--', 
        'arima_v3': '-.',
        'gbm': ':',
        'nbeats': '-',
        'tft': '--',
    }
    
    # Plot residuals for each model
    for model_name, df in all_data.items():
        forecasted = df['forecasted'].values
        actual = df['actual'].values
        steps = df['step'].values
        start_point = df['start_point'].iloc[0]  # Get start point from data
        
        # Calculate residuals (forecasted - actual)
        residuals = forecasted - actual
        
        # Plot the residuals (add start_point offset to steps)
        plot_steps = steps + start_point
        
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
        
        color = colors.get(model_name, np.random.rand(3,))  # Random color if not in dict
        linestyle = line_styles.get(model_name, '-')
        
        plt.plot(plot_steps, residuals,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=readable_name,
                alpha=0.7,
                marker='o',
                markersize=3)
        
        # Print residual statistics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"✅ {readable_name}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Customize plot
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Residuals (Forecasted - Actual)', fontsize=12) 
    plt.title('Residuals for 500+ Extrapolation Forecasts', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'graphics/residuals_500plus_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {plot_filename}")
    
    # Show summary
    print(f"\n{'='*60}")
    print("RESIDUALS PLOT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully plotted residuals for: {len(all_data)} models")
    for model_name in all_data.keys():
        print(f"  ✅ {model_name}")
    
    if failed_models:
        print(f"❌ Failed models: {len(failed_models)}")
        for model_name in failed_models:
            print(f"  ❌ {model_name}")
    else:
        print("✅ All available models plotted successfully!")
    
    print(f"📊 Plot dimensions: {plt.gcf().get_size_inches()}")
    print(f"💾 Saved as: {plot_filename}")
    
    # Display the plot
    plt.show()

def main():
    """Main execution function."""
    print("="*60)
    print("500+ EXTRAPOLATION RESIDUALS PLOTTER")
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
    print("\n3. Creating residuals plot...")
    create_residuals_plot(all_data, failed_models)
    
    print(f"\n{'='*60}")
    print("PLOTTING COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()