#!/usr/bin/env python3
"""
Plot multi-horizon accuracy results showing model performance across different forecast horizons.
X-axis: Forecast horizons (1, 3, 5, 10, 20)
Y-axis: Accuracy measures (R², MAPE, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def load_mae_data():
    """Load the MAE summary data from horizon experiments."""
    # Find the most recent MAE summary file
    mae_file = '../horizon_experiments/mae_summary_20260108_022446.csv'
    
    if not os.path.exists(mae_file):
        print(f"❌ MAE summary file not found: {mae_file}")
        return None
    
    try:
        df = pd.read_csv(mae_file)
        print(f"✅ Loaded MAE data from: {mae_file}")
        print(f"📊 Data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Failed to load MAE data: {e}")
        return None

def create_mae_horizon_plot(df):
    """Create line plot of MAE vs forecast horizon for all models."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define horizons
    horizons = [1, 3, 5, 10, 20]
    
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
    
    # Line styles - use dashed for neural models that failed back to naive
    line_styles = {
        'naive': '-',
        'arima_statsmodels': '-',
        'arima_v3': '-',
        'gbm': '-',
        'nbeats': '--',  # Dashed because it fell back to naive
        'tft': '--',     # Dashed because it fell back to naive
    }
    
    # Plot each model
    plotted_models = []
    
    for _, row in df.iterrows():
        model_name = row['model']
        
        # Skip empty rows
        if pd.isna(model_name) or model_name == '':
            continue
        
        # Get MAE values for each horizon
        mae_values = [row[str(h)] for h in horizons]
        
        # Create readable label
        if model_name == 'arima_statsmodels':
            readable_name = 'ARIMA (Statsmodels)'
        elif model_name == 'arima_v3':
            readable_name = 'ARIMA v3'
        elif model_name == 'gbm':
            readable_name = 'GBM'
        elif model_name == 'naive':
            readable_name = 'Naive'
        elif model_name == 'nbeats':
            readable_name = 'N-BEATS (→ Naive)'  # Indicate fallback
        elif model_name == 'tft':
            readable_name = 'TFT (→ Naive)'      # Indicate fallback
        else:
            readable_name = model_name.replace('_', ' ').title()
        
        # Get styling
        color = colors.get(model_name, '#666666')
        marker = markers.get(model_name, 'o')
        linestyle = line_styles.get(model_name, '-')
        
        # Plot the line
        ax.plot(horizons, mae_values, 
                color=color, marker=marker, linestyle=linestyle,
                linewidth=2.5, markersize=8, markeredgewidth=1.5, 
                markeredgecolor='white', label=readable_name)
        
        plotted_models.append(readable_name)
        
        # Print values for verification
        mae_str = ', '.join([f'H={h}: {mae:.6f}' for h, mae in zip(horizons, mae_values)])
        print(f"✅ Plotted {readable_name}: {mae_str}")
    
    # Customize the plot
    ax.set_xlabel('Forecast Horizon (steps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Across Forecast Horizons\nMulti-Horizon MAE Evaluation', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis to show only the horizon values
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    
    # Add grid for easier reading
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(bottom=0)
    
    # Add legend with better positioning
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'multi_horizon_mae_{timestamp}.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {plot_filename}")
    
    # Display summary statistics
    print(f"\n{'='*60}")
    print("MULTI-HORIZON MAE PLOT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully plotted: {len(plotted_models)} models")
    print(f"📊 Horizons evaluated: {horizons}")
    print(f"💾 Saved as: {plot_filename}")
    
    # Find best performing model for each horizon
    print(f"\nBest performing model by horizon:")
    for i, horizon in enumerate(horizons):
        horizon_maes = []
        model_names = []
        for _, row in df.iterrows():
            if pd.notna(row['model']) and row['model'] != '':
                # Exclude neural models that fell back to naive for fair comparison
                if row['model'] in ['nbeats', 'tft']:
                    continue
                horizon_maes.append(row[str(horizon)])
                model_names.append(row['model'])
        
        if horizon_maes:
            best_idx = np.argmin(horizon_maes)
            best_model = model_names[best_idx]
            best_mae = horizon_maes[best_idx]
            print(f"  H={horizon}: {best_model} (MAE: {best_mae:.6f})")
    
    # Show the plot
    plt.show()
    
    return fig

def main():
    """Main execution function."""
    print("="*60)
    print("MULTI-HORIZON MAE PLOTTER")
    print("="*60)
    
    # Load MAE data
    print("1. Loading MAE summary data...")
    df = load_mae_data()
    
    if df is None:
        print("❌ FAILED: Could not load MAE data!")
        return
    
    # Create plot
    print("\n2. Creating MAE vs horizon plot...")
    fig = create_mae_horizon_plot(df)
    
    print(f"\n{'='*60}")
    print("PLOTTING COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()