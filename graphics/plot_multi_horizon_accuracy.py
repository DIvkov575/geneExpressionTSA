#!/usr/bin/env python3
"""
Plot multi-horizon accuracy results showing model performance across different forecast horizons.
X-axis: Forecast horizons (1, 3, 5, 10, 20)
Y-axis: Accuracy measures (Accuracy % based on relative error thresholds)
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

def mae_to_accuracy_metrics(df):
    """Convert MAE values to accuracy metrics."""
    horizons = [1, 3, 5, 10, 20]
    
    # Calculate relative accuracy metrics
    # We'll use the typical value range from the data to compute relative accuracy
    
    # Estimate typical data range from the MAE values (MAE gives us scale)
    # For CRE data, values typically range around 0.4-0.6 based on our plots
    typical_range = 0.2  # Estimated from previous plots
    
    accuracy_df = df.copy()
    
    # Convert MAE to accuracy percentage
    # Accuracy % = 100 * (1 - MAE/typical_range), capped at 0%
    for horizon in horizons:
        mae_col = str(horizon)
        accuracy_col = f'accuracy_{horizon}'
        
        # Calculate accuracy as percentage
        # Higher MAE = lower accuracy
        accuracy_df[accuracy_col] = 100 * np.maximum(0, 1 - df[mae_col] / typical_range)
        
        # Also calculate MAPE-style relative error percentage
        mape_col = f'rel_error_{horizon}'
        accuracy_df[mape_col] = 100 * (df[mae_col] / typical_range)
    
    return accuracy_df

def create_accuracy_horizon_plot(df):
    """Create line plot of accuracy vs forecast horizon for all models."""
    
    # Convert MAE to accuracy metrics
    accuracy_df = mae_to_accuracy_metrics(df)
    
    # Set up the plot with subplots for different metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
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
    
    # Plot 1: Accuracy Percentage
    plotted_models = []
    
    for _, row in accuracy_df.iterrows():
        model_name = row['model']
        
        # Skip empty rows
        if pd.isna(model_name) or model_name == '':
            continue
        
        # Get accuracy values for each horizon
        accuracy_values = [row[f'accuracy_{h}'] for h in horizons]
        
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
        
        # Plot the accuracy line
        ax1.plot(horizons, accuracy_values, 
                color=color, marker=marker, linestyle=linestyle,
                linewidth=2.5, markersize=8, markeredgewidth=1.5, 
                markeredgecolor='white', label=readable_name)
        
        plotted_models.append(readable_name)
        
        # Print values for verification
        acc_str = ', '.join([f'H={h}: {acc:.1f}%' for h, acc in zip(horizons, accuracy_values)])
        print(f"✅ Plotted {readable_name} accuracy: {acc_str}")
    
    # Customize the accuracy plot
    ax1.set_xlabel('Forecast Horizon (steps)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Across Forecast Horizons\n(Higher is Better)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Set x-axis to show only the horizon values
    ax1.set_xticks(horizons)
    ax1.set_xticklabels([str(h) for h in horizons])
    
    # Add grid for easier reading
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set y-axis limits for percentage
    ax1.set_ylim(0, 100)
    
    # Add legend
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    # Plot 2: Relative Error Percentage (inverse of accuracy)
    for _, row in accuracy_df.iterrows():
        model_name = row['model']
        
        # Skip empty rows
        if pd.isna(model_name) or model_name == '':
            continue
        
        # Get relative error values for each horizon
        rel_error_values = [row[f'rel_error_{h}'] for h in horizons]
        
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
        
        # Plot the relative error line
        ax2.plot(horizons, rel_error_values, 
                color=color, marker=marker, linestyle=linestyle,
                linewidth=2.5, markersize=8, markeredgewidth=1.5, 
                markeredgecolor='white', label=readable_name)
        
        # Print values for verification
        err_str = ', '.join([f'H={h}: {err:.1f}%' for h, err in zip(horizons, rel_error_values)])
        print(f"✅ Plotted {readable_name} rel. error: {err_str}")
    
    # Customize the relative error plot
    ax2.set_xlabel('Forecast Horizon (steps)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Relative Error Across Forecast Horizons\n(Lower is Better)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Set x-axis to show only the horizon values
    ax2.set_xticks(horizons)
    ax2.set_xticklabels([str(h) for h in horizons])
    
    # Add grid for easier reading
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set y-axis to start from 0
    ax2.set_ylim(bottom=0)
    
    # Add legend
    ax2.legend(fontsize=10, loc='upper left', framealpha=0.9)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'multi_horizon_accuracy_{timestamp}.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {plot_filename}")
    
    # Display summary statistics
    print(f"\n{'='*60}")
    print("MULTI-HORIZON ACCURACY PLOT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully plotted: {len(plotted_models)} models")
    print(f"📊 Horizons evaluated: {horizons}")
    print(f"💾 Saved as: {plot_filename}")
    
    # Find best performing model for each horizon (highest accuracy)
    print(f"\nBest performing model by horizon (accuracy %):")
    for horizon in horizons:
        accuracy_col = f'accuracy_{horizon}'
        horizon_accs = []
        model_names = []
        for _, row in accuracy_df.iterrows():
            if pd.notna(row['model']) and row['model'] != '':
                # Exclude neural models that fell back to naive for fair comparison
                if row['model'] in ['nbeats', 'tft']:
                    continue
                horizon_accs.append(row[accuracy_col])
                model_names.append(row['model'])
        
        if horizon_accs:
            best_idx = np.argmax(horizon_accs)  # Highest accuracy
            best_model = model_names[best_idx]
            best_acc = horizon_accs[best_idx]
            print(f"  H={horizon}: {best_model} (Accuracy: {best_acc:.1f}%)")
    
    # Show the plot
    plt.show()
    
    return fig, accuracy_df

def main():
    """Main execution function."""
    print("="*60)
    print("MULTI-HORIZON ACCURACY PLOTTER")
    print("="*60)
    
    # Load MAE data
    print("1. Loading MAE summary data...")
    df = load_mae_data()
    
    if df is None:
        print("❌ FAILED: Could not load MAE data!")
        return
    
    # Create plot
    print("\n2. Converting MAE to accuracy metrics...")
    print("3. Creating accuracy vs horizon plots...")
    fig, accuracy_df = create_accuracy_horizon_plot(df)
    
    print(f"\n{'='*60}")
    print("PLOTTING COMPLETED")
    print(f"{'='*60}")
    
    return accuracy_df

if __name__ == "__main__":
    accuracy_data = main()