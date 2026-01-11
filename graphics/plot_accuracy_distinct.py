#!/usr/bin/env python3
"""
Create a distinct accuracy plot with better scaling to distinguish between series.
Also generates accuracy table for analysis.
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
    # For better scaling, we'll use a different approach
    # Use the data range based on actual values we've seen in plots (0.4-0.6 range)
    typical_range = 0.2  # Estimated from previous CRE plots
    
    accuracy_df = df.copy()
    
    # Convert MAE to accuracy percentage
    for horizon in horizons:
        mae_col = str(horizon)
        accuracy_col = f'accuracy_{horizon}'
        
        # Calculate accuracy as percentage
        # Higher MAE = lower accuracy
        accuracy_df[accuracy_col] = 100 * np.maximum(0, 1 - df[mae_col] / typical_range)
    
    return accuracy_df

def create_accuracy_table(accuracy_df):
    """Create and display accuracy table."""
    horizons = [1, 3, 5, 10, 20]
    
    # Create a clean table for display
    table_data = []
    
    for _, row in accuracy_df.iterrows():
        model_name = row['model']
        
        # Skip empty rows
        if pd.isna(model_name) or model_name == '':
            continue
        
        # Create readable label
        if model_name == 'arima_statsmodels':
            readable_name = 'ARIMA'
        elif model_name == 'arima_v3':
            readable_name = 'ARIMA v3'
        elif model_name == 'gbm':
            readable_name = 'Gradient Boosting'
        elif model_name == 'naive':
            readable_name = 'Naive'
        elif model_name == 'nbeats':
            readable_name = 'N-BEATS'
        elif model_name == 'tft':
            readable_name = 'TFT'
        else:
            readable_name = model_name.replace('_', ' ').title()
        
        # Get accuracy values
        row_data = [readable_name]
        for horizon in horizons:
            accuracy = row[f'accuracy_{horizon}']
            row_data.append(f"{accuracy:.1f}%")
        
        table_data.append(row_data)
    
    # Create DataFrame for the table
    columns = ['Forecasting Model'] + [f'{h} Steps Ahead' for h in horizons]
    table_df = pd.DataFrame(table_data, columns=columns)
    
    # Print the table
    print(f"\n{'='*100}")
    print("FORECASTING MODEL ACCURACY COMPARISON (Higher is Better)")
    print(f"{'='*100}")
    print(table_df.to_string(index=False))
    print(f"{'='*100}")
    
    # Save the table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_filename = f'accuracy_table_{timestamp}.csv'
    table_df.to_csv(table_filename, index=False)
    print(f"✅ Accuracy table saved as: {table_filename}")
    
    return table_df

def create_distinct_accuracy_plot(accuracy_df):
    """Create a single, distinct accuracy plot with better scaling."""
    
    # Set up single plot with larger size for clarity
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define horizons
    horizons = [1, 3, 5, 10, 20]
    
    # Enhanced color scheme for better distinction
    colors = {
        'naive': '#1f77b4',           # Blue
        'arima_statsmodels': '#ff7f0e',  # Orange  
        'arima_v3': '#2ca02c',        # Green
        'gbm': '#d62728',             # Red
        'nbeats': '#9467bd',          # Purple
        'tft': '#8c564b',             # Brown
    }
    
    # Different marker styles for better distinction
    markers = {
        'naive': 'o',
        'arima_statsmodels': 's',
        'arima_v3': '^',
        'gbm': 'D',
        'nbeats': 'v',
        'tft': 'X',
    }
    
    # Line styles - use dashed for neural models that fell back to naive
    line_styles = {
        'naive': '-',
        'arima_statsmodels': '-',
        'arima_v3': '-',
        'gbm': '-',
        'nbeats': '--',  # Dashed because it fell back to naive
        'tft': ':',      # Dotted because it fell back to naive
    }
    
    # Find the range of accuracy values for better scaling
    all_accuracies = []
    model_data = {}
    
    for _, row in accuracy_df.iterrows():
        model_name = row['model']
        
        # Skip empty rows
        if pd.isna(model_name) or model_name == '':
            continue
        
        # Get accuracy values for each horizon
        accuracy_values = [row[f'accuracy_{h}'] for h in horizons]
        all_accuracies.extend(accuracy_values)
        model_data[model_name] = accuracy_values
    
    # Calculate optimal y-axis range for better distinction
    min_acc = min(all_accuracies)
    max_acc = max(all_accuracies)
    acc_range = max_acc - min_acc
    
    # Add padding but focus on the actual data range
    padding = acc_range * 0.1
    y_min = max(0, min_acc - padding)
    y_max = min(100, max_acc + padding)
    
    print(f"📊 Accuracy range: {min_acc:.1f}% - {max_acc:.1f}%")
    print(f"📊 Y-axis range: {y_min:.1f}% - {y_max:.1f}%")
    
    # Plot each model
    plotted_models = []
    
    for model_name, accuracy_values in model_data.items():
        
        # Create readable label
        if model_name == 'arima_statsmodels':
            readable_name = 'ARIMA'
        elif model_name == 'arima_v3':
            readable_name = 'ARIMA v3'
        elif model_name == 'gbm':
            readable_name = 'Gradient Boosting'
        elif model_name == 'naive':
            readable_name = 'Naive'
        elif model_name == 'nbeats':
            readable_name = 'N-BEATS'
        elif model_name == 'tft':
            readable_name = 'TFT'
        else:
            readable_name = model_name.replace('_', ' ').title()
        
        # Get styling
        color = colors.get(model_name, '#666666')
        marker = markers.get(model_name, 'o')
        linestyle = line_styles.get(model_name, '-')
        
        # Plot with enhanced styling for distinction
        ax.plot(horizons, accuracy_values, 
                color=color, marker=marker, linestyle=linestyle,
                linewidth=3.0, markersize=10, markeredgewidth=2, 
                markeredgecolor='white', label=readable_name,
                alpha=0.9)
        
        plotted_models.append(readable_name)
        
        # Print values for verification
        acc_str = ', '.join([f'H={h}: {acc:.1f}%' for h, acc in zip(horizons, accuracy_values)])
        print(f"✅ Plotted {readable_name}: {acc_str}")
    
    # Customize the plot for better distinction
    ax.set_xlabel('Prediction Horizon (Time Steps Ahead)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Time Series Forecasting Model Performance\nAccuracy vs. Prediction Horizon', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis to show only the horizon values
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    
    # Enhanced grid for better readability
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.7)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, which='minor')
    
    # Set optimized y-axis limits for better distinction
    ax.set_ylim(y_min, y_max)
    
    # Add minor ticks for finer resolution
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Enhanced legend with better positioning
    legend = ax.legend(fontsize=12, loc='lower left', framealpha=0.95, 
                      fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # Add some styling improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'accuracy_distinct_{timestamp}.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Distinct accuracy plot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    return fig

def main():
    """Main execution function."""
    print("="*80)
    print("DISTINCT ACCURACY PLOT GENERATOR")
    print("="*80)
    
    # Load MAE data
    print("1. Loading MAE summary data...")
    df = load_mae_data()
    
    if df is None:
        print("❌ FAILED: Could not load MAE data!")
        return
    
    # Convert to accuracy metrics
    print("\n2. Converting MAE to accuracy metrics...")
    accuracy_df = mae_to_accuracy_metrics(df)
    
    # Create and display accuracy table
    print("\n3. Creating accuracy table...")
    table_df = create_accuracy_table(accuracy_df)
    
    # Create distinct accuracy plot
    print("\n4. Creating distinct accuracy plot with optimized scaling...")
    fig = create_distinct_accuracy_plot(accuracy_df)
    
    print(f"\n{'='*80}")
    print("DISTINCT ACCURACY PLOTTING COMPLETED")
    print(f"{'='*80}")
    
    return accuracy_df, table_df

if __name__ == "__main__":
    accuracy_data, table_data = main()