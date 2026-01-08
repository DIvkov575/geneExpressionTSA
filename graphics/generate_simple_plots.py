#!/usr/bin/env python3
"""
Generate ONLY the 4 required plots for column 2 with all models superimposed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_column2_data():
    """Load column 2 forecast data for all available models."""
    results = {}
    
    # First try to load from our latest comprehensive recursive forecasts file
    latest_recursive_file = '../forecasts/recursive_forecasts_20260107_124923.csv'
    if os.path.exists(latest_recursive_file):
        df_all = pd.read_csv(latest_recursive_file)
        # Filter for column 2 only
        df_col2 = df_all[df_all['column'] == 2].copy()
        
        if not df_col2.empty:
            # Group by model and create individual dataframes
            for model_name in df_col2['model'].unique():
                model_df = df_col2[df_col2['model'] == model_name].copy()
                # Rename columns to match expected format
                model_df = model_df.rename(columns={
                    'predicted_value': 'forecast',
                    'actual_value': 'actual'
                })
                # Calculate absolute error and MAPE
                model_df['absolute_error'] = np.abs(model_df['actual'] - model_df['forecast'])
                model_df['mape'] = np.abs((model_df['actual'] - model_df['forecast']) / model_df['actual']) * 100
                model_df = model_df.dropna()
                
                # Use proper model names for display
                display_name = {
                    'arima_v3': 'ARIMA v3',
                    'naive': 'Naive', 
                    'arima_statsmodels': 'ARIMA Stats',
                    'gbm': 'GBM',
                    'tft': 'TFT',
                    'nbeats': 'NBEATS'
                }.get(model_name, model_name.upper())
                
                results[display_name] = model_df
                print(f"✓ Loaded {display_name}: {len(model_df)} points")
            
            return results
    
    # Fallback to old method if recursive file not available
    model_files = {
        'GBM': '../run_forward/gbm_column_2_results.csv',
        'NBEATS': '../run_forward/nbeats_column_2_results.csv'
    }
    
    walk_forward_files = {
        'Naive': '../run_forward/naive_walk_forward_results.csv',
        'ARIMA Stats': '../run_forward/arima_statsmodels_walk_forward_results.csv',
        'ARIMA v3': '../run_forward/arima_v3_walk_forward_results.csv',
        'TFT': '../run_forward/tft_walk_forward_results.csv'
    }
    
    all_files = {**model_files, **walk_forward_files}
    
    for model_name, filepath in all_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if all(col in df.columns for col in ['step', 'actual', 'forecast', 'absolute_error']):
                # Calculate MAPE
                df['mape'] = np.abs((df['actual'] - df['forecast']) / df['actual']) * 100
                df = df.dropna()
                results[model_name] = df
                print(f"✓ Loaded {model_name}: {len(df)} points")
    
    return results

def load_extrapolation_data():
    """Load column 2 extrapolation data."""
    # Use the latest extrapolation data with ALL 6 models
    extrap_file = '../forecasts/column2_extrapolation_20260107_124923.csv'
    
    if os.path.exists(extrap_file):
        df = pd.read_csv(extrap_file)
        print(f"✓ Loaded extrapolation data: {extrap_file}")
        print(f"  Models included: {sorted(df['model'].unique())}")
        return df
    return None

def load_original_data():
    """Load original CRE.csv column 2 data."""
    data_file = '../data/CRE.csv'
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        return df['2'].values if '2' in df.columns else None
    return None

def plot1_accuracy_vs_time():
    """Plot 1: R² vs time for all models superimposed"""
    results = load_column2_data()
    if not results:
        print("No data for accuracy plot")
        return
    
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    for i, (model_name, df) in enumerate(results.items()):
        # Calculate rolling R² (using 50-step window)
        window_size = min(50, len(df) // 10)
        rolling_r2 = []
        steps = []
        
        for j in range(window_size, len(df)):  # Use all data points, don't skip
            window_actual = df['actual'].iloc[j-window_size:j]
            window_forecast = df['forecast'].iloc[j-window_size:j]
            
            # Calculate R²
            ss_res = np.sum((window_actual - window_forecast) ** 2)
            ss_tot = np.sum((window_actual - np.mean(window_actual)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Avoid division by zero
            
            rolling_r2.append(max(-5, min(1, r2)))  # Wider range, less clamping
            steps.append(df['step'].iloc[j])
        
        if len(rolling_r2) > 0:
            plt.plot(steps, rolling_r2, label=model_name, 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Rolling R² (50-step window)')
    plt.title('Forecast Accuracy (R²) vs Time (Column 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 1)  # Wider range
    plt.tight_layout()
    plt.savefig('accuracy_vs_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: accuracy_vs_time.png")

def plot2_forecast_vs_time():
    """Plot 2: Forecast vs time for all models superimposed"""
    results = load_column2_data()
    if not results:
        print("No data for forecast plot")
        return
    
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Get actual values (same for all models)
    first_model = list(results.values())[0]
    plt.plot(first_model['step'], first_model['actual'], 'black', 
            linewidth=2, label='Actual', alpha=0.9)
    
    # Plot each model's forecasts
    for i, (model_name, df) in enumerate(results.items()):
        plt.plot(df['step'], df['forecast'], '--', label=f'{model_name} Forecast',
                color=colors[i % len(colors)], linewidth=2, alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value - Log Scale')
    plt.yscale('log')
    plt.title('Forecast vs Time (Column 2) - Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_vs_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: forecast_vs_time.png")

def plot3_extrapolation_vs_time():
    """Plot 3: Extrapolated forecast vs time"""
    original_data = load_original_data()
    extrap_data = load_extrapolation_data()
    
    if original_data is None or extrap_data is None:
        print("No data for extrapolation plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot original data
    time_steps = np.arange(len(original_data))
    plt.plot(time_steps, original_data, 'black', linewidth=2, label='Original Data', alpha=0.9)
    
    # Plot extrapolations by model
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, model_name in enumerate(extrap_data['model'].unique()):
        model_data = extrap_data[extrap_data['model'] == model_name]
        extrap_mask = model_data['is_extrapolated']
        extrap_steps = model_data[extrap_mask]['step']
        extrap_values = model_data[extrap_mask]['value']
        
        if len(extrap_steps) > 0:
            plt.plot(extrap_steps, extrap_values, '--', 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.7,
                    label=f'{model_name} Extrapolation')
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Extrapolated Forecast vs Time (Column 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('extrapolation_vs_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: extrapolation_vs_time.png")

def plot4_true_vs_predicted():
    """Plot 4: True vs predicted scatter for all models"""
    results = load_column2_data()
    if not results:
        print("No data for scatter plot")
        return
    
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Collect all actual values to set axis range
    all_actual = []
    all_forecast = []
    
    for model_name, df in results.items():
        all_actual.extend(df['actual'].values)
        all_forecast.extend(df['forecast'].values)
    
    min_val = min(min(all_actual), min(all_forecast))
    max_val = max(max(all_actual), max(all_forecast))
    
    # Plot perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
    
    # Plot each model's scatter
    for i, (model_name, df) in enumerate(results.items()):
        plt.scatter(df['actual'], df['forecast'], 
                   color=colors[i % len(colors)], alpha=0.6, s=20, label=model_name)
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted (Column 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Set proper axis limits and ticks
    margin = (max_val - min_val) * 0.05
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    
    # Set dynamic ticks
    tick_range = np.linspace(min_val, max_val, 8)
    plt.xticks(tick_range, [f'{x:.3f}' for x in tick_range])
    plt.yticks(tick_range, [f'{y:.3f}' for y in tick_range])
    
    plt.tight_layout()
    plt.savefig('true_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: true_vs_predicted.png")

def main():
    print("🎯 Generating 4 Required Column 2 Plots")
    print("=" * 40)
    
    plot1_accuracy_vs_time()
    plot2_forecast_vs_time() 
    plot3_extrapolation_vs_time()
    plot4_true_vs_predicted()
    
    print("\n✅ All 4 plots generated successfully!")
    print("Files created:")
    print("- accuracy_vs_time.png")
    print("- forecast_vs_time.png") 
    print("- extrapolation_vs_time.png")
    print("- true_vs_predicted.png")

if __name__ == "__main__":
    main()