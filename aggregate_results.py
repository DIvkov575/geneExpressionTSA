#!/usr/bin/env python3
"""
Script to aggregate individual model result CSV files into a single results table.
Only includes models that are currently in results.csv.
"""

import pandas as pd
import os
from pathlib import Path

def aggregate_results():
    """Aggregate model results from individual CSV files into a single table."""
    
    # Define the mapping between results.csv columns and their corresponding files
    model_files = {
        'naive': 'results/naive_mae_results.csv',
        'gbm': 'results/gbm_mae_results.csv', 
        'moving_average': 'results/moving_average_mae_results.csv',
        'arima_v3': 'results/arima_mae_results.csv',  # arima_v3 uses arima_mae_results.csv
        'arima_statsmodels': 'results/arima_statsmodels_mae_results.csv',
        'nbeats': 'results/nbeats_mae_results.csv',
        'tft': 'results/tft_mae_results.csv'
    }
    
    print("Aggregating model results...")
    
    # Initialize the results dataframe with horizon column
    results_df = None
    
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            print(f"Processing {model_name} from {file_path}")
            
            # Read the individual model results
            model_df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if 'horizon' not in model_df.columns or 'mae' not in model_df.columns:
                print(f"Warning: {file_path} missing required columns, skipping...")
                continue
            
            # Extract horizon and mae columns
            model_data = model_df[['horizon', 'mae']].copy()
            model_data = model_data.rename(columns={'mae': model_name})
            
            # Merge with results dataframe
            if results_df is None:
                results_df = model_data
            else:
                results_df = pd.merge(results_df, model_data, on='horizon', how='outer')
        else:
            print(f"Warning: {file_path} not found, skipping {model_name}")
    
    if results_df is None:
        print("Error: No valid result files found!")
        return
    
    # Sort by horizon
    results_df = results_df.sort_values('horizon').reset_index(drop=True)
    
    # Round to 6 decimal places for consistency
    numeric_cols = [col for col in results_df.columns if col != 'horizon']
    results_df[numeric_cols] = results_df[numeric_cols].round(6)
    
    # Save the aggregated results
    output_file = 'results_aggregated.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nAggregated results saved to {output_file}")
    
    # Display the results
    print(f"\nAggregated Results:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Show model rankings for horizon=1
    horizon_1 = results_df[results_df['horizon'] == 1].iloc[0]
    model_scores = [(col, horizon_1[col]) for col in numeric_cols if pd.notna(horizon_1[col])]
    model_scores.sort(key=lambda x: x[1])  # Sort by MAE (lower is better)
    
    print(f"\nModel Rankings (Horizon=1, by MAE):")
    print("=" * 40)
    for rank, (model, score) in enumerate(model_scores, 1):
        print(f"{rank:2d}. {model:20s} {score:.6f}")

if __name__ == "__main__":
    aggregate_results()