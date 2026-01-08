#!/usr/bin/env python3
"""
Create per-horizon MAE analysis table for column 2 forecasts.
"""

import pandas as pd
import numpy as np

def analyze_per_horizon_mae():
    """Analyze MAE by forecast horizon for column 2."""
    
    # Load recursive forecasts
    forecast_file = '../forecasts/recursive_forecasts_20260107_124923.csv'
    df = pd.read_csv(forecast_file)
    
    # Filter for column 2 and recursive forecasts only
    df_col2 = df[(df['column'] == 2) & (df['is_recursive'] == True)].copy()
    
    if df_col2.empty:
        print("No recursive forecast data found for column 2")
        return
    
    # Calculate absolute error
    df_col2['absolute_error'] = np.abs(df_col2['actual_value'] - df_col2['predicted_value'])
    
    # Group by model and step to get horizon-based performance
    horizon_mae = df_col2.groupby(['model', 'step'])['absolute_error'].mean().reset_index()
    horizon_mae = horizon_mae.rename(columns={'absolute_error': 'mae'})
    
    # Pivot to get models as columns and horizons as rows
    mae_table = horizon_mae.pivot(index='step', columns='model', values='mae')
    
    # Define horizon ranges for summary
    horizons = [1, 5, 10, 20, 30, 40, 50]
    
    print("📊 Per-Horizon MAE Analysis (Column 2)")
    print("=" * 60)
    print("\n🔍 Detailed MAE by Forecast Horizon:")
    print("-" * 60)
    
    # Show detailed table for key horizons
    available_horizons = [h for h in horizons if h in mae_table.index]
    if available_horizons:
        detailed_table = mae_table.loc[available_horizons]
        print(detailed_table.round(6))
        
        print(f"\n📈 MAE Degradation Summary:")
        print("-" * 40)
        
        # Calculate degradation from 1-step to 10-step, 20-step, etc.
        if 1 in mae_table.index:
            baseline_mae = mae_table.loc[1]
            
            for horizon in [5, 10, 20, 30, 40, 50]:
                if horizon in mae_table.index:
                    current_mae = mae_table.loc[horizon]
                    degradation = ((current_mae / baseline_mae - 1) * 100).round(1)
                    
                    print(f"\n{horizon}-step vs 1-step MAE increase (%):")
                    for model in degradation.index:
                        print(f"  {model:15s}: {degradation[model]:+6.1f}%")
    
    # Overall horizon performance summary
    print(f"\n📊 Average MAE Across All Horizons:")
    print("-" * 40)
    overall_mae = mae_table.mean().sort_values()
    for model, mae in overall_mae.items():
        print(f"{model:15s}: {mae:.6f}")
    
    # Best/worst horizons for each model
    print(f"\n🎯 Best/Worst Horizons by Model:")
    print("-" * 40)
    for model in mae_table.columns:
        model_data = mae_table[model].dropna()
        if len(model_data) > 0:
            best_horizon = model_data.idxmin()
            worst_horizon = model_data.idxmax()
            best_mae = model_data.min()
            worst_mae = model_data.max()
            print(f"{model:15s}: Best = {best_horizon:2d}-step ({best_mae:.6f}), Worst = {worst_horizon:2d}-step ({worst_mae:.6f})")

if __name__ == "__main__":
    analyze_per_horizon_mae()