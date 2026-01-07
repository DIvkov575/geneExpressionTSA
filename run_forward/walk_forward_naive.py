#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys

sys.path.append('..')

def naive_forecast(history, steps=1):
    """Simple naive forecast - repeat last value."""
    return np.full(steps, history[-1])

def walk_forward_forecast(data, min_history=5):
    """
    Perform walk-forward forecasting on a time series.
    After the initial seed, model only uses its own generated predictions.
    
    Args:
        data: Time series data (1D array)
        min_history: Minimum number of historical points to use for first forecast
    
    Returns:
        forecasts: List of forecasts
        actuals: List of actual values
        errors: List of absolute errors
    """
    forecasts = []
    actuals = []
    errors = []
    
    # Initialize with actual data for the seed period
    synthetic_history = list(data[:min_history])
    
    # Start forecasting after min_history points
    for i in range(min_history, len(data)):
        actual = data[i]
        
        # Make forecast using synthetic history (seed + previous predictions)
        forecast = naive_forecast(synthetic_history, steps=1)[0]
        
        # Add the forecast to synthetic history for next iteration
        synthetic_history.append(forecast)
        
        forecasts.append(forecast)
        actuals.append(actual)
        errors.append(abs(actual - forecast))
        
        print(f"Step {i}: Synthetic history length {len(synthetic_history)}, Actual: {actual:.6f}, Forecast: {forecast:.6f}, Error: {abs(actual - forecast):.6f}")
    
    return forecasts, actuals, errors, len(forecasts)

def main():
    print("Walk-Forward Naive Forecasting for CRE.csv Column '1'")
    print("=" * 60)
    
    # Load data
    data_path = '../data/CRE.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Extract column '1'
    if '1' not in df.columns:
        print(f"Error: Column '1' not found in data. Available columns: {df.columns.tolist()}")
        return
    
    series_data = df['1'].values
    print(f"Loaded {len(series_data)} data points from column '1'")
    print(f"Data range: {series_data.min():.6f} to {series_data.max():.6f}")
    print()
    
    # Perform walk-forward forecasting
    forecasts, actuals, errors, successful_forecasts = walk_forward_forecast(series_data, min_history=5)
    
    # Calculate summary statistics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print()
    print("Walk-Forward Forecasting Results:")
    print("=" * 40)
    print(f"Total forecasts made: {len(forecasts)}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
    
    # Save results
    results_df = pd.DataFrame({
        'step': range(5, len(series_data)),
        'actual': actuals,
        'forecast': forecasts,
        'absolute_error': errors
    })
    
    output_path = 'naive_walk_forward_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Show first few and last few results
    print("\nFirst 10 forecasting steps:")
    print(results_df.head(10).to_string(index=False))
    
    print("\nLast 10 forecasting steps:")
    print(results_df.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()