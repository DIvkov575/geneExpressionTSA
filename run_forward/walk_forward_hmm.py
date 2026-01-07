#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import warnings

# Add the parent directory to the system path to import train_hmm and hmm_forecast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_validate_hmm import train_hmm, hmm_forecast

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def walk_forward_hmm_forecast(data, min_history=50, n_states=4):
    """
    Perform walk-forward forecasting on a time series using HMM.
    After the initial seed, model only uses its own generated predictions.
    
    Args:
        data: Time series data (1D array)
        min_history: Minimum number of historical points to use for first forecast
        n_states: Number of hidden states for the HMM
    
    Returns:
        forecasts: List of forecasts
        actuals: List of actual values
        errors: List of absolute errors
        successful_forecasts: Number of successful forecasts
    """
    forecasts = []
    actuals = []
    errors = []
    successful_forecasts = 0
    
    # Initialize with actual data for the seed period
    synthetic_history = list(data[:min_history])
    
    # Start forecasting after min_history points
    for i in range(min_history, len(data)):
        actual = data[i]
        
        # Limit synthetic history to manage memory (keep recent data)
        max_lookback = 100  
        if len(synthetic_history) > max_lookback:
            synthetic_history = synthetic_history[-max_lookback:]
        
        # Train HMM on recent synthetic history
        model = train_hmm(synthetic_history, n_states=n_states)
        
        # Make forecast for next point using synthetic history
        if model:
            try:
                forecast_values = hmm_forecast(model, synthetic_history, horizon=1)
                forecast = forecast_values[0]
                successful_forecasts += 1
            except Exception as e:
                print(f"HMM forecast failed at step {i}: {e}")
                forecast = synthetic_history[-1] # Fallback to naive
        else:
            forecast = synthetic_history[-1] # Fallback to naive if model training fails
        
        # Add the forecast to synthetic history for next iteration
        synthetic_history.append(forecast)
        
        forecasts.append(forecast)
        actuals.append(actual)
        errors.append(abs(actual - forecast))
        
        print(f"Step {i}: Synthetic history length {len(synthetic_history)}, Actual: {actual:.6f}, Forecast: {forecast:.6f}, Error: {abs(actual - forecast):.6f}")
    
    return forecasts, actuals, errors, successful_forecasts

def main():
    print("Walk-Forward HMM Forecasting for CRE.csv Column '1'")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CRE.csv')
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
    
    print("HMM model ready!")
    print()
    
    # Perform walk-forward forecasting
    print("Starting walk-forward forecasting...")
    forecasts, actuals, errors, successful_forecasts = walk_forward_hmm_forecast(
        series_data, min_history=50, n_states=4
    )
    
    # Calculate summary statistics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print()
    print("Walk-Forward HMM Forecasting Results:")
    print("=" * 50)
    print(f"Total forecasts made: {len(forecasts)}")
    print(f"Successful HMM forecasts: {successful_forecasts}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
    
    # Save results
    results_df = pd.DataFrame({
        'step': range(50, len(series_data)),
        'actual': actuals,
        'forecast': forecasts,
        'absolute_error': errors
    })
    
    output_path = 'hmm_walk_forward_results.csv'
    results_df.to_csv(os.path.join(os.path.dirname(__file__), output_path), index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Show first few and last few results
    print("\nFirst 10 forecasting steps:")
    print(results_df.head(10).to_string(index=False))
    
    print("\nLast 10 forecasting steps:")
    print(results_df.tail(10).to_string(index=False))
    
    # Compare with naive forecasting performance
    naive_errors = []
    for i in range(50, len(series_data)):
        naive_forecast_val = series_data[i-1]  # Use previous value
        actual = series_data[i]
        naive_errors.append(abs(actual - naive_forecast_val))
    
    naive_mae = np.mean(naive_errors)
    print(f"\nComparison with Naive Forecasting:")
    print(f"HMM MAE: {mae:.6f}")
    print(f"Naive MAE: {naive_mae:.6f}")
    print(f"Improvement: {((naive_mae - mae) / naive_mae * 100):.2f}%" if naive_mae > mae else f"Degradation: {((mae - naive_mae) / naive_mae * 100):.2f}%")

if __name__ == "__main__":
    main()