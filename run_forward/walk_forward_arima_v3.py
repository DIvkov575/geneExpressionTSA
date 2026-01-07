#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings

sys.path.append('..')
from models.ARIMA_model_v3 import MultiHorizonARIMA_v3
warnings.filterwarnings("ignore")

def load_arima_v3_model(filepath):
    """Load trained ARIMA v3 model from disk."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file not found: {filepath}")
        return None

def robust_arima_v3_forecast(model, history, steps=1):
    """Make robust forecast using ARIMA v3 model with extensive parameter search."""
    if len(history) < 10:
        raise ValueError(f"Insufficient history length: {len(history)}. Need at least 10 points.")
    
    # Preprocessing: remove outliers and normalize
    history_array = np.array(history)
    q75, q25 = np.percentile(history_array, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - 2 * iqr
    upper_bound = q75 + 2 * iqr
    history_clean = np.clip(history_array, lower_bound, upper_bound)
    
    # Try multiple configurations systematically
    window_configs = [
        {'size': min(30, len(history)), 'maxiter': 100},
        {'size': min(25, len(history)), 'maxiter': 150},
        {'size': min(20, len(history)), 'maxiter': 200},
        {'size': min(15, len(history)), 'maxiter': 100},
        {'size': min(12, len(history)), 'maxiter': 80},
        {'size': min(10, len(history)), 'maxiter': 60}
    ]
    
    last_error = None
    
    for config in window_configs:
        window_size = config['size']
        maxiter = config['maxiter']
        
        if window_size < 10:
            continue
            
        try:
            # Use recent history window
            recent_history = history_clean[-window_size:] if len(history_clean) > window_size else history_clean
            
            # Create fresh model copy to avoid state issues
            import copy
            model_copy = copy.deepcopy(model)
            
            # Fit with specific maxiter
            model_copy.fit([recent_history.tolist()], maxiter=maxiter)
            
            # Make forecast
            forecast = model_copy.forecast(recent_history.tolist(), steps=steps)
            
            # Validate forecast
            if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
                raise ValueError(f"Invalid forecast values: {forecast}")
            
            # Sanity check: forecast should be within reasonable bounds
            recent_mean = np.mean(recent_history[-5:])
            recent_std = np.std(recent_history[-10:]) if len(recent_history) >= 10 else np.std(recent_history)
            
            for f in forecast:
                if abs(f - recent_mean) > 4 * recent_std:
                    raise ValueError(f"Forecast {f} too extreme (mean={recent_mean:.6f}, std={recent_std:.6f})")
            
            return forecast
            
        except Exception as e:
            last_error = e
            continue
    
    # If all attempts fail, raise the last error
    raise RuntimeError(f"ARIMA v3 failed to converge after trying all configurations. Last error: {last_error}")

def walk_forward_arima_v3_forecast(data, model, min_history=30):
    """
    Perform walk-forward forecasting on a time series using ARIMA v3.
    After the initial seed, model only uses its own generated predictions.
    
    Args:
        data: Time series data (1D array)
        model: Trained ARIMA v3 model
        min_history: Minimum number of historical points to use for first forecast
    
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
        
        # Make forecast for next point using synthetic history
        try:
            forecast_values = robust_arima_v3_forecast(model, synthetic_history, steps=1)
            forecast = forecast_values[0]
            successful_forecasts += 1
        except Exception as e:
            print(f"ARIMA v3 forecast failed at step {i}: {e}")
            print(f"Synthetic history length: {len(synthetic_history)}, recent values: {synthetic_history[-5:]}")
            raise RuntimeError(f"ARIMA v3 forecasting failed at step {i}: {e}")
        
        # Add the forecast to synthetic history for next iteration
        synthetic_history.append(forecast)
        
        forecasts.append(forecast)
        actuals.append(actual)
        errors.append(abs(actual - forecast))
        
        print(f"Step {i}: Synthetic history length {len(synthetic_history)}, Actual: {actual:.6f}, Forecast: {forecast:.6f}, Error: {abs(actual - forecast):.6f}")
    
    return forecasts, actuals, errors, successful_forecasts

def main():
    print("Walk-Forward ARIMA v3 Forecasting for CRE.csv Column '1'")
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
    
    # Load ARIMA v3 model
    model_path = '../models/arima_v3.pkl'
    print(f"Loading ARIMA v3 model from {model_path}...")
    model = load_arima_v3_model(model_path)
    
    if model is None:
        print("Failed to load ARIMA v3 model. Exiting.")
        return
    
    print("ARIMA v3 model loaded successfully!")
    print()
    
    # Perform walk-forward forecasting
    print("Starting walk-forward forecasting...")
    forecasts, actuals, errors, successful_forecasts = walk_forward_arima_v3_forecast(
        series_data, model, min_history=30
    )
    
    # Calculate summary statistics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print()
    print("Walk-Forward ARIMA v3 Forecasting Results:")
    print("=" * 50)
    print(f"Total forecasts made: {len(forecasts)}")
    print(f"Successful ARIMA v3 forecasts: {successful_forecasts}")
    print(f"Fallback to naive forecasts: {len(forecasts) - successful_forecasts}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
    
    # Save results
    results_df = pd.DataFrame({
        'step': range(30, len(series_data)),
        'actual': actuals,
        'forecast': forecasts,
        'absolute_error': errors
    })
    
    output_path = 'arima_v3_walk_forward_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Show first few and last few results
    print("\nFirst 10 forecasting steps:")
    print(results_df.head(10).to_string(index=False))
    
    print("\nLast 10 forecasting steps:")
    print(results_df.tail(10).to_string(index=False))
    
    # Compare with naive forecasting performance
    naive_errors = []
    for i in range(30, len(series_data)):
        naive_forecast = series_data[i-1]  # Use previous value
        actual = series_data[i]
        naive_errors.append(abs(actual - naive_forecast))
    
    naive_mae = np.mean(naive_errors)
    print(f"\nComparison with Naive Forecasting:")
    print(f"ARIMA v3 MAE: {mae:.6f}")
    print(f"Naive MAE: {naive_mae:.6f}")
    print(f"Improvement: {((naive_mae - mae) / naive_mae * 100):.2f}%" if naive_mae > mae else f"Degradation: {((mae - naive_mae) / naive_mae * 100):.2f}%")

if __name__ == "__main__":
    main()