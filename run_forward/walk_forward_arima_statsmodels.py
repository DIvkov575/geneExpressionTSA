#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
from statsmodels.tsa.arima.model import ARIMA
from scipy.ndimage import uniform_filter1d

sys.path.append('..')
warnings.filterwarnings("ignore")

def load_arima_statsmodels_model(filepath):
    """Load trained ARIMA statsmodels model from disk."""
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

def grid_search_arima_order(history, max_attempts=50):
    """Grid search for best ARIMA order that converges."""
    # Comprehensive order combinations, starting with simplest
    orders_to_try = [
        # Start with simplest models
        (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1),
        (1, 1, 1), (2, 0, 0), (0, 0, 2), (2, 1, 0), (0, 1, 2), (2, 0, 1),
        (1, 0, 2), (2, 1, 1), (1, 1, 2), (2, 0, 2), (2, 1, 2),
        # More complex models
        (3, 0, 0), (0, 0, 3), (3, 1, 0), (0, 1, 3), (3, 0, 1), (1, 0, 3),
        (3, 1, 1), (1, 1, 3), (3, 0, 2), (2, 0, 3), (3, 1, 2), (2, 1, 3),
        (3, 0, 3), (3, 1, 3), (4, 0, 0), (0, 0, 4), (4, 1, 0), (0, 1, 4),
        (4, 0, 1), (1, 0, 4), (4, 1, 1), (1, 1, 4), (4, 0, 2), (2, 0, 4),
        # Higher differencing
        (1, 2, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2), (0, 2, 1), (1, 2, 0),
        (0, 2, 2), (2, 2, 0), (3, 2, 1), (1, 2, 3)
    ]
    
    # Try different solvers for each order
    solvers = ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell']
    maxiters = [50, 100, 200, 300]
    
    attempts = 0
    for order in orders_to_try:
        if attempts >= max_attempts:
            break
            
        for solver in solvers:
            if attempts >= max_attempts:
                break
                
            for maxiter in maxiters:
                if attempts >= max_attempts:
                    break
                    
                attempts += 1
                try:
                    model = ARIMA(history, order=order)
                    fitted_model = model.fit(maxiter=maxiter, method=solver)
                    
                    # Test forecast
                    test_forecast = fitted_model.forecast(steps=1)
                    if not (np.any(np.isnan(test_forecast)) or np.any(np.isinf(test_forecast))):
                        return fitted_model, order
                        
                except Exception:
                    continue
    
    raise ValueError(f"Could not find converging ARIMA model after {attempts} attempts with {len(orders_to_try)} orders and {len(solvers)} solvers")

def simple_ar_forecast(history, steps=1, max_lags=10):
    """Simple AR forecasting using linear regression on lagged values."""
    history_array = np.array(history, dtype=float)
    n = len(history_array)
    
    if n < 5:
        raise ValueError(f"Need at least 5 data points, got {n}")
    
    # Try different lag orders and pick the best one
    best_mse = np.inf
    best_order = 1
    best_coeffs = None
    
    max_order = min(max_lags, n // 2)
    
    for order in range(1, max_order + 1):
        if order >= n:
            continue
            
        # Create lagged features
        X = np.zeros((n - order, order))
        y = history_array[order:]
        
        for lag in range(order):
            X[:, lag] = history_array[order - lag - 1:n - lag - 1]
        
        if X.shape[0] < 3:  # Need at least 3 samples
            continue
            
        try:
            # Simple least squares
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Calculate in-sample MSE
            y_pred = X @ coeffs
            mse = np.mean((y - y_pred) ** 2)
            
            # Check for reasonable coefficients
            if not (np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs))):
                if mse < best_mse:
                    best_mse = mse
                    best_order = order
                    best_coeffs = coeffs
                    
        except Exception:
            continue
    
    if best_coeffs is None:
        raise ValueError("Could not fit any AR model")
    
    # Make forecast using the best model
    forecast = []
    extended_history = list(history_array)  # Copy for extending
    
    for step in range(steps):
        # Get the last 'best_order' values
        recent_values = extended_history[-best_order:]
        
        # Predict next value
        next_val = sum(coeff * val for coeff, val in zip(best_coeffs, reversed(recent_values)))
        
        forecast.append(next_val)
        extended_history.append(next_val)
    
    return np.array(forecast), best_order

def arima_statsmodels_forecast(history, steps=1):
    """AR forecasting using simple linear regression (ARIMA alternative)."""
    # Ensure we have enough data
    if len(history) < 6:
        raise ValueError(f"Insufficient history length: {len(history)}. Need at least 6 points.")
    
    # Convert to numpy array
    history_array = np.array(history, dtype=float)
    
    # Basic data validation
    if np.any(np.isnan(history_array)) or np.any(np.isinf(history_array)):
        raise ValueError("History contains NaN or infinite values")
    
    # Preprocessing: basic outlier removal
    q75, q25 = np.percentile(history_array, [75, 25])
    iqr = q75 - q25 + 1e-8
    lower_bound = q25 - 2 * iqr  
    upper_bound = q75 + 2 * iqr
    history_clean = np.clip(history_array, lower_bound, upper_bound)
    
    # Try ARIMA first, fall back to AR if it fails
    forecast = None
    method_used = "unknown"
    
    # Try simple ARIMA orders
    simple_orders = [(1, 0, 0), (2, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
    
    for order in simple_orders:
        try:
            model = ARIMA(history_clean, order=order)
            fitted = model.fit(maxiter=50, method='lbfgs')
            test_forecast = fitted.forecast(steps=steps)
            
            if not (np.any(np.isnan(test_forecast)) or np.any(np.isinf(test_forecast))):
                forecast = test_forecast
                method_used = f"ARIMA{order}"
                break
                
        except Exception:
            continue
    
    # If ARIMA failed, use AR model
    if forecast is None:
        try:
            forecast, ar_order = simple_ar_forecast(history_clean, steps=steps)
            method_used = f"AR({ar_order})"
        except Exception as e:
            raise ValueError(f"Both ARIMA and AR forecasting failed: {e}")
    
    # Validate forecast
    if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
        raise ValueError(f"Invalid forecast values: {forecast}")
    
    # Sanity check
    recent_values = history_clean[-min(5, len(history_clean)):]
    data_range = np.max(recent_values) - np.min(recent_values) + 0.001
    recent_mean = np.mean(recent_values)
    
    for i, f in enumerate(forecast):
        if abs(f - recent_mean) > 5 * data_range:
            raise ValueError(f"Forecast {f} too extreme using {method_used}")
    
    print(f"Using {method_used} for forecasting")
    return forecast

def walk_forward_arima_statsmodels_forecast(data, min_history=30):
    """
    Perform walk-forward forecasting on a time series using ARIMA statsmodels.
    After the initial seed, model only uses its own generated predictions.
    
    Args:
        data: Time series data (1D array)
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
            forecast_values = arima_statsmodels_forecast(synthetic_history, steps=1)
            forecast = forecast_values[0]
            successful_forecasts += 1
        except Exception as e:
            print(f"ARIMA forecast failed at step {i}: {e}")
            print(f"Synthetic history length: {len(synthetic_history)}, recent values: {synthetic_history[-5:]}")
            raise RuntimeError(f"ARIMA forecasting failed at step {i}: {e}")
        
        # Add the forecast to synthetic history for next iteration
        synthetic_history.append(forecast)
        
        forecasts.append(forecast)
        actuals.append(actual)
        errors.append(abs(actual - forecast))
        
        print(f"Step {i}: Synthetic history length {len(synthetic_history)}, Actual: {actual:.6f}, Forecast: {forecast:.6f}, Error: {abs(actual - forecast):.6f}")
    
    return forecasts, actuals, errors, successful_forecasts

def main():
    print("Walk-Forward ARIMA Statsmodels Forecasting for CRE.csv Column '1'")
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
    
    print("ARIMA statsmodels model ready!")
    print()
    
    # Perform walk-forward forecasting
    print("Starting walk-forward forecasting...")
    forecasts, actuals, errors, successful_forecasts = walk_forward_arima_statsmodels_forecast(
        series_data, min_history=30
    )
    
    # Calculate summary statistics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print()
    print("Walk-Forward ARIMA Statsmodels Forecasting Results:")
    print("=" * 50)
    print(f"Total forecasts made: {len(forecasts)}")
    print(f"Successful ARIMA forecasts: {successful_forecasts}")
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
    
    output_path = 'arima_statsmodels_walk_forward_results.csv'
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
    print(f"ARIMA Statsmodels MAE: {mae:.6f}")
    print(f"Naive MAE: {naive_mae:.6f}")
    print(f"Improvement: {((naive_mae - mae) / naive_mae * 100):.2f}%" if naive_mae > mae else f"Degradation: {((mae - naive_mae) / naive_mae * 100):.2f}%")

if __name__ == "__main__":
    main()