#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
import joblib
from sklearn.ensemble import GradientBoostingRegressor

sys.path.append('..')
warnings.filterwarnings("ignore")

def load_gbm_model(filepath):
    """Load trained GBM model from disk."""
    if os.path.exists(filepath):
        try:
            return joblib.load(filepath)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file not found: {filepath}")
        return None

def create_advanced_features(series, window_size=15):
    """Create enhanced time series features for GBM."""
    n = len(series)
    if n < 5:
        return None
        
    features = []
    
    # Lag features (multiple lags)
    lags = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20]
    lag_features = []
    for lag in lags:
        if n > lag:
            lag_features.append(series[-lag])
        else:
            lag_features.append(series[-1])
    
    features.extend(lag_features)
    
    # Multiple window rolling statistics
    windows = [5, 10, 15, 20]
    for w in windows:
        if n >= w:
            window_data = series[-w:]
            features.extend([
                np.mean(window_data),
                np.std(window_data),
                np.min(window_data),
                np.max(window_data),
                np.median(window_data)
            ])
        else:
            # Fallback for shorter series
            features.extend([series[-1], 0, series[-1], series[-1], series[-1]])
    
    # Current value vs historical stats
    if n >= 10:
        recent_mean = np.mean(series[-10:])
        features.extend([
            series[-1] - recent_mean,
            series[-1] / (recent_mean + 1e-8)
        ])
    else:
        features.extend([0, 1])
    
    # Trend features
    if n >= 5:
        recent_5 = series[-5:]
        trend_5 = np.polyfit(range(5), recent_5, 1)[0]
        features.append(trend_5)
    else:
        features.append(0)
    
    if n >= 10:
        recent_10 = series[-10:]
        trend_10 = np.polyfit(range(10), recent_10, 1)[0]
        features.append(trend_10)
    else:
        features.append(0)
    
    # Volatility features
    if n >= 10:
        returns = np.diff(series[-10:])
        volatility = np.std(returns)
        features.append(volatility)
    else:
        features.append(0)
    
    return np.array(features).reshape(1, -1)

def gbm_forecast(model, history, steps=1):
    """Make forecast using trained GBM model."""
    try:
        features = create_advanced_features(history)
        if features is None:
            return np.full(steps, history[-1])
        
        predictions = []
        current_history = list(history)
        
        for _ in range(steps):
            # Create features for current history
            current_features = create_advanced_features(current_history)
            if current_features is None:
                predictions.append(current_history[-1])
            else:
                # Make prediction
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                # Add prediction to history for next step
                current_history.append(pred)
        
        return np.array(predictions)
        
    except Exception as e:
        print(f"GBM forecast error: {e}")
        # Fallback to naive forecast
        return np.full(steps, history[-1])

def walk_forward_gbm_forecast(data, model, min_history=30):
    """
    Perform walk-forward forecasting on a time series using GBM.
    After the initial seed, model only uses its own generated predictions.
    
    Args:
        data: Time series data (1D array)
        model: Trained GBM model
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
            forecast_values = gbm_forecast(model, synthetic_history, steps=1)
            forecast = forecast_values[0] if len(forecast_values) > 0 else synthetic_history[-1]
            successful_forecasts += 1
        except Exception as e:
            print(f"Forecast failed at step {i}: {e}")
            # Use naive forecast as fallback
            forecast = synthetic_history[-1]
        
        # Add the forecast to synthetic history for next iteration
        synthetic_history.append(forecast)
        
        forecasts.append(forecast)
        actuals.append(actual)
        errors.append(abs(actual - forecast))
        
        print(f"Step {i}: Synthetic history length {len(synthetic_history)}, Actual: {actual:.6f}, Forecast: {forecast:.6f}, Error: {abs(actual - forecast):.6f}")
    
    return forecasts, actuals, errors, successful_forecasts

def main():
    print("Walk-Forward GBM Forecasting for CRE.csv Column '1'")
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
    
    # Load GBM model
    model_path = '../models/gbm_model.pkl'
    print(f"Loading GBM model from {model_path}...")
    model = load_gbm_model(model_path)
    
    if model is None:
        print("Failed to load GBM model. Exiting.")
        return
    
    print("GBM model loaded successfully!")
    print()
    
    # Perform walk-forward forecasting
    print("Starting walk-forward forecasting...")
    forecasts, actuals, errors, successful_forecasts = walk_forward_gbm_forecast(
        series_data, model, min_history=30
    )
    
    # Calculate summary statistics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print()
    print("Walk-Forward GBM Forecasting Results:")
    print("=" * 50)
    print(f"Total forecasts made: {len(forecasts)}")
    print(f"Successful GBM forecasts: {successful_forecasts}")
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
    
    output_path = 'gbm_walk_forward_results.csv'
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
    print(f"GBM MAE: {mae:.6f}")
    print(f"Naive MAE: {naive_mae:.6f}")
    print(f"Improvement: {((naive_mae - mae) / naive_mae * 100):.2f}%" if naive_mae > mae else f"Degradation: {((mae - naive_mae) / naive_mae * 100):.2f}%")

if __name__ == "__main__":
    main()