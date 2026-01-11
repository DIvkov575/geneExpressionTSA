#!/usr/bin/env python3
"""Quick test of recursive forecasting for full series with just ARIMA v3 and Naive models."""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    from models.ARIMA_model_v3 import MultiHorizonARIMA_v3
    ARIMA_V3_AVAILABLE = True
except ImportError:
    ARIMA_V3_AVAILABLE = False

def load_data():
    """Load the CRE dataset and extract column 2."""
    df = pd.read_csv('data/CRE.csv')
    return df['2'].values

def naive_forecast_single(history, horizon=1):
    """Naive forecast - repeat last value."""
    return np.full(horizon, history[-1])

def arima_v3_forecast_single(model, history, horizon=1):
    """ARIMA v3 forecast."""
    if not ARIMA_V3_AVAILABLE:
        return naive_forecast_single(history, horizon)
    
    try:
        model.fit([history], maxiter=50)
        forecast = model.forecast(history, steps=horizon)
        return forecast
    except Exception as e:
        print(f"ARIMA v3 failed: {e}")
        return naive_forecast_single(history, horizon)

def recursive_forecast_quick_test(data, model_type='naive', model=None, initial_window=20, forecast_points=50):
    """Quick test - forecast only 50 points instead of full series."""
    
    if len(data) < initial_window + forecast_points:
        raise ValueError(f"Data too short for test.")
    
    # Initialize with first few real values
    forecasted_series = list(data[:initial_window])
    
    print(f"Quick test: forecasting {forecast_points} points after {initial_window} real values")
    
    # Forecast points recursively
    for i in range(forecast_points):
        current_history = np.array(forecasted_series)
        
        if model_type == 'naive':
            pred = naive_forecast_single(current_history, 1)[0]
        elif model_type == 'arima_v3':
            pred = arima_v3_forecast_single(model, current_history, 1)[0]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        forecasted_series.append(pred)
        
        if (i + 1) % 10 == 0:
            print(f"Forecasted {i + 1}/{forecast_points} points, latest: {pred:.6f}")
    
    return np.array(forecasted_series)

def main():
    print("="*60)
    print("QUICK TEST - RECURSIVE FORECASTING")  
    print("="*60)
    
    # Load data
    data = load_data()
    print(f"Loaded {len(data)} data points")
    
    # Test models
    models_to_test = [
        ('naive', None),
    ]
    
    if ARIMA_V3_AVAILABLE and os.path.exists('models/arima_v3.pkl'):
        print("Loading ARIMA v3 model...")
        with open('models/arima_v3.pkl', 'rb') as f:
            arima_model = pickle.load(f)
        models_to_test.append(('arima_v3', arima_model))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('recursive_forecasts', exist_ok=True)
    
    for model_name, model in models_to_test:
        print(f"\n{'='*40}")
        print(f"TESTING: {model_name.upper()}")
        print(f"{'='*40}")
        
        try:
            forecasted = recursive_forecast_quick_test(
                data, model_type=model_name, model=model, 
                initial_window=20, forecast_points=50
            )
            
            # Calculate metrics for forecasted portion
            actual_forecast_portion = data[20:70]  # 50 points starting from position 20
            forecasted_portion = forecasted[20:70]
            
            mae = np.mean(np.abs(actual_forecast_portion - forecasted_portion))
            mse = np.mean((actual_forecast_portion - forecasted_portion) ** 2)
            
            print(f"✅ SUCCESS - {model_name}")
            print(f"   MAE (forecast portion): {mae:.6f}")
            print(f"   MSE (forecast portion): {mse:.6f}")
            
            # Save quick test results
            result_df = pd.DataFrame({
                'actual': data[:70],
                'forecasted': forecasted,
                'step': range(70),
                'is_forecast': [i >= 20 for i in range(70)]
            })
            
            output_file = f'recursive_forecasts/{model_name}_quick_test_{timestamp}.csv'
            result_df.to_csv(output_file, index=False)
            print(f"   Quick test results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ FAILED - {model_name}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("QUICK TEST COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()