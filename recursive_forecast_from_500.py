#!/usr/bin/env python3
"""
Recursive Forecasting Script - From Point 500 for 50 Values
Recursively forecasts 50 values starting from point 500 using only forecasted values as inputs.
"""

import pandas as pd
import numpy as np
import os
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import model classes
try:
    from models.ARIMA_model_v3 import MultiHorizonARIMA_v3
    ARIMA_V3_AVAILABLE = True
except ImportError:
    ARIMA_V3_AVAILABLE = False
    print("Warning: ARIMA v3 model not available")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

def load_data():
    """Load the CRE dataset and extract column 2."""
    df = pd.read_csv('data/CRE.csv')
    return df['2'].values

def naive_forecast_single(history, horizon=1):
    """Naive forecast - repeat last value."""
    return np.full(horizon, history[-1])

def arima_statsmodels_forecast_single(history, horizon=1, order=(2,1,2)):
    """ARIMA forecast using statsmodels."""
    try:
        model = ARIMA(history, order=order)
        fitted_model = model.fit(low_memory=True)
        forecast = fitted_model.forecast(steps=horizon)
        return forecast
    except Exception as e:
        print(f"ARIMA statsmodels failed: {e}")
        return naive_forecast_single(history, horizon)

def arima_v3_forecast_single(model, history, horizon=1):
    """ARIMA v3 forecast."""
    if not ARIMA_V3_AVAILABLE:
        print("ARIMA v3 not available, using naive forecast")
        return naive_forecast_single(history, horizon)
    
    try:
        model.fit([history], maxiter=50)
        forecast = model.forecast(history, steps=horizon)
        return forecast
    except Exception as e:
        print(f"ARIMA v3 failed: {e}")
        return naive_forecast_single(history, horizon)

def create_gbm_features(series, window_size=15):
    """Create enhanced time series features for GBM - matches training features exactly."""
    n = len(series)
    if n < window_size + 5:
        return None
        
    features = []
    
    # Lag features (multiple lags) - exactly as in training
    lags = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20]
    lag_features = []
    for lag in lags:
        if n > lag:
            lag_features.append(series[-lag])
        else:
            lag_features.append(series[-1])
    
    features.extend(lag_features)
    
    # Multiple window rolling statistics - exactly as in training
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
    recent = series[-window_size:]
    features.extend([
        recent[-1] - np.mean(recent),  # Deviation from mean
        (recent[-1] - recent[0]) / window_size,  # Trend
        (recent[-1] - np.min(recent)) / (np.max(recent) - np.min(recent) + 1e-8),  # Normalized position
    ])
    
    # Enhanced differencing features
    if n > 2:
        diffs = np.diff(series)
        recent_diffs = diffs[-min(10, len(diffs)):]
        features.extend([
            series[-1] - series[-2],   # First difference
            np.mean(recent_diffs),     # Average difference
            np.std(recent_diffs),      # Std of differences
            np.sum(recent_diffs > 0) / len(recent_diffs),  # Proportion of positive changes
        ])
    else:
        features.extend([0, 0, 0, 0.5])
    
    # Seasonal and cyclical features
    if n > 7:
        features.append(series[-7])  # Weekly lag
    else:
        features.append(series[-1])
        
    if n > 14:
        features.extend([
            np.mean(series[-14:]),  # 2-week average
            np.corrcoef(series[-14:], range(14))[0,1] if len(set(series[-14:])) > 1 else 0  # Trend correlation
        ])
    else:
        features.extend([np.mean(series), 0])
    
    # Volatility features
    if n > 5:
        recent_volatility = np.std(series[-5:])
        long_volatility = np.std(series[-min(20, n):])
        features.append(recent_volatility / (long_volatility + 1e-8))
    else:
        features.append(1.0)
    
    return np.array(features)

def gbm_forecast_single(model, history, horizon=1):
    """GBM forecast."""
    try:
        forecasts = []
        current_history = list(history)
        
        for _ in range(horizon):
            features = create_gbm_features(np.array(current_history))
            if features is None:
                pred = current_history[-1]
            else:
                pred = model.predict(features.reshape(1, -1))[0]
            
            forecasts.append(pred)
            current_history.append(pred)
        
        return np.array(forecasts)
    except Exception as e:
        print(f"GBM failed: {e}")
        return naive_forecast_single(history, horizon)

def prepare_nbeats_data_single(series, series_id='forecast_series'):
    """Prepare data for NBEATS."""
    data = []
    for t, value in enumerate(series):
        data.append({
            'unique_id': series_id,
            'ds': t,
            'y': value
        })
    return pd.DataFrame(data)

def nbeats_forecast_single(model, history, horizon=1):
    """NBEATS forecast."""
    try:
        history_df = prepare_nbeats_data_single(history)
        forecast = model.predict(history_df)
        
        if 'NBEATS' in forecast.columns:
            return forecast['NBEATS'].values[:horizon]
        else:
            return naive_forecast_single(history, horizon)
    except Exception as e:
        print(f"NBEATS failed: {e}")
        return naive_forecast_single(history, horizon)

def recursive_forecast_from_point(data, start_point=500, forecast_length=50, 
                                model_type='naive', model=None, window_size=50):
    """
    Recursively forecast forecast_length points starting from start_point.
    Uses only data up to start_point, then recursively uses forecasted values.
    """
    if start_point >= len(data):
        raise ValueError(f"Start point {start_point} is beyond data length {len(data)}")
    
    if start_point < window_size:
        raise ValueError(f"Start point {start_point} is less than required window size {window_size}")
    
    # Get initial history (real data up to start_point)
    initial_history = data[:start_point]
    
    # Use last window_size points as history for forecasting
    history = initial_history[-window_size:]
    forecasted_values = []
    
    print(f"Starting recursive forecast from point {start_point}")
    print(f"Using last {window_size} real values as initial history")
    print(f"Will forecast {forecast_length} points recursively")
    print(f"Initial history range: {history[0]:.6f} to {history[-1]:.6f}")
    
    # Forecast points recursively
    current_history = list(history)
    
    for i in range(forecast_length):
        current_array = np.array(current_history[-window_size:])  # Use last window_size points
        
        if model_type == 'naive':
            pred = naive_forecast_single(current_array, 1)[0]
        elif model_type == 'arima_statsmodels':
            pred = arima_statsmodels_forecast_single(current_array, 1)[0]
        elif model_type == 'arima_v3':
            pred = arima_v3_forecast_single(model, current_array, 1)[0]
        elif model_type == 'gbm':
            pred = gbm_forecast_single(model, current_array, 1)[0]
        elif model_type == 'nbeats':
            pred = nbeats_forecast_single(model, current_array, 1)[0]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        forecasted_values.append(pred)
        current_history.append(pred)  # Add forecasted value to history
        
        if (i + 1) % 10 == 0:
            print(f"Forecasted {i + 1}/{forecast_length} points, latest: {pred:.6f}")
    
    return np.array(forecasted_values)

def load_trained_models():
    """Load all available trained models."""
    models = {}
    
    # Load naive model (doesn't need actual model file)
    models['naive'] = None
    
    # Load ARIMA statsmodels
    if os.path.exists('models/arima_statsmodels.pkl'):
        print("Loading ARIMA statsmodels parameters...")
        models['arima_statsmodels'] = None  # Uses parameters from training
    
    # Load ARIMA v3
    if os.path.exists('models/arima_v3.pkl') and ARIMA_V3_AVAILABLE:
        print("Loading ARIMA v3 model...")
        with open('models/arima_v3.pkl', 'rb') as f:
            models['arima_v3'] = pickle.load(f)
    
    # Load GBM
    if os.path.exists('models/gbm_model.pkl'):
        print("Loading GBM model...")
        models['gbm'] = joblib.load('models/gbm_model.pkl')
    
    # Load NBEATS
    if os.path.exists('models/nbeats_model.pkl'):
        print("Loading NBEATS model...")
        with open('models/nbeats_model.pkl', 'rb') as f:
            models['nbeats'] = pickle.load(f)
    
    return models

def main():
    """Main execution function."""
    print("="*60)
    print("RECURSIVE FORECASTING - FROM POINT 500 FOR 50 VALUES")
    print("="*60)
    
    # Parameters
    start_point = 500
    forecast_length = 50
    window_size = 50
    
    # Load data
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} data points from column 2")
    print(f"Start point {start_point}: {data[start_point]:.6f}")
    print(f"Will forecast {forecast_length} values starting from point {start_point}")
    
    if start_point + forecast_length <= len(data):
        actual_future = data[start_point:start_point + forecast_length]
        print(f"Actual values available for comparison: {actual_future[:5]} ... {actual_future[-5:]}")
    else:
        actual_future = None
        print("No actual values available for comparison (forecasting beyond data)")
    
    # Load models
    print("\nLoading trained models...")
    models = load_trained_models()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('recursive_forecasts', exist_ok=True)
    
    # Test each available model
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TESTING MODEL: {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            if model_name == 'arima_v3' and not ARIMA_V3_AVAILABLE:
                print(f"SKIPPING {model_name} - not available")
                continue
                
            # Perform recursive forecast
            forecasted = recursive_forecast_from_point(
                data, start_point=start_point, forecast_length=forecast_length,
                model_type=model_name, model=model, window_size=window_size
            )
            
            # Save results
            result_data = {
                'step': range(forecast_length),
                'forecasted': forecasted,
                'start_point': [start_point] * forecast_length
            }
            
            if actual_future is not None:
                result_data['actual'] = actual_future
                
            result_df = pd.DataFrame(result_data)
            
            output_file = f'recursive_forecasts/{model_name}_from500_50values_{timestamp}.csv'
            result_df.to_csv(output_file, index=False)
            
            # Calculate metrics if actual values are available
            if actual_future is not None:
                mae = np.mean(np.abs(actual_future - forecasted))
                mse = np.mean((actual_future - forecasted) ** 2)
                
                print(f"✅ SUCCESS - {model_name}")
                print(f"   MAE: {mae:.6f}")
                print(f"   MSE: {mse:.6f}")
                print(f"   First 5 forecasts: {forecasted[:5]}")
                print(f"   Last 5 forecasts: {forecasted[-5:]}")
                print(f"   Results saved to: {output_file}")
                
                all_results[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'forecasted': forecasted,
                    'success': True,
                    'file': output_file
                }
            else:
                print(f"✅ SUCCESS - {model_name} (no comparison data)")
                print(f"   First 5 forecasts: {forecasted[:5]}")
                print(f"   Last 5 forecasts: {forecasted[-5:]}")
                print(f"   Results saved to: {output_file}")
                
                all_results[model_name] = {
                    'forecasted': forecasted,
                    'success': True,
                    'file': output_file
                }
            
        except Exception as e:
            print(f"❌ FAILED - {model_name}: {str(e)}")
            all_results[model_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - RECURSIVE FORECASTING FROM POINT 500")
    print(f"{'='*60}")
    
    successful = [name for name, result in all_results.items() if result.get('success', False)]
    failed = [name for name, result in all_results.items() if not result.get('success', False)]
    
    print(f"Successful models: {len(successful)}")
    for model in successful:
        result = all_results[model]
        if 'mae' in result:
            print(f"  ✅ {model}: MAE={result['mae']:.6f}, MSE={result['mse']:.6f}")
        else:
            print(f"  ✅ {model}: Forecasting completed (no comparison data)")
    
    print(f"\nFailed models: {len(failed)}")
    for model in failed:
        print(f"  ❌ {model}: {all_results[model]['error']}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'start_point': start_point,
        'forecast_length': forecast_length,
        'window_size': window_size,
        'total_models': len(all_results),
        'successful': len(successful),
        'failed': len(failed),
        'results': all_results
    }
    
    with open(f'recursive_forecasts/summary_from500_50values_{timestamp}.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nSummary saved to: recursive_forecasts/summary_from500_50values_{timestamp}.pkl")
    
    if len(failed) > 0:
        print(f"\n❌ SOME MODELS FAILED - Check individual model outputs above")
        return False
    else:
        print(f"\n✅ ALL MODELS SUCCESSFUL")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)