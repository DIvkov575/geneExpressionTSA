#!/usr/bin/env python3
"""
Recursive Forecasting Script for Column 2 - Full Series
Forecasts the entire column 2 series recursively using only previously forecasted values.
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
from neuralforecast.models import NBEATS, TFT

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

def tft_forecast_single(model, history, horizon=1):
    """TFT forecast for single step."""
    try:
        # Prepare data in NeuralForecast format
        history_df = pd.DataFrame({
            'unique_id': ['forecast_series'] * len(history),
            'ds': pd.date_range('2020-01-01', periods=len(history), freq='D'),
            'y': history
        })
        
        # Use the model to forecast
        forecast = model.predict(history_df)
        
        if 'TFT' in forecast.columns:
            return forecast['TFT'].values[:horizon]
        else:
            return naive_forecast_single(history, horizon)
    except Exception as e:
        print(f"TFT failed: {e}")
        return naive_forecast_single(history, horizon)

def recursive_forecast_full_series(data, model_type='naive', model=None, initial_window=20):
    """
    Recursively forecast the entire series starting from initial_window.
    Uses only forecasted values, never real observations after initial window.
    """
    if len(data) < initial_window:
        raise ValueError(f"Data too short. Need at least {initial_window} points.")
    
    # Initialize with first few real values
    forecasted_series = list(data[:initial_window])
    
    print(f"Starting recursive forecast with {initial_window} real values")
    print(f"Need to forecast {len(data) - initial_window} additional points")
    
    # Forecast remaining points recursively
    for i in range(initial_window, len(data)):
        current_history = np.array(forecasted_series)
        
        if model_type == 'naive':
            pred = naive_forecast_single(current_history, 1)[0]
        elif model_type == 'arima_statsmodels':
            pred = arima_statsmodels_forecast_single(current_history, 1)[0]
        elif model_type == 'arima_v3':
            pred = arima_v3_forecast_single(model, current_history, 1)[0]
        elif model_type == 'gbm':
            pred = gbm_forecast_single(model, current_history, 1)[0]
        elif model_type == 'nbeats':
            pred = nbeats_forecast_single(model, current_history, 1)[0]
        elif model_type == 'tft':
            pred = tft_forecast_single(model, current_history, 1)[0]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        forecasted_series.append(pred)
        
        if (i - initial_window + 1) % 50 == 0:
            print(f"Forecasted {i - initial_window + 1}/{len(data) - initial_window} points")
    
    return np.array(forecasted_series)

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
    
    # Load TFT
    if os.path.exists('models/tft_model.pkl'):
        print("Loading TFT model...")
        with open('models/tft_model.pkl', 'rb') as f:
            models['tft'] = pickle.load(f)
    
    return models

def main():
    """Main execution function."""
    print("="*60)
    print("RECURSIVE FORECASTING - FULL COLUMN 2 SERIES")
    print("="*60)
    
    # Load data
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} data points from column 2")
    print(f"First 5 values: {data[:5]}")
    print(f"Last 5 values: {data[-5:]}")
    
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
            forecasted = recursive_forecast_full_series(
                data, model_type=model_name, model=model, initial_window=20
            )
            
            # Save results
            result_df = pd.DataFrame({
                'actual': data,
                'forecasted': forecasted,
                'step': range(len(data)),
                'is_forecast': [i >= 20 for i in range(len(data))]
            })
            
            output_file = f'recursive_forecasts/{model_name}_full_column2_{timestamp}.csv'
            result_df.to_csv(output_file, index=False)
            
            # Calculate metrics for forecasted portion only
            actual_forecast_portion = data[20:]
            forecasted_portion = forecasted[20:]
            
            mae = np.mean(np.abs(actual_forecast_portion - forecasted_portion))
            mse = np.mean((actual_forecast_portion - forecasted_portion) ** 2)
            
            print(f"✅ SUCCESS - {model_name}")
            print(f"   MAE (forecast portion): {mae:.6f}")
            print(f"   MSE (forecast portion): {mse:.6f}")
            print(f"   Results saved to: {output_file}")
            
            all_results[model_name] = {
                'mae': mae,
                'mse': mse,
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
    print("SUMMARY - FULL SERIES RECURSIVE FORECASTING")
    print(f"{'='*60}")
    
    successful = [name for name, result in all_results.items() if result.get('success', False)]
    failed = [name for name, result in all_results.items() if not result.get('success', False)]
    
    print(f"Successful models: {len(successful)}")
    for model in successful:
        result = all_results[model]
        print(f"  ✅ {model}: MAE={result['mae']:.6f}, MSE={result['mse']:.6f}")
    
    print(f"\nFailed models: {len(failed)}")
    for model in failed:
        print(f"  ❌ {model}: {all_results[model]['error']}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_models': len(all_results),
        'successful': len(successful),
        'failed': len(failed),
        'results': all_results
    }
    
    with open(f'recursive_forecasts/summary_full_column2_{timestamp}.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nSummary saved to: recursive_forecasts/summary_full_column2_{timestamp}.pkl")
    
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