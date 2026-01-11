#!/usr/bin/env python3
"""
Multi-horizon evaluation for all models.
Evaluates models at horizons 1, 3, 5, 10, 20 and records MAE scores.
"""

import pandas as pd
import numpy as np
import os
import warnings
import pickle
import joblib
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import models
sys.path.append('..')

# Import from parent directory
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from proper_ts_evaluation import load_time_series_data, temporal_train_test_split
from sklearn.metrics import mean_absolute_error

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

warnings.filterwarnings("ignore")

def naive_forecast(history, horizon=1):
    """Naive forecast - repeat last value."""
    return np.full(horizon, history[-1])

def arima_statsmodels_forecast(history, horizon=1, order=(2,0,2)):
    """ARIMA forecast using statsmodels."""
    try:
        model = ARIMA(history, order=order)
        fitted_model = model.fit(low_memory=True)
        forecast = fitted_model.forecast(steps=horizon)
        return forecast
    except Exception as e:
        print(f"ARIMA statsmodels failed: {e}")
        return naive_forecast(history, horizon)

def arima_v3_forecast(model, history, horizon=1):
    """ARIMA v3 forecast."""
    try:
        model.fit([history], maxiter=50)
        forecast = model.forecast(history, steps=horizon)
        return forecast
    except Exception as e:
        print(f"ARIMA v3 failed: {e}")
        return naive_forecast(history, horizon)

def gbm_forecast(model, history, horizon=1, lookback=30):
    """GBM forecast."""
    try:
        # Prepare features using sliding window
        if len(history) < lookback:
            lookback = len(history)
        
        X = []
        for i in range(lookback, len(history)):
            X.append(history[i-lookback:i])
        
        if len(X) == 0:
            return naive_forecast(history, horizon)
        
        X = np.array(X)
        y = history[lookback:]
        
        # Retrain model
        model.fit(X, y)
        
        # Generate forecasts
        forecasts = []
        current_window = history[-lookback:].copy()
        
        for _ in range(horizon):
            next_pred = model.predict([current_window])[0]
            forecasts.append(next_pred)
            # Update window
            current_window = np.append(current_window[1:], next_pred)
        
        return np.array(forecasts)
    except Exception as e:
        print(f"GBM failed: {e}")
        return naive_forecast(history, horizon)

def prepare_neural_data(series, series_id='eval_series'):
    """Prepare data for neural forecast models."""
    return pd.DataFrame({
        'unique_id': [series_id] * len(series),
        'ds': pd.date_range('2020-01-01', periods=len(series), freq='D'),
        'y': series
    })

def nbeats_forecast(model, history, horizon=1):
    """N-BEATS forecast."""
    try:
        history_df = prepare_neural_data(history)
        forecast = model.predict(history_df)
        
        if 'NBEATS' in forecast.columns:
            return forecast['NBEATS'].values[:horizon]
        else:
            return naive_forecast(history, horizon)
    except Exception as e:
        print(f"N-BEATS failed: {e}")
        return naive_forecast(history, horizon)

def tft_forecast(model, history, horizon=1):
    """TFT forecast."""
    try:
        history_df = prepare_neural_data(history)
        forecast = model.predict(history_df)
        
        if 'TFT' in forecast.columns:
            return forecast['TFT'].values[:horizon]
        else:
            return naive_forecast(history, horizon)
    except Exception as e:
        print(f"TFT failed: {e}")
        return naive_forecast(history, horizon)

def evaluate_model_horizon(train_series, test_series, model_type, model, horizon, lookback=30):
    """Evaluate a single model at a specific horizon."""
    if len(test_series) < horizon:
        return np.nan, 0
    
    actuals = []
    predictions = []
    
    # Use expanding window for evaluation
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    successful_forecasts = 0
    
    # Evaluate every 5th point to speed up, but ensure we get enough samples
    step_size = 5
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        try:
            if model_type == 'naive':
                pred = naive_forecast(history, horizon)
            elif model_type == 'arima_statsmodels':
                pred = arima_statsmodels_forecast(history, horizon)
            elif model_type == 'arima_v3':
                pred = arima_v3_forecast(model, history, horizon)
            elif model_type == 'gbm':
                pred = gbm_forecast(model, history, horizon, lookback)
            elif model_type == 'nbeats':
                pred = nbeats_forecast(model, history, horizon)
            elif model_type == 'tft':
                pred = tft_forecast(model, history, horizon)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Get actual values
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
            
        except Exception as e:
            continue
    
    if len(predictions) == 0:
        return np.nan, 0
    
    # Calculate MAE
    mae = mean_absolute_error(actuals, predictions)
    
    return mae, len(predictions)

def load_trained_models():
    """Load all available trained models."""
    models = {}
    
    # Load naive model (doesn't need actual model file)
    models['naive'] = None
    
    # Load ARIMA statsmodels
    models['arima_statsmodels'] = None  # Uses parameters from training
    
    # Load ARIMA v3
    if os.path.exists('../models/arima_v3.pkl') and ARIMA_V3_AVAILABLE:
        print("Loading ARIMA v3 model...")
        with open('../models/arima_v3.pkl', 'rb') as f:
            models['arima_v3'] = pickle.load(f)
    
    # Load GBM - create new instance for each evaluation
    models['gbm'] = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    # Load NBEATS
    if os.path.exists('../models/nbeats_model.pkl'):
        print("Loading NBEATS model...")
        with open('../models/nbeats_model.pkl', 'rb') as f:
            models['nbeats'] = pickle.load(f)
    
    # Load TFT
    if os.path.exists('../models/tft_model.pkl'):
        print("Loading TFT model...")
        with open('../models/tft_model.pkl', 'rb') as f:
            models['tft'] = pickle.load(f)
    
    return models

def run_multi_horizon_evaluation():
    """Run multi-horizon evaluation for all models."""
    print("="*60)
    print("MULTI-HORIZON MODEL EVALUATION")
    print("="*60)
    
    # Load data
    print("Loading time series data...")
    time_series = load_time_series_data('../data/CRE.csv')
    
    # Use column '1' for evaluation
    column_1_series = {'1': time_series['1']}
    train_data, test_data = temporal_train_test_split(column_1_series, train_ratio=0.8, exclude_last_n=50)
    
    # Load models
    print("Loading trained models...")
    models = load_trained_models()
    
    # Define horizons to evaluate
    horizons = [1, 3, 5, 10, 20]
    
    # Results storage
    results = []
    
    print(f"\nEvaluating {len(models)} models at {len(horizons)} horizons...")
    print(f"Horizons: {horizons}")
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"{'='*50}")
        
        model_results = {
            'model': model_name,
            'horizons': {},
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        for horizon in horizons:
            print(f"  Horizon {horizon}...")
            
            # Create fresh model instance for some models
            if model_name == 'arima_v3' and ARIMA_V3_AVAILABLE:
                eval_model = MultiHorizonARIMA_v3(p=1, d=0, q=1)
            elif model_name == 'gbm':
                eval_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            else:
                eval_model = model
            
            mae, n_preds = evaluate_model_horizon(
                train_data['1'], test_data['1'], model_name, eval_model, horizon
            )
            
            model_results['horizons'][horizon] = {
                'mae': mae,
                'n_predictions': n_preds
            }
            
            if not np.isnan(mae):
                print(f"    MAE: {mae:.6f} (predictions: {n_preds})")
            else:
                print(f"    FAILED - no successful predictions")
            
            # Store individual result
            results.append({
                'model': model_name,
                'horizon': horizon,
                'mae': mae,
                'n_predictions': n_preds,
                'timestamp': model_results['timestamp']
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as CSV
    results_df = pd.DataFrame(results)
    results_file = f'multi_horizon_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Detailed results saved to: {results_file}")
    
    # Create summary table
    print(f"\n{'='*60}")
    print("MULTI-HORIZON EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    # Pivot table for better visualization
    summary_df = results_df.pivot(index='model', columns='horizon', values='mae')
    
    print("\nMAE Scores by Model and Horizon:")
    print("-" * 60)
    print(f"{'Model':<20}", end="")
    for h in horizons:
        print(f"H={h:<10}", end="")
    print()
    print("-" * 60)
    
    for model in summary_df.index:
        print(f"{model:<20}", end="")
        for h in horizons:
            mae_val = summary_df.loc[model, h]
            if not np.isnan(mae_val):
                print(f"{mae_val:<10.6f}", end="")
            else:
                print(f"{'FAILED':<10}", end="")
        print()
    
    # Save summary table
    summary_file = f'mae_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file)
    print(f"\n✅ Summary table saved to: {summary_file}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED")
    print(f"{'='*60}")
    
    return results_df, summary_df

if __name__ == "__main__":
    results_df, summary_df = run_multi_horizon_evaluation()