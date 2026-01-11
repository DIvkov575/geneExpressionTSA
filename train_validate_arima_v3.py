import pandas as pd
import numpy as np
import os
import warnings
import argparse
import pickle
from pathlib import Path
from models.ARIMA_model_v3 import MultiHorizonARIMA_v3
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split
from sklearn.metrics import mean_absolute_error

# Suppress ARIMA warnings for cleaner output
warnings.filterwarnings("ignore")

def pure_arima_forecast(model, history, horizon=1, max_retries=3):
    """Pure ARIMA forecasting without sanity checks or fallbacks."""
    
    # Try different window sizes if forecast fails
    window_sizes = [min(30, len(history)), min(20, len(history)), min(15, len(history))]
    
    for attempt in range(max_retries):
        for window_size in window_sizes:
            if window_size < 10:
                continue
                
            try:
                # Use smaller window for faster convergence
                recent_history = history[-window_size:] if len(history) > window_size else history
                
                # Fit with fewer iterations
                model.fit([recent_history], maxiter=50)
                
                # Make forecast - no sanity checks
                forecast = model.forecast(recent_history, steps=horizon)
                
                # Only check for NaN/inf - no other sanity checks
                if not (np.any(np.isnan(forecast)) or np.any(np.isinf(forecast))):
                    return forecast
                    
            except Exception as e:
                continue
    
    # If all attempts fail, raise exception instead of fallback
    raise RuntimeError(f"ARIMA failed to converge for horizon {horizon}")

def evaluate_arima_walk_forward(train_series, test_series, p=1, d=0, q=1, lookback=15, horizon=1):
    """Walk-forward evaluation for pure ARIMA."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    # Use expanding window for evaluation
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    # Create model once
    model = MultiHorizonARIMA_v3(p=p, d=d, q=q)
    
    successful_forecasts = 0
    failed_forecasts = 0
    
    # Evaluate every 5th point to speed up
    step_size = 5
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        try:
            # Make prediction with pure ARIMA (no fallbacks)
            pred = pure_arima_forecast(model, history, horizon)
            
            # Get actual values
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
            
        except Exception as e:
            # Count failures but don't add anything to results
            failed_forecasts += 1
            continue
    
    print(f"    Successful: {successful_forecasts}, Failed: {failed_forecasts}")
    
    if len(predictions) == 0:
        return np.nan, np.nan, 0
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    
    # MAPE calculation
    mape_values = []
    for actual, pred in zip(actuals, predictions):
        if abs(actual) > 1e-8:
            mape_values.append(abs((actual - pred) / actual) * 100)
    
    mape = np.mean(mape_values) if mape_values else np.nan
    
    return mae, mape, len(predictions)

def evaluate_multiple_series_arima(train_data, test_data, p=1, d=0, q=1, lookback=15, horizon=1, max_series=10):
    """Evaluate pure ARIMA on multiple time series."""
    all_maes = []
    all_mapes = []
    total_predictions = 0
    
    # Limit to first N series for speed
    series_ids = list(train_data.keys())[:max_series]
    
    for i, series_id in enumerate(series_ids):
        if series_id not in test_data:
            continue
            
        print(f"  Processing series {i+1}/{len(series_ids)}...")
        
        mae, mape, n_preds = evaluate_arima_walk_forward(
            train_data[series_id], test_data[series_id], p, d, q, lookback, horizon
        )
        
        if not np.isnan(mae):
            all_maes.append(mae)
            total_predictions += n_preds
            
        if not np.isnan(mape):
            all_mapes.append(mape)
    
    if len(all_maes) == 0:
        return np.nan, np.nan, 0
    
    avg_mae = np.mean(all_maes)
    avg_mape = np.mean(all_mapes) if all_mapes else np.nan
    
    return avg_mae, avg_mape, total_predictions

def save_arima_v3_model(model, filepath):
    """Save ARIMA v3 model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"ARIMA v3 model saved to {filepath}")

def load_arima_v3_model(filepath):
    """Load ARIMA v3 model from disk."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"ARIMA v3 model loaded from {filepath}")
        return model
    else:
        print(f"ARIMA v3 model file {filepath} not found")
        return None

def run_arima_evaluation(save_weights=False, load_weights=False, model_path="models/arima_v3.pkl"):
    """Run pure ARIMA evaluation without fallbacks."""
    print("Running ARIMA(1,0,1) WITHOUT sanity checks or fallbacks...")
    
    # Load data with proper temporal structure
    time_series = load_time_series_data('data/CRE.csv')
    
    # For non-windowing models: only train on column '1' with last 50 points excluded
    column_1_series = {'1': time_series['1']}
    train_data, test_data = temporal_train_test_split(column_1_series, train_ratio=0.8, exclude_last_n=50)
    
    results = []
    
    # Test all horizons
    horizons = [1, 2, 3, 5, 7, 10]
    
    for horizon in horizons:
        print(f"Evaluating horizon {horizon}...")
        
        # Use fewer series for longer horizons to avoid timeouts
        max_series = 10 if horizon <= 5 else 5
        
        mae, mape, n_preds = evaluate_multiple_series_arima(
            train_data, test_data, p=1, d=0, q=1, 
            lookback=15, horizon=horizon, max_series=max_series
        )
        
        results.append({
            'horizon': horizon,
            'mae': mae,
            'mape': mape,
            'n_predictions': n_preds
        })
        
        if not np.isnan(mae):
            print(f"Horizon {horizon}: MAE={mae:.6f}, MAPE={mape:.2f}%, Preds={n_preds}")
        else:
            print(f"Horizon {horizon}: FAILED - no successful predictions")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/arima_mae_results.csv', index=False)
    print("Results saved to results/arima_mae_results.csv")
    
    # Save model weights if requested
    if save_weights:
        model = MultiHorizonARIMA_v3(p=1, d=0, q=1)
        save_arima_v3_model(model, model_path)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate ARIMA v3 for time series forecasting')
    parser.add_argument('--save-weights', action='store_true', 
                        help='Save trained model weights to disk')
    parser.add_argument('--load-weights', action='store_true',
                        help='Load trained model weights from disk (skip training)')
    parser.add_argument('--model-path', type=str, default='models/arima_v3.pkl',
                        help='Path to save/load model weights (default: models/arima_v3.pkl)')
    
    args = parser.parse_args()
    
    results = run_arima_evaluation(
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        model_path=args.model_path
    )