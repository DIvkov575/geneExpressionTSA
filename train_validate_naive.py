import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import pickle
import argparse
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split

def naive_forecast(history, steps):
    """Simple naive forecast - repeat last value."""
    return np.full(steps, history[-1])

def evaluate_horizon(test_windows, horizon):
    """Evaluate naive forecasting for specific horizon."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < 10:
            continue
            
        history = window[:history_size]
        actual_future = window[history_size:]
        
        pred = naive_forecast(history, horizon)
        
        actuals.extend(actual_future)
        predictions.extend(pred)
    
    mae = mean_absolute_error(actuals, predictions)
    return mae

def save_naive_model(model_params, filepath):
    """Save naive model parameters to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)
    print(f"Naive model parameters saved to {filepath}")

def load_naive_model(filepath):
    """Load naive model parameters from disk."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            model_params = pickle.load(f)
        print(f"Naive model parameters loaded from {filepath}")
        return model_params
    else:
        print(f"Naive model file {filepath} not found")
        return None

def evaluate_naive_walk_forward(train_series, test_series, horizon=1):
    """Walk-forward evaluation for naive forecasting."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    # Use expanding window for evaluation
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    successful_forecasts = 0
    step_size = 5
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        
        if len(history) < 10:
            continue
            
        try:
            # Naive forecast - use last value
            pred = naive_forecast(history, horizon)
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
            
        except Exception:
            continue
    
    if len(predictions) == 0:
        return np.nan, np.nan, 0
    
    mae = mean_absolute_error(actuals, predictions)
    
    mape_values = []
    for actual, pred in zip(actuals, predictions):
        if abs(actual) > 1e-8:
            mape_values.append(abs((actual - pred) / actual) * 100)
    
    mape = np.mean(mape_values) if mape_values else np.nan
    
    return mae, mape, len(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate naive model for time series forecasting')
    parser.add_argument('--save-weights', action='store_true', 
                        help='Save trained model weights to disk')
    parser.add_argument('--load-weights', action='store_true',
                        help='Load trained model weights from disk (skip training)')
    parser.add_argument('--model-path', type=str, default='models/naive_model.pkl',
                        help='Path to save/load model weights (default: models/naive_model.pkl)')
    
    args = parser.parse_args()
    
    print("Naive forecasting evaluation...")
    
    # Load data with proper temporal structure  
    time_series = load_time_series_data('data/CRE.csv')
    
    # For non-windowing models: only train on column '1' with last 50 points excluded
    column_1_series = {'1': time_series['1']}
    train_data, test_data = temporal_train_test_split(column_1_series, train_ratio=0.8, exclude_last_n=50)
    
    # Evaluate multiple horizons
    results = []
    for horizon in [1, 2, 3, 5, 7, 10]:
        mae, mape, n_preds = evaluate_naive_walk_forward(
            train_data['1'], test_data['1'], horizon
        )
        results.append({
            'horizon': horizon, 
            'mae': mae,
            'mape': mape,
            'n_predictions': n_preds
        })
        print(f"Horizon {horizon}: MAE = {mae:.6f}, MAPE = {mape:.2f}%, Preds = {n_preds}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/naive_mae_results.csv', index=False)
    print("Results saved to results/naive_mae_results.csv")
    
    # Save model weights if requested
    if args.save_weights:
        model_params = {
            'model_type': 'naive',
            'results': results
        }
        save_naive_model(model_params, args.model_path)