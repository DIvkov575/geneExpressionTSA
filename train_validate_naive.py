import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import pickle
import argparse

def load_data(file_path):
    """Load data and create sliding windows."""
    df = pd.read_csv(file_path)
    series_cols = [col for col in df.columns if col != 'time-axis']
    all_windows = []
    
    for col in series_cols:
        series = df[col].values
        for i in range(len(series) - 25 + 1):
            all_windows.append(series[i:i + 25])
    
    return np.array(all_windows)

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
    
    # Load data
    windows = load_data('data/CRE.csv')
    
    # Train/test split
    train_size = int(0.8 * len(windows))
    np.random.seed(42)
    np.random.shuffle(windows)
    
    test_windows = windows[train_size:]
    
    # Evaluate multiple horizons
    results = []
    for horizon in [1, 2, 3, 5, 7, 10]:
        mae = evaluate_horizon(test_windows, horizon)
        results.append({'horizon': horizon, 'naive_mae': mae})
        print(f"Horizon {horizon}: MAE = {mae:.6f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/naive_mae_results.csv', index=False)
    print("Results saved to results/naive_mae_results.csv")
    
    # Save model weights if requested
    if args.save_weights:
        model_params = {
            'model_type': 'naive',
            'results': results,
            'train_size': train_size,
            'random_seed': 42
        }
        save_naive_model(model_params, args.model_path)