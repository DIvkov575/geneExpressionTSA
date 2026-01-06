import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os

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

if __name__ == "__main__":
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