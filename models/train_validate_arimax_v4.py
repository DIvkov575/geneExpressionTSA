import pandas as pd
import numpy as np
from ARIMA_model_v4 import MultiHorizonARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")

def load_data(file_path, window_size=25):
    """Load data and create sliding windows."""
    df = pd.read_csv(file_path)
    series_cols = [col for col in df.columns if col != 'time-axis']
    all_windows = []
    
    for col in series_cols:
        series = df[col].values
        if len(series) < window_size:
            continue
        for i in range(len(series) - window_size + 1):
            all_windows.append(series[i : i + window_size])
            
    return np.array(all_windows)

def load_naive_baseline(filepath):
    """Load naive baseline MAE values for MASE calculation."""
    df = pd.read_csv(filepath)
    naive_maes = {}
    for _, row in df.iterrows():
        naive_maes[int(row['horizon'])] = row['naive_mae']
    return naive_maes

def evaluate_horizon(model, test_windows, horizon, naive_mae):
    """Evaluate ARIMAX model for a specific forecast horizon using MASE."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < 10:
            continue
            
        initial_history = window[:history_size]
        actual_future = window[history_size:]
        
        # Create exogenous variables (timestamps)
        hist_X = np.arange(history_size).reshape(-1, 1)
        future_X = np.arange(history_size, history_size + horizon).reshape(-1, 1)
        
        try:
            pred = model.forecast(initial_history, hist_X, future_X, steps=horizon)
        except:
            pred = np.full(horizon, np.nan)
        
        actuals.extend(actual_future)
        predictions.extend(pred)
    
    valid_idx = [i for i, p in enumerate(predictions) if not np.isnan(p)]
    y_true = np.array([actuals[i] for i in valid_idx])
    y_pred = np.array([predictions[i] for i in valid_idx])
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mase = mae / naive_mae
    
    return {'MASE': mase, 'MSE': mse, 'MAE': mae}

if __name__ == "__main__":
    # Configuration
    FILE_PATH = 'data/CRE.csv'
    WINDOW_SIZE = 25
    HORIZONS = [1, 2, 3, 5, 7, 10]
    
    print(f"Loading data from {FILE_PATH}...")
    windows = load_data(FILE_PATH, WINDOW_SIZE)
    
    TRAIN_SIZE = int(0.8 * len(windows))
    TEST_SIZE = len(windows) - TRAIN_SIZE
    
    np.random.seed(42)
    np.random.shuffle(windows)
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:]
    
    # Downsample training data for speed if needed
    MAX_TRAIN_SAMPLES = 2000
    if len(train_windows) > MAX_TRAIN_SAMPLES:
        print(f"Downsampling training data from {len(train_windows)} to {MAX_TRAIN_SAMPLES} windows for faster fitting...")
        train_subset = train_windows[:MAX_TRAIN_SAMPLES]
    else:
        train_subset = train_windows

    # Load naive baseline
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    print("\nLoading naive baseline MAE values...")
    naive_baseline_path = os.path.join(output_dir, 'naive_results.csv')
    naive_maes = load_naive_baseline(naive_baseline_path)

    # Create exogenous variables (timestamps) for training
    print(f"\nPreparing exogenous variables (timestamps)...")
    train_X = [np.arange(len(w)).reshape(-1, 1) for w in train_subset]

    print(f"Training ARIMAX v4 (1,1,1) with exogenous variables on {len(train_subset)} windows...")
    model = MultiHorizonARIMAX(p=1, d=1, q=1, exog_dim=1)
    
    # Fit model
    model.fit(train_subset, train_X, maxiter=500)
    
    params = model.get_params()
    print("\n" + "="*60)
    print(f"ARIMAX v4 Model Summary:")
    print("="*60)
    print(f"Constant: {params['constant']:.6f}")
    print(f"AR Coefficient: {params['ar_coefs']}")
    print(f"MA Coefficient: {params['ma_coefs']}")
    print(f"Exog Coefficient: {params['exog_coefs']}")
    print(f"Variance: {params['sigma2']:.6f}")
    print("="*60)
    
    print("\n" + "="*50)
    print("    ARIMAX v4 MULTI-HORIZON EVALUATION (MASE)")
    print("="*50)
    print(f"{'Horizon':<10} | {'MASE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        metrics = evaluate_horizon(model, test_windows, h, naive_maes[h])
        print(f"{h:<10} | {metrics['MASE']:>9.4f} | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mase': metrics['MASE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE']
        })
    
    print("="*50)
    
    output_path = os.path.join(output_dir, 'arimax_v4_results.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")