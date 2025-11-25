import pandas as pd
import numpy as np
from ARIMA_model_v3 import MultiHorizonARIMA_v3
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

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

def evaluate_horizon(model, test_windows, horizon):
    """Evaluate ARIMA model for a specific forecast horizon."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < 10:
            continue
            
        initial_history = window[:history_size]
        actual_future = window[history_size:]
        
        try:
            # v3 forecast method signature: forecast(series, steps=1, return_differenced=False)
            pred = model.forecast(initial_history, steps=horizon)
        except:
            pred = np.full(horizon, np.nan)
        
        actuals.extend(actual_future)
        predictions.extend(pred)
    
    valid_idx = [i for i, p in enumerate(predictions) if not np.isnan(p)]
    y_true = np.array([actuals[i] for i in valid_idx])
    y_pred = np.array([predictions[i] for i in valid_idx])
    
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'MAPE': mape, 'MSE': mse, 'MAE': mae}

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

    print(f"Training ARIMA v3 (1,1,1) on {len(train_subset)} windows...")
    # Using same parameters as v1: p=1, d=1, q=1
    model = MultiHorizonARIMA_v3(p=1, d=1, q=1)
    
    # Fit model
    model.fit(train_subset, maxiter=500)
    model.summary()
    
    print("\n" + "="*50)
    print("    ARIMA v3 MULTI-HORIZON EVALUATION")
    print("="*50)
    print(f"{'Horizon':<10} | {'MAPE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        metrics = evaluate_horizon(model, test_windows, h)
        print(f"{h:<10} | {metrics['MAPE']:>9.2f}% | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mape': metrics['MAPE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE']
        })
    
    print("="*50)
    
    pd.DataFrame(results).to_csv('arima_v3_results.csv', index=False)
    print("\nResults saved to 'arima_v3_results.csv'")
