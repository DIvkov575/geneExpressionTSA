import pandas as pd
import numpy as np
from ARIMA_model_v4 import MultiHorizonARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path, window_size=25):
    """Load data and create sliding windows with timestamps."""
    df = pd.read_csv(file_path)
    series_cols = [col for col in df.columns if col != 'time-axis']
    
    # Extract timestamps
    timestamps = df['time-axis'].values
    
    all_windows = []
    all_timestamps = []
    
    for col in series_cols:
        series = df[col].values
        if len(series) < window_size:
            continue
        for i in range(len(series) - window_size + 1):
            all_windows.append(series[i : i + window_size])
            all_timestamps.append(timestamps[i : i + window_size])
            
    return np.array(all_windows), np.array(all_timestamps)

def evaluate_horizon(model, test_windows, test_timestamps, horizon):
    """Evaluate ARIMAX model for a specific forecast horizon."""
    actuals, predictions = [], []
    
    for idx, window in enumerate(test_windows):
        history_size = len(window) - horizon
        if history_size < 10:
            continue
            
        initial_history = window[:history_size]
        actual_future = window[history_size:]
        
        # Get corresponding timestamps
        timestamps = test_timestamps[idx]
        timestamp_history = timestamps[:history_size].reshape(-1, 1)
        timestamp_future = timestamps[history_size:history_size+horizon].reshape(-1, 1)
        
        try:
            pred = model.forecast(
                initial_history, 
                exog_history=timestamp_history,
                exog_future=timestamp_future,
                steps=horizon
            )
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
    FILE_PATH = 'data/CRE.csv'
    WINDOW_SIZE = 25
    HORIZONS = [1, 2, 3, 5, 7, 10]
    
    windows, timestamps = load_data(FILE_PATH, WINDOW_SIZE)
    
    TRAIN_SIZE = int(0.8 * len(windows))
    TEST_SIZE = len(windows) - TRAIN_SIZE
    
    np.random.seed(42)
    indices = np.arange(len(windows))
    np.random.shuffle(indices)
    
    train_windows = windows[indices[:TRAIN_SIZE]]
    train_timestamps = timestamps[indices[:TRAIN_SIZE]]
    test_windows = windows[indices[TRAIN_SIZE:]]
    test_timestamps = timestamps[indices[TRAIN_SIZE:]]
    
    # MAX_TRAIN_SAMPLES = 2000
    # if len(train_windows) > MAX_TRAIN_SAMPLES:
    #     train_windows = train_windows[:MAX_TRAIN_SAMPLES]
    #     train_timestamps = train_timestamps[:MAX_TRAIN_SAMPLES]
    
    train_exog_list = [ts.reshape(-1, 1) for ts in train_timestamps]
    
    model = MultiHorizonARIMAX(p=1, d=1, q=1, exog_dim=1)
    model.fit(train_windows.tolist(), exog_list=train_exog_list, maxiter=500)
    
    params = model.get_params()
    print(f"\nModel fitted successfully:")
    print(f"  AR coefficient: {params['ar_coefs'][0]:.4f}")
    print(f"  MA coefficient: {params['ma_coefs'][0]:.4f}")
    print(f"  Timestamp coefficient (beta): {params['exog_coefs'][0]:.6f}")
    
    print("\n" + "="*50)
    print("    ARIMAX v4 MULTI-HORIZON EVALUATION")
    print("="*50)
    print(f"{'Horizon':<10} | {'MAPE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        metrics = evaluate_horizon(model, test_windows, test_timestamps, h)
        print(f"{h:<10} | {metrics['MAPE']:>9.2f}% | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mape': metrics['MAPE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE']
        })
    
    print("="*50)
    
    pd.DataFrame(results).to_csv('arimax_v4_results.csv', index=False)
    print("\nResults saved to 'arimax_v4_results.csv'")
