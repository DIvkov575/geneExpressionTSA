import pandas as pd
import numpy as np
from ARIMA_model_v2 import MultiHorizonARIMA
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
    """Evaluate MultiHorizonARIMA for a specific forecast horizon."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < 10: # Arbitrary minimum history
            continue
            
        initial_history = window[:history_size]
        
        actual_target = window[history_size + horizon - 1]
        
        try:
            pred = model.forecast(initial_history, steps=horizon)
            
            predictions.append(pred[-1])
            actuals.append(actual_target)
        except Exception as e:
            continue
    
    if len(actuals) == 0:
        return {'MAPE': np.nan, 'MSE': np.nan, 'MAE': np.nan}
    
    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    
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
    
    # Load and split data
    print(f"Loading data from {FILE_PATH}...")
    windows = load_data(FILE_PATH, WINDOW_SIZE)
    print(f"Total windows: {len(windows)}")
    
    np.random.seed(42)
    np.random.shuffle(windows)
    
    TRAIN_SIZE = int(0.8 * len(windows))
    TEST_SIZE = len(windows) - TRAIN_SIZE
    
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:]
    
    print(f"Train: {len(train_windows)}, Test: {len(test_windows)}")
    
    # Train MultiHorizonARIMA model
    # p=5 lags, d=1 differencing
    print("\nTraining Direct MultiHorizon ARI(p=5, d=1)...")
    model = MultiHorizonARIMA(p=5, d=1)
    model.fit(train_windows, horizons=HORIZONS)
    print("Training complete.")
    model.summary()
    
    # Multi-horizon evaluation
    print("\n" + "="*50)
    print("    MULTI-HORIZON ARIMA V2 (DIRECT) EVALUATION")
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
    
    # Save results
    pd.DataFrame(results).to_csv('arima_v2_results.csv', index=False)
    print("\nResults saved to 'arima_v2_results.csv'")
