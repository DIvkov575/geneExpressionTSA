import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
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

def difference(series):
    """Apply first order differencing."""
    return np.diff(series)

def inverse_difference(last_val, diff_forecast):
    """Integrate differenced forecast back to original scale."""
    return last_val + np.cumsum(diff_forecast)

def create_lag_features(series, n_lags=10):
    """Create lag features from time series."""
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def evaluate_horizon(model, test_windows, horizon, n_lags=10):
    """Evaluate GBM for a specific forecast horizon using recursive prediction on differenced data."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < n_lags + 1: # +1 because differencing reduces length by 1
            continue
            
        # Get history and actual future in original scale
        history_orig = window[:history_size]
        actual_future = window[history_size:]
        
        # Difference the history
        history_diff = list(difference(history_orig))
        last_val = history_orig[-1]
        
        # Recursive forecasting in differenced space
        preds_diff = []
        current_history_diff = history_diff[:]
        
        for _ in range(horizon):
            if len(current_history_diff) >= n_lags:
                X = np.array(current_history_diff[-n_lags:]).reshape(1, -1)
                pred_diff = model.predict(X)[0]
                preds_diff.append(pred_diff)
                current_history_diff.append(pred_diff)
            else:
                # Fallback if not enough history (shouldn't happen with check above)
                preds_diff.append(0)
                current_history_diff.append(0)
        
        # Inverse difference to get predictions in original scale
        preds_orig = inverse_difference(last_val, preds_diff)
        
        actuals.extend(actual_future)
        predictions.extend(preds_orig)
    
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
    N_LAGS = 10
    
    # Load and split data
    print(f"Loading data from {FILE_PATH}...")
    windows = load_data(FILE_PATH, WINDOW_SIZE)
    
    TRAIN_SIZE = int(0.8 * len(windows))
    
    np.random.seed(42)
    np.random.shuffle(windows)
    
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:]
    
    # Downsample for speed
    MAX_TRAIN = 2000
    if len(train_windows) > MAX_TRAIN:
        train_windows = train_windows[:MAX_TRAIN]
    
    print(f"Train windows: {len(train_windows)}")
    
    # Prepare training data (Differenced)
    print("\nPreparing training data (with differencing)...")
    X_train, y_train = [], []
    for window in train_windows:
        # Difference the window
        diff_window = difference(window)
        
        # Create lags from differenced data
        X, y = create_lag_features(diff_window, N_LAGS)
        if len(X) > 0:
            X_train.append(X)
            y_train.append(y)
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print(f"Training samples: {len(X_train)}")
    
    # Train GBM model
    print("\nTraining Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    
    # Multi-horizon evaluation
    print("\n" + "="*50)
    print("    GBM (DIFFERENCED) EVALUATION")
    print("="*50)
    print(f"{'Horizon':<10} | {'MAPE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        metrics = evaluate_horizon(model, test_windows, h, N_LAGS)
        print(f"{h:<10} | {metrics['MAPE']:>9.2f}% | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mape': metrics['MAPE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE']
        })
    
    print("="*50)
