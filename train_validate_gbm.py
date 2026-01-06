import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
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

def create_lag_features(series, n_lags=10):
    """Create lag features from time series."""
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def evaluate_horizon(model, test_windows, horizon, n_lags=10):
    """Evaluate GBM for specific forecast horizon."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < n_lags + 1:
            continue
            
        history = window[:history_size]
        actual_future = window[history_size:]
        
        # Recursive forecasting
        preds = []
        current_history = list(history)
        
        for _ in range(horizon):
            if len(current_history) >= n_lags:
                X = np.array(current_history[-n_lags:]).reshape(1, -1)
                pred = model.predict(X)[0]
                preds.append(pred)
                current_history.append(pred)
            else:
                preds.append(0)
        
        actuals.extend(actual_future)
        predictions.extend(preds)
    
    mae = mean_absolute_error(actuals, predictions)
    return mae

if __name__ == "__main__":
    print("Training GBM on CRE data...")
    
    # Load data
    windows = load_data('data/CRE.csv')
    
    # Train/test split
    train_size = int(0.8 * len(windows))
    np.random.seed(42)
    np.random.shuffle(windows)
    
    train_windows = windows[:train_size]
    test_windows = windows[train_size:]
    
    # Limit training data for speed
    if len(train_windows) > 2000:
        train_windows = train_windows[:2000]
    
    # Prepare training data
    X_train, y_train = [], []
    for window in train_windows:
        X, y = create_lag_features(window, 10)
        if len(X) > 0:
            X_train.append(X)
            y_train.append(y)
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    results = []
    for horizon in [1, 2, 3, 5, 7, 10]:
        mae = evaluate_horizon(model, test_windows, horizon)
        results.append({'horizon': horizon, 'mae': mae})
        print(f"Horizon {horizon}: MAE = {mae:.6f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/gbm_mae_results.csv', index=False)
    print("Results saved to results/gbm_mae_results.csv")