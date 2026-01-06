import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def load_time_series_data(file_path):
    """Load time series data properly - each column is a separate time series."""
    df = pd.read_csv(file_path)
    series_cols = [col for col in df.columns if col != 'time-axis']
    
    # Return as dictionary of time series
    time_series = {}
    for col in series_cols:
        time_series[col] = df[col].values
    
    return time_series

def temporal_train_test_split(time_series, train_ratio=0.8):
    """Proper temporal split - no data leakage."""
    train_data = {}
    test_data = {}
    
    for series_id, series in time_series.items():
        split_point = int(len(series) * train_ratio)
        train_data[series_id] = series[:split_point]
        test_data[series_id] = series[split_point:]
    
    return train_data, test_data

def create_supervised_data(series, lookback=10, horizon=1):
    """Create supervised learning data from time series."""
    X, y = [], []
    for i in range(lookback, len(series) - horizon + 1):
        X.append(series[i-lookback:i])
        y.append(series[i:i+horizon])
    return np.array(X), np.array(y)

def evaluate_walk_forward(model, train_series, test_series, lookback=10, horizon=1, model_type='sklearn'):
    """Walk-forward evaluation to prevent data leakage."""
    if len(test_series) < lookback + horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    # Use expanding window for training
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    for i in range(train_end, len(full_series) - horizon + 1):
        # Use all data up to current point for training
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        # Prepare training data
        if model_type == 'sklearn':
            X_train, y_train = create_supervised_data(history, lookback, 1)
            if len(X_train) == 0:
                continue
            
            # Fit model on expanding window
            model.fit(X_train, y_train.ravel())
            
            # Make prediction
            X_test = history[-lookback:].reshape(1, -1)
            pred = model.predict(X_test)[0]
            
        elif model_type == 'arima':
            # For ARIMA, fit on recent history and predict
            recent_history = history[-50:] if len(history) > 50 else history
            try:
                model.fit([recent_history], maxiter=100)
                pred = model.forecast(recent_history, steps=1)[0]
            except:
                continue
                
        elif model_type == 'naive':
            pred = history[-1]  # Last value
        
        # Multi-step prediction for horizon > 1
        if horizon > 1:
            current_history = list(history)
            preds = [pred]
            
            for step in range(1, horizon):
                current_history.append(pred)
                if model_type == 'sklearn':
                    X_next = np.array(current_history[-lookback:]).reshape(1, -1)
                    pred = model.predict(X_next)[0]
                elif model_type == 'naive':
                    pred = current_history[-1]
                preds.append(pred)
            
            pred = preds
        else:
            pred = [pred]
        
        # Get actual values
        actual = full_series[i:i+horizon]
        
        actuals.extend(actual)
        predictions.extend(pred)
    
    if len(predictions) == 0:
        return np.nan, np.nan, 0
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    
    # MAPE calculation with protection against division by zero
    mape_values = []
    for actual, pred in zip(actuals, predictions):
        if abs(actual) > 1e-8:  # Avoid division by very small numbers
            mape_values.append(abs((actual - pred) / actual) * 100)
    
    mape = np.mean(mape_values) if mape_values else np.nan
    
    return mae, mape, len(predictions)

def evaluate_multiple_series(model, train_data, test_data, lookback=10, horizon=1, model_type='sklearn'):
    """Evaluate model on multiple time series."""
    all_maes = []
    all_mapes = []
    total_predictions = 0
    
    for series_id in train_data.keys():
        if series_id not in test_data:
            continue
            
        mae, mape, n_preds = evaluate_walk_forward(
            model, train_data[series_id], test_data[series_id], 
            lookback, horizon, model_type
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

# Test the framework
if __name__ == "__main__":
    print("Testing proper time series evaluation framework...")
    
    # Load data
    time_series = load_time_series_data('data/CRE.csv')
    print(f"Loaded {len(time_series)} time series")
    
    # Temporal split
    train_data, test_data = temporal_train_test_split(time_series, train_ratio=0.8)
    
    train_lengths = [len(series) for series in train_data.values()]
    test_lengths = [len(series) for series in test_data.values()]
    
    print(f"Train length: {np.mean(train_lengths):.0f} ± {np.std(train_lengths):.0f}")
    print(f"Test length: {np.mean(test_lengths):.0f} ± {np.std(test_lengths):.0f}")
    
    # Test naive baseline
    mae, mape, n_preds = evaluate_multiple_series(
        None, train_data, test_data, lookback=10, horizon=1, model_type='naive'
    )
    
    print(f"\nNaive baseline (horizon 1):")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Predictions: {n_preds}")
    
    print("\nFramework ready for model evaluation.")