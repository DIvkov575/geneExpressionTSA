import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

def load_single_series(file_path, column='1'):
    """Load a single time series column from the data."""
    df = pd.read_csv(file_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in data")
    return df[column].values

def evaluate_horizon(series, train_size, horizon, order=(1,1,1)):
    """
    Evaluate standard ARIMA for a specific forecast horizon.
    Uses a rolling window approach.
    """
    predictions = []
    actuals = []
    
    # Rolling forecast
    for i in range(train_size, len(series) - horizon + 1):
        train = series[:i]
        actual_future = series[i:i+horizon]
        
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            pred = model_fit.forecast(steps=horizon)
        except:
            pred = np.full(horizon, np.nan)
        
        predictions.extend(pred)
        actuals.extend(actual_future)
        
        if (i - train_size) % 50 == 0 and i > train_size:
            print(f"  Evaluated {i - train_size} windows...")
    
    # Filter NaNs
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
    COLUMN = '1'  # Which column to use
    TRAIN_SIZE = 500  # Initial training size
    HORIZONS = [1, 2, 3, 5, 7, 10]
    ORDER = (1, 1, 1)  # ARIMA(p,d,q)
    
    # Load data
    print(f"Loading column '{COLUMN}' from {FILE_PATH}...")
    series = load_single_series(FILE_PATH, COLUMN)
    print(f"Series length: {len(series)}")
    print(f"Train size: {TRAIN_SIZE}, Test size: {len(series) - TRAIN_SIZE}")
    
    # Multi-horizon evaluation
    print("\n" + "="*50)
    print(f"  STANDARD ARIMA{ORDER} EVALUATION")
    print(f"  (Single Series: Column '{COLUMN}')")
    print("="*50)
    print(f"{'Horizon':<10} | {'MAPE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        print(f"Evaluating {h}-step ahead...")
        metrics = evaluate_horizon(series, TRAIN_SIZE, h, ORDER)
        print(f"{h:<10} | {metrics['MAPE']:>9.2f}% | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mape': metrics['MAPE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE']
        })
    
    print("="*50)
    
    # Save results
    pd.DataFrame(results).to_csv('standard_arima_results.csv', index=False)
    print("\nResults saved to 'standard_arima_results.csv'")
