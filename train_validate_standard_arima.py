import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")

def load_single_series(file_path, column='1'):
    """Load a single time series column from the data."""
    df = pd.read_csv(file_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in data")
    return df[column].values

def evaluate_horizon(model_fit, test, horizon):
    """
    Evaluate a trained ARIMA model on test set for a specific horizon.
    Model is trained ONCE, then used to forecast at different horizons.
    """
    predictions = []
    actuals = []
    
    for i in range(len(test) - horizon + 1):
        actual = test[i + horizon - 1]
        
        try:
            pred = model_fit.forecast(steps=horizon)
            predictions.append(pred[horizon-1])
            actuals.append(actual)
        except:
            continue
    
    if len(predictions) == 0:
        return {'MAPE': np.nan, 'MSE': np.nan, 'MAE': np.nan}
    
    valid_idx = [i for i, p in enumerate(predictions) if not np.isnan(p) and np.isfinite(p)]
    
    if len(valid_idx) == 0:
        return {'MAPE': np.nan, 'MSE': np.nan, 'MAE': np.nan}
    
    y_true = np.array([actuals[i] for i in valid_idx])
    y_pred = np.array([predictions[i] for i in valid_idx])
    
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'MAPE': mape, 'MSE': mse, 'MAE': mae}

if __name__ == "__main__":
    FILE_PATH = 'data/CRE.csv'
    COLUMN = '1'
    TRAIN_RATIO = 0.8
    HORIZONS = [1, 2, 3, 5, 7, 10]
    ORDER = (1, 1, 1)
    
    series = load_single_series(FILE_PATH, COLUMN)
    
    train_size = int(len(series) * TRAIN_RATIO)
    train = series[:train_size]
    test = series[train_size:]
    
    print(f"Series length: {len(series)}")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    model = ARIMA(train, order=ORDER)
    model_fit = model.fit()
    print("Training complete.")
    
    print("\n" + "="*50)
    print(f"  STANDARD ARIMA{ORDER} EVALUATION")
    print(f"  (Single Series: Column '{COLUMN}')")
    print("="*50)
    print(f"{'Horizon':<10} | {'MAPE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        metrics = evaluate_horizon(model_fit, test, h)
        print(f"{h:<10} | {metrics['MAPE']:>9.2f}% | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mape': metrics['MAPE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE']
        })
    
    print("="*50)
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'standard_arima_results.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")