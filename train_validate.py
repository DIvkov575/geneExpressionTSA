import pandas as pd
import numpy as np
import random
from ARIMA_model import MultiSeriesARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path, window_size=25):
    """Load data and create sliding windows."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    series_cols = [col for col in df.columns if col != 'time-axis']
    all_windows = []
    
    for col in series_cols:
        series = df[col].values
        if len(series) < window_size:
            continue
            
        for i in range(len(series) - window_size + 1):
            window = series[i : i + window_size]
            all_windows.append(window)
            
    return np.array(all_windows)

def naive_forecast(history):
    """Naive baseline: Predict the last observed value."""
    return history[-1]

def train_and_evaluate(train_windows, test_windows):
    """Train ARIMA on train_windows, then evaluate ARIMA and Naive on test_windows."""
    # Train ARIMA
    print(f"\nTraining ARIMA on {len(train_windows)} samples...")
    arima_model = MultiSeriesARIMA(p=1, d=1, q=1)
    arima_model.fit(train_windows)
    print("ARIMA training complete.")
    arima_model.summary()
    
    # Evaluate on Test Set
    print(f"\nEvaluating on {len(test_windows)} test samples...")
    
    actuals = []
    arima_preds = []
    naive_preds = []
    
    for i, window in enumerate(test_windows):
        history = window[:-1]
        actual = window[-1]
        
        try:
            pred_arima = arima_model.forecast(history, steps=1)[0]
        except:
            pred_arima = np.nan
            
        pred_naive = naive_forecast(history)
        
        actuals.append(actual)
        arima_preds.append(pred_arima)
        naive_preds.append(pred_naive)
        
        if i % 5000 == 0 and i > 0:
            print(f"Evaluated {i}/{len(test_windows)} samples...")

    # Filter NaNs
    valid_indices = [i for i, p in enumerate(arima_preds) if not np.isnan(p)]
    
    y_true = np.array([actuals[i] for i in valid_indices])
    y_arima = np.array([arima_preds[i] for i in valid_indices])
    y_naive = np.array([naive_preds[i] for i in valid_indices])
    
    # Compute Metrics
    metrics = {}
    
    # Absolute Metrics
    metrics['ARIMA_MSE'] = mean_squared_error(y_true, y_arima)
    metrics['ARIMA_RMSE'] = np.sqrt(metrics['ARIMA_MSE'])
    metrics['ARIMA_MAE'] = mean_absolute_error(y_true, y_arima)
    
    metrics['Naive_MSE'] = mean_squared_error(y_true, y_naive)
    metrics['Naive_RMSE'] = np.sqrt(metrics['Naive_MSE'])
    metrics['Naive_MAE'] = mean_absolute_error(y_true, y_naive)
    
    # Normalized Metrics
    epsilon = 1e-10
    metrics['ARIMA_MAPE'] = np.mean(np.abs((y_true - y_arima) / (np.abs(y_true) + epsilon))) * 100
    metrics['Naive_MAPE'] = np.mean(np.abs((y_true - y_naive) / (np.abs(y_true) + epsilon))) * 100
    
    y_range = np.max(y_true) - np.min(y_true)
    if y_range > 0:
        metrics['ARIMA_NRMSE'] = metrics['ARIMA_RMSE'] / y_range * 100
        metrics['Naive_NRMSE'] = metrics['Naive_RMSE'] / y_range * 100
    else:
        metrics['ARIMA_NRMSE'] = 0
        metrics['Naive_NRMSE'] = 0
    
    return metrics

if __name__ == "__main__":
    FILE_PATH = 'data/CRE.csv'
    WINDOW_SIZE = 25
    TRAIN_SIZE = 1000
    
    # Load Data
    windows = load_data(FILE_PATH, WINDOW_SIZE)
    print(f"Total windows: {len(windows)}")
    
    # Shuffle and Split
    np.random.seed(42)
    np.random.shuffle(windows)
    
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:]
    
    print(f"Train set size: {len(train_windows)}")
    print(f"Test set size: {len(test_windows)}")
    
    # Run Evaluation
    results = train_and_evaluate(train_windows, test_windows)
    
    # Print Results
    print("\n" + "="*50)
    print("         MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"{'Metric':<15} | {'ARIMA':<12} | {'Naive':<12}")
    print("-" * 50)
    print("\nNormalized Metrics (%):")
    print(f"{'MAPE':<15} | {results['ARIMA_MAPE']:>11.2f}% | {results['Naive_MAPE']:>11.2f}%")
    print(f"{'NRMSE':<15} | {results['ARIMA_NRMSE']:>11.2f}% | {results['Naive_NRMSE']:>11.2f}%")
    print("\nAbsolute Metrics:")
    print(f"{'MSE':<15} | {results['ARIMA_MSE']:>12.6f} | {results['Naive_MSE']:>12.6f}")
    print(f"{'RMSE':<15} | {results['ARIMA_RMSE']:>12.6f} | {results['Naive_RMSE']:>12.6f}")
    print(f"{'MAE':<15} | {results['ARIMA_MAE']:>12.6f} | {results['Naive_MAE']:>12.6f}")
    print("="*50)
    
    print("\nKey Findings:")
    improvement = ((results['Naive_MAPE'] - results['ARIMA_MAPE']) / results['Naive_MAPE']) * 100
    print(f"- ARIMA achieves {results['ARIMA_MAPE']:.2f}% MAPE")
    print(f"- {improvement:.1f}% improvement over Naive baseline")
