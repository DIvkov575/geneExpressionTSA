import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")

class NaiveForecaster:
    """
    Naive forecasting model that predicts the last observed value.
    For multi-step forecasting, recursively uses its own predictions.
    """
    def __init__(self):
        self.fitted_ = False
    
    def fit(self, series_list):
        """Fit method for API consistency (Naive doesn't need training)."""
        self.fitted_ = True
        return self
    
    def forecast(self, history, steps=1):
        """
        Forecast future values by repeating the last observed value.
        
        Parameters:
        -----------
        history : array-like
            Historical time series values
        steps : int
            Number of steps ahead to forecast
            
        Returns:
        --------
        forecasts : ndarray
            Forecasted values (all equal to last observed value)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before forecasting")
        
        history = np.asarray(history, dtype=float)
        last_value = history[-1]
        
        # For naive, all future predictions are the last observed value
        forecasts = np.full(steps, last_value)
        
        return forecasts
    
    def summary(self):
        """Print model summary."""
        print("Naive Forecasting Model")
        print("="*50)
        print("Strategy: Predict last observed value")
        print("Parameters: None (non-parametric)")
        print("="*50)

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
    """Evaluate Naive model for a specific forecast horizon."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < 10:
            continue
            
        initial_history = window[:history_size]
        actual_future = window[history_size:]
        
        pred = model.forecast(initial_history, steps=horizon)
        
        actuals.extend(actual_future)
        predictions.extend(pred)
    
    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mase = 1.0  # By definition, naive baseline has MASE = 1.0
    
    return {'MASE': mase, 'MSE': mse, 'MAE': mae, 'Naive_MAE': mae}

if __name__ == "__main__":
    # Configuration
    FILE_PATH = 'data/CRE.csv'
    WINDOW_SIZE = 25
    TRAIN_SIZE = 1000
    TEST_SIZE = 3000
    HORIZONS = [1, 2, 3, 5, 7, 10]
    
    # Load and split data
    print(f"Loading data from {FILE_PATH}...")
    windows = load_data(FILE_PATH, WINDOW_SIZE)
    print(f"Total windows: {len(windows)}")
    
    np.random.seed(42)
    np.random.shuffle(windows)
    
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    
    print(f"Train: {len(train_windows)}, Test: {len(test_windows)}")
    
    # Train Naive model
    print("\nTraining Naive Forecaster...")
    model = NaiveForecaster()
    model.fit(train_windows)
    print("Training complete.")
    model.summary()
    
    # Multi-horizon evaluation
    print("\n" + "="*50)
    print("    NAIVE MULTI-HORIZON EVALUATION (MASE)")
    print("="*50)
    print(f"{'Horizon':<10} | {'MASE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        metrics = evaluate_horizon(model, test_windows, h)
        print(f"{h:<10} | {metrics['MASE']:>9.2f} | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mase': metrics['MASE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE'],
            'naive_mae': metrics['Naive_MAE']
        })
    
    print("="*50)
    
    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'naive_results.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")