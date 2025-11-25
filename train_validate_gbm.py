import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    GBM_LIBRARY = "LightGBM"
except ImportError:
    try:
        from xgboost import XGBRegressor as LGBMRegressor
        GBM_LIBRARY = "XGBoost"
    except ImportError:
        raise ImportError("Please install either lightgbm or xgboost: pip install lightgbm")


class GradientBoostedForecaster:
    """
    Gradient Boosted forecasting model using sliding window features.
    Creates lag features from historical data to predict future values.
    """
    def __init__(self, n_lags=10, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
        """
        Parameters:
        -----------
        n_lags : int
            Number of lag features to create
        n_estimators : int
            Number of boosting iterations
        learning_rate : float
            Learning rate for gradient boosting
        max_depth : int
            Maximum tree depth
        random_state : int
            Random seed for reproducibility
        """
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = {}  # Store one model per forecast step
        self.fitted_ = False
    
    def create_features(self, history, horizon=1):
        """
        Create lag features from historical time series.
        
        Parameters:
        -----------
        history : array-like
            Historical time series values
        horizon : int
            Forecast horizon (used to determine target position)
            
        Returns:
        --------
        X : ndarray
            Feature matrix with lag features
        y : ndarray (optional)
            Target values if enough data available
        """
        history = np.asarray(history, dtype=float)
        
        # Create lag features
        features = []
        for i in range(self.n_lags, len(history) - horizon + 1):
            lag_features = history[i - self.n_lags:i]
            features.append(lag_features)
        
        if len(features) == 0:
            return None, None
        
        X = np.array(features)
        
        # Create targets (values at horizon steps ahead)
        if len(history) >= self.n_lags + horizon:
            y = history[self.n_lags + horizon - 1:]
            # Ensure X and y have same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
        else:
            y = None
        
        return X, y
    
    def fit(self, series_list, max_horizon=10):
        """
        Fit separate models for each forecast horizon.
        
        Parameters:
        -----------
        series_list : list of arrays
            List of time series windows for training
        max_horizon : int
            Maximum forecast horizon to train for
        """
        print(f"Training {GBM_LIBRARY} models for horizons 1 to {max_horizon}...")
        
        for h in range(1, max_horizon + 1):
            X_all, y_all = [], []
            
            # Create features from all training windows
            for series in series_list:
                X, y = self.create_features(series, horizon=h)
                if X is not None and y is not None and len(X) > 0:
                    X_all.append(X)
                    y_all.append(y)
            
            if len(X_all) == 0:
                print(f"  Warning: No valid training data for horizon {h}")
                continue
            
            X_train = np.vstack(X_all)
            y_train = np.concatenate(y_all)
            
            # Train model for this horizon
            model = LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            self.models[h] = model
            
            if h % 2 == 0 or h == 1:
                print(f"  Trained model for horizon {h} with {len(X_train)} samples")
        
        self.fitted_ = True
        return self
    
    def forecast(self, history, steps=1):
        """
        Forecast future values using trained models.
        
        Parameters:
        -----------
        history : array-like
            Historical time series values
        steps : int
            Number of steps ahead to forecast
            
        Returns:
        --------
        forecasts : ndarray
            Forecasted values
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before forecasting")
        
        history = np.asarray(history, dtype=float)
        forecasts = []
        
        for step in range(1, steps + 1):
            if step not in self.models:
                # If no model for this horizon, use the last available model
                available_horizons = sorted(self.models.keys())
                if len(available_horizons) == 0:
                    raise ValueError("No trained models available")
                step_to_use = available_horizons[-1]
            else:
                step_to_use = step
            
            # Create features from the most recent n_lags values
            if len(history) >= self.n_lags:
                features = history[-self.n_lags:].reshape(1, -1)
                pred = self.models[step_to_use].predict(features)[0]
            else:
                # Not enough history, use mean of available history
                pred = np.mean(history)
            
            forecasts.append(pred)
        
        return np.array(forecasts)
    
    def summary(self):
        """Print model summary."""
        print("Gradient Boosted Forecasting Model")
        print("="*50)
        print(f"Library: {GBM_LIBRARY}")
        print(f"Number of lag features: {self.n_lags}")
        print(f"Number of estimators: {self.n_estimators}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Max depth: {self.max_depth}")
        print(f"Trained horizons: {sorted(self.models.keys())}")
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
    """Evaluate GBM model for a specific forecast horizon."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < model.n_lags + 1:  # Need enough history for lag features
            continue
            
        initial_history = window[:history_size]
        actual_future = window[history_size:]
        
        try:
            pred = model.forecast(initial_history, steps=horizon)
            actuals.extend(actual_future)
            predictions.extend(pred)
        except Exception as e:
            # Skip windows that cause errors
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
    TRAIN_SIZE = 1000
    TEST_SIZE = 3000
    HORIZONS = [1, 2, 3, 5, 7, 10]
    
    # Model hyperparameters
    N_LAGS = 10  # Number of lag features
    N_ESTIMATORS = 100  # Number of boosting rounds
    LEARNING_RATE = 0.1
    MAX_DEPTH = 5
    
    # Load and split data
    print(f"Loading data from {FILE_PATH}...")
    windows = load_data(FILE_PATH, WINDOW_SIZE)
    print(f"Total windows: {len(windows)}")
    
    np.random.seed(42)
    np.random.shuffle(windows)
    
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    
    print(f"Train: {len(train_windows)}, Test: {len(test_windows)}")
    
    # Train GBM model
    print(f"\nTraining Gradient Boosted Forecaster...")
    model = GradientBoostedForecaster(
        n_lags=N_LAGS,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    model.fit(train_windows, max_horizon=max(HORIZONS))
    print("Training complete.")
    model.summary()
    
    # Multi-horizon evaluation
    print("\n" + "="*50)
    print("    GRADIENT BOOSTED MULTI-HORIZON EVALUATION")
    print("="*50)
    print(f"{'Horizon':<10} | {'MAPE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        print(f"Evaluating horizon {h}...")
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
    pd.DataFrame(results).to_csv('gbm_results.csv', index=False)
    print("\nResults saved to 'gbm_results.csv'")
