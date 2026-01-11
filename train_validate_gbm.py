import pandas as pd
import numpy as np
import os
import warnings
import argparse
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split

warnings.filterwarnings("ignore")

def create_advanced_features(series, window_size=15):
    """Create enhanced time series features for GBM."""
    n = len(series)
    if n < window_size + 5:
        return None
        
    features = []
    
    # Lag features (multiple lags)
    lags = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20]
    lag_features = []
    for lag in lags:
        if n > lag:
            lag_features.append(series[-lag])
        else:
            lag_features.append(series[-1])
    
    features.extend(lag_features)
    
    # Multiple window rolling statistics
    windows = [5, 10, 15, 20]
    for w in windows:
        if n >= w:
            window_data = series[-w:]
            features.extend([
                np.mean(window_data),
                np.std(window_data),
                np.min(window_data),
                np.max(window_data),
                np.median(window_data)
            ])
        else:
            # Fallback for shorter series
            features.extend([series[-1], 0, series[-1], series[-1], series[-1]])
    
    # Current value vs historical stats
    recent = series[-window_size:]
    features.extend([
        recent[-1] - np.mean(recent),  # Deviation from mean
        (recent[-1] - recent[0]) / window_size,  # Trend
        (recent[-1] - np.min(recent)) / (np.max(recent) - np.min(recent) + 1e-8),  # Normalized position
    ])
    
    # Enhanced differencing features
    if n > 2:
        diffs = np.diff(series)
        recent_diffs = diffs[-min(10, len(diffs)):]
        features.extend([
            series[-1] - series[-2],   # First difference
            np.mean(recent_diffs),     # Average difference
            np.std(recent_diffs),      # Std of differences
            np.sum(recent_diffs > 0) / len(recent_diffs),  # Proportion of positive changes
        ])
    else:
        features.extend([0, 0, 0, 0.5])
    
    # Seasonal and cyclical features
    if n > 7:
        features.append(series[-7])  # Weekly lag
    else:
        features.append(series[-1])
        
    if n > 14:
        features.extend([
            np.mean(series[-14:]),  # 2-week average
            np.corrcoef(series[-14:], range(14))[0,1] if len(set(series[-14:])) > 1 else 0  # Trend correlation
        ])
    else:
        features.extend([np.mean(series), 0])
    
    # Volatility features
    if n > 5:
        recent_volatility = np.std(series[-5:])
        long_volatility = np.std(series[-min(20, n):])
        features.append(recent_volatility / (long_volatility + 1e-8))
    else:
        features.append(1.0)
    
    return np.array(features)

def gbm_forecast(model, history, horizon=1):
    """Advanced GBM forecasting with feature engineering."""
    try:
        # Prepare features
        features = create_advanced_features(history)
        if features is None:
            return np.full(horizon, history[-1])
        
        # Recursive forecasting
        preds = []
        current_history = list(history)
        
        for _ in range(horizon):
            X = create_advanced_features(np.array(current_history))
            if X is None:
                pred = current_history[-1]
            else:
                pred = model.predict(X.reshape(1, -1))[0]
            
            preds.append(pred)
            current_history.append(pred)
        
        return np.array(preds)
        
    except Exception as e:
        return np.full(horizon, history[-1])

def evaluate_gbm_walk_forward(model, train_series, test_series, horizon=1, lookback=50):
    """Walk-forward evaluation for optimized GBM."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    # Use expanding window for evaluation
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    successful_forecasts = 0
    failed_forecasts = 0
    
    # Evaluate every 5th point for speed
    step_size = 5
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        # Use recent history for fitting
        recent_history = history[-lookback:] if len(history) > lookback else history
        
        try:
            # Make prediction with optimized GBM
            pred = gbm_forecast(model, recent_history, horizon)
            
            # Get actual values
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
            
        except Exception as e:
            failed_forecasts += 1
            continue
    
    print(f"    Successful: {successful_forecasts}, Failed: {failed_forecasts}")
    
    if len(predictions) == 0:
        return np.nan, np.nan, 0
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    
    # MAPE calculation
    mape_values = []
    for actual, pred in zip(actuals, predictions):
        if abs(actual) > 1e-8:
            mape_values.append(abs((actual - pred) / actual) * 100)
    
    mape = np.mean(mape_values) if mape_values else np.nan
    
    return mae, mape, len(predictions)

def evaluate_multiple_series_gbm(model, train_data, test_data, horizon=1, lookback=50, max_series=10):
    """Evaluate optimized GBM on multiple time series."""
    all_maes = []
    all_mapes = []
    total_predictions = 0
    
    # Limit to first N series for speed
    series_ids = list(train_data.keys())[:max_series]
    
    for i, series_id in enumerate(series_ids):
        if series_id not in test_data:
            continue
            
        print(f"  Processing series {i+1}/{len(series_ids)}...")
        
        mae, mape, n_preds = evaluate_gbm_walk_forward(
            model, train_data[series_id], test_data[series_id], horizon, lookback
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

def save_model(model, filepath):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load trained model from disk."""
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    else:
        print(f"Model file {filepath} not found")
        return None

def run_gbm_evaluation(save_weights=False, load_weights=False, model_path="models/gbm_model.pkl"):
    """Run optimized GBM evaluation with hyperparameter tuning."""
    print("Running Optimized GBM evaluation with hyperparameter tuning...")
    
    # Load data with proper temporal structure
    time_series = load_time_series_data('data/CRE.csv')
    
    # For windowing models: exclude last 50 points from ALL columns
    train_data, test_data = temporal_train_test_split(time_series, train_ratio=0.8, exclude_last_n=50)
    
    # Prepare training data with advanced features
    X_train, y_train = [], []
    series_ids = list(train_data.keys())[:15]  # Limit for training speed
    
    for series_id in series_ids:
        series = train_data[series_id]
        
        # Create training samples with advanced features
        for i in range(20, len(series)):  # Start from index 20 for enough history
            features = create_advanced_features(series[:i])
            if features is not None:
                X_train.append(features)
                y_train.append(series[i])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training data shape: {X_train.shape}")
    
    # Try to load existing model if requested
    best_gbm = None
    if load_weights:
        best_gbm = load_model(model_path)
    
    # Train new model if not loaded or load failed
    if best_gbm is None:
        # Enhanced hyperparameter optimization
        param_grid = {
            'n_estimators': [300, 500, 800],
            'learning_rate': [0.02, 0.05, 0.08, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.85, 0.9, 0.95],
            'max_features': ['sqrt', 'log2', 0.8, 1.0]
        }
        
        # Use RandomizedSearchCV for efficiency
        gbm_base = GradientBoostingRegressor(random_state=42)
        
        print("Performing hyperparameter optimization...")
        gbm_search = RandomizedSearchCV(
            gbm_base,
            param_distributions=param_grid,
            n_iter=40,  # More iterations for better optimization
            cv=5,  # More folds for better validation
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1
        )
        
        gbm_search.fit(X_train, y_train)
        
        # Get best model
        best_gbm = gbm_search.best_estimator_
        print(f"Best parameters: {gbm_search.best_params_}")
        
        # Save model if requested
        if save_weights:
            save_model(best_gbm, model_path)
    else:
        print("Using loaded model, skipping training")
    
    results = []
    
    # Test all horizons
    horizons = [1, 2, 3, 5, 7, 10]
    
    for horizon in horizons:
        print(f"Evaluating horizon {horizon}...")
        
        # Use fewer series for longer horizons to avoid timeouts
        max_series = 10 if horizon <= 5 else 8
        
        mae, mape, n_preds = evaluate_multiple_series_gbm(
            best_gbm, train_data, test_data, horizon=horizon, 
            lookback=50, max_series=max_series
        )
        
        results.append({
            'horizon': horizon,
            'mae': mae,
            'mape': mape,
            'n_predictions': n_preds
        })
        
        if not np.isnan(mae):
            print(f"Horizon {horizon}: MAE={mae:.6f}, MAPE={mape:.2f}%, Preds={n_preds}")
        else:
            print(f"Horizon {horizon}: FAILED - no successful predictions")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/gbm_mae_results.csv', index=False)
    print("Results saved to results/gbm_mae_results.csv")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate GBM model for time series forecasting')
    parser.add_argument('--save-weights', action='store_true', 
                        help='Save trained model weights to disk')
    parser.add_argument('--load-weights', action='store_true',
                        help='Load trained model weights from disk (skip training)')
    parser.add_argument('--model-path', type=str, default='models/gbm_model.pkl',
                        help='Path to save/load model weights (default: models/gbm_model.pkl)')
    
    args = parser.parse_args()
    
    results = run_gbm_evaluation(
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        model_path=args.model_path
    )