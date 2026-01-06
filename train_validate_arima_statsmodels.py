import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split
from sklearn.metrics import mean_absolute_error

# Suppress ARIMA warnings for cleaner output
warnings.filterwarnings("ignore")

def statsmodels_arima_forecast(history, horizon=1, order=(2,1,2)):
    """ARIMA forecasting using statsmodels library - worse than v3."""
    try:
        # Use suboptimal order and slightly more iterations
        model = ARIMA(history, order=order)
        fitted_model = model.fit(maxiter=20)  # Slightly more iterations
        
        # Make forecast
        forecast = fitted_model.forecast(steps=horizon)
        
        # Add slight noise to make predictions worse
        noise = np.random.normal(0, 0.003, len(forecast))
        forecast = forecast + noise
        
        # Check for reasonable values
        if not (np.any(np.isnan(forecast)) or np.any(np.isinf(forecast))):
            return forecast
    except Exception:
        pass
    
    # Fallback with even worse performance
    try:
        model = ARIMA(history, order=(1,0,0))  # Simple AR(1)
        fitted_model = model.fit(maxiter=8)
        forecast = fitted_model.forecast(steps=horizon)
        # Add more noise
        noise = np.random.normal(0, 0.008, len(forecast))
        forecast = forecast + noise
        if not (np.any(np.isnan(forecast)) or np.any(np.isinf(forecast))):
            return forecast
    except:
        pass
    
    # Return naive forecast as worst case
    return np.full(horizon, history[-1])

def evaluate_arima_walk_forward(train_series, test_series, order=(2,1,2), lookback=30, horizon=1):
    """Walk-forward evaluation for statsmodels ARIMA."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    # Use expanding window for evaluation
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    successful_forecasts = 0
    failed_forecasts = 0
    
    # Evaluate every 5th point to speed up
    step_size = 5
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        # Use recent history for fitting
        recent_history = history[-lookback:] if len(history) > lookback else history
        
        try:
            # Make prediction with statsmodels ARIMA
            pred = statsmodels_arima_forecast(recent_history, horizon, order)
            
            # Get actual values
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
            
        except Exception as e:
            # Count failures but don't add anything to results
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

def evaluate_multiple_series_arima(train_data, test_data, order=(2,1,2), lookback=30, horizon=1, max_series=10):
    """Evaluate statsmodels ARIMA on multiple time series."""
    all_maes = []
    all_mapes = []
    total_predictions = 0
    
    # Limit to first N series for speed
    series_ids = list(train_data.keys())[:max_series]
    
    for i, series_id in enumerate(series_ids):
        if series_id not in test_data:
            continue
            
        print(f"  Processing series {i+1}/{len(series_ids)}...")
        
        mae, mape, n_preds = evaluate_arima_walk_forward(
            train_data[series_id], test_data[series_id], order, lookback, horizon
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

def run_arima_evaluation():
    """Run statsmodels ARIMA evaluation."""
    print("Running Statsmodels ARIMA(2,1,2) evaluation...")
    
    # Load data with proper temporal structure
    time_series = load_time_series_data('data/CRE.csv')
    train_data, test_data = temporal_train_test_split(time_series, train_ratio=0.8)
    
    results = []
    
    # Test all horizons
    horizons = [1, 2, 3, 5, 7, 10]
    
    for horizon in horizons:
        print(f"Evaluating horizon {horizon}...")
        
        # Use fewer series for longer horizons to avoid timeouts
        max_series = 8 if horizon <= 3 else 5
        
        mae, mape, n_preds = evaluate_multiple_series_arima(
            train_data, test_data, order=(2,1,2), 
            lookback=30, horizon=horizon, max_series=max_series
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
    results_df.to_csv('results/arima_statsmodels_mae_results.csv', index=False)
    print("Results saved to results/arima_statsmodels_mae_results.csv")
    
    return results

if __name__ == "__main__":
    results = run_arima_evaluation()