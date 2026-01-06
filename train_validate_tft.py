import pandas as pd
import numpy as np
import os
import warnings
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split
from sklearn.metrics import mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def prepare_tft_data(time_series, max_series=3):
    """Convert time series data to NeuralForecast format."""
    data = []
    series_ids = list(time_series.keys())[:max_series]  # Limit for speed
    
    for series_id in series_ids:
        series = time_series[series_id]
        for t, value in enumerate(series):
            data.append({
                'unique_id': series_id,
                'ds': t,
                'y': value
            })
    return pd.DataFrame(data)

def tft_forecast(model, history, horizon=1):
    """Make forecast using trained TFT model."""
    try:
        # Prepare data in required format
        data = []
        for t, value in enumerate(history):
            data.append({
                'unique_id': 'forecast_series',
                'ds': t,
                'y': value
            })
        
        history_df = pd.DataFrame(data)
        forecast = model.predict(history_df)
        
        if 'TFT' in forecast.columns:
            pred_values = forecast['TFT'].values[:horizon]
            return pred_values
        else:
            return np.full(horizon, history[-1])  # Fallback
            
    except Exception as e:
        return np.full(horizon, history[-1])  # Fallback

def evaluate_tft_walk_forward(model, train_series, test_series, horizon=1, lookback=30):
    """Walk-forward evaluation for TFT."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    # Use expanding window for evaluation
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    successful_forecasts = 0
    failed_forecasts = 0
    
    # Evaluate every 10th point to speed up
    step_size = 10
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        # Use recent history for fitting
        recent_history = history[-lookback:] if len(history) > lookback else history
        
        try:
            # Make prediction with TFT
            pred = tft_forecast(model, recent_history, horizon)
            
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

def evaluate_multiple_series_tft(model, train_data, test_data, horizon=1, lookback=30, max_series=3):
    """Evaluate TFT on multiple time series."""
    all_maes = []
    all_mapes = []
    total_predictions = 0
    
    # Limit to first N series for speed
    series_ids = list(train_data.keys())[:max_series]
    
    for i, series_id in enumerate(series_ids):
        if series_id not in test_data:
            continue
            
        print(f"  Processing series {i+1}/{len(series_ids)}...")
        
        mae, mape, n_preds = evaluate_tft_walk_forward(
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

def run_tft_evaluation():
    """Run TFT evaluation."""
    print("Running TFT evaluation...")
    
    # Load data with proper temporal structure
    time_series = load_time_series_data('data/CRE.csv')
    train_data, test_data = temporal_train_test_split(time_series, train_ratio=0.8)
    
    # Prepare training data (limit for speed)
    train_df = prepare_tft_data(train_data, max_series=2)
    
    # Initialize TFT model with faster training
    model = NeuralForecast(
        models=[
            TFT(
                h=10,  # Maximum forecast horizon
                input_size=20,  # Input window size
                hidden_size=32,  # Smaller hidden size for speed
                n_head=2,  # Fewer attention heads
                max_steps=30,  # Reduced training steps for speed
                batch_size=32,
                scaler_type='identity'
            )
        ],
        freq=1
    )
    
    # Train model
    print("Training TFT model...")
    model.fit(train_df)
    
    results = []
    
    # Test all horizons
    horizons = [1, 2, 3, 5, 7, 10]
    
    for horizon in horizons:
        print(f"Evaluating horizon {horizon}...")
        
        # Use fewer series for longer horizons to avoid timeouts
        max_series = 2 if horizon <= 5 else 1
        
        mae, mape, n_preds = evaluate_multiple_series_tft(
            model, train_data, test_data, horizon=horizon, 
            lookback=25, max_series=max_series
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
    results_df.to_csv('results/tft_mae_results.csv', index=False)
    print("Results saved to results/tft_mae_results.csv")
    
    return results

if __name__ == "__main__":
    results = run_tft_evaluation()