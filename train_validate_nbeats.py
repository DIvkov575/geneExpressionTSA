import pandas as pd
import numpy as np
import os
import warnings
import argparse
import pickle
from tqdm import tqdm
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

def prepare_nbeats_data(time_series, max_series=None):
    """Convert time series data to NeuralForecast format."""
    data = []
    series_ids = list(time_series.keys())
    if max_series:
        series_ids = series_ids[:max_series]
    
    for series_id in tqdm(series_ids, desc="Preparing data"):
        series = time_series[series_id]
        for t, value in enumerate(series):
            data.append({
                'unique_id': series_id,
                'ds': t,
                'y': value
            })
    return pd.DataFrame(data)

def nbeats_forecast(model, history, horizon=1):
    """Make forecast using trained NBEATS model."""
    try:
        data = [{'unique_id': 'forecast_series', 'ds': t, 'y': value} 
                for t, value in enumerate(history)]
        
        history_df = pd.DataFrame(data)
        forecast = model.predict(history_df)
        
        return forecast['NBEATS'].values[:horizon] if 'NBEATS' in forecast.columns else np.full(horizon, history[-1])
    except:
        return np.full(horizon, history[-1])

def evaluate_nbeats_walk_forward(model, train_series, test_series, horizon=1, lookback=30):
    """Walk-forward evaluation for NBEATS."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals, predictions = [], []
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    successful_forecasts = 0
    
    step_size = 5  # Reduced for more evaluation points
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        if len(history) < lookback:
            continue
            
        recent_history = history[-lookback:] if len(history) > lookback else history
        
        try:
            pred = nbeats_forecast(model, recent_history, horizon)
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
        except:
            continue
    
    if len(predictions) == 0:
        return np.nan, np.nan, 0
    
    mae = mean_absolute_error(actuals, predictions)
    
    mape_values = [abs((actual - pred) / actual) * 100 
                   for actual, pred in zip(actuals, predictions) 
                   if abs(actual) > 1e-8]
    mape = np.mean(mape_values) if mape_values else np.nan
    
    return mae, mape, len(predictions)

def evaluate_multiple_series_nbeats(model, train_data, test_data, horizon=1, lookback=30, max_series=None):
    """Evaluate NBEATS on multiple time series."""
    all_maes, all_mapes = [], []
    total_predictions = 0
    
    series_ids = list(train_data.keys())
    if max_series:
        series_ids = series_ids[:max_series]
    
    for series_id in tqdm(series_ids, desc=f"Evaluating horizon {horizon}"):
        if series_id not in test_data:
            continue
        
        mae, mape, n_preds = evaluate_nbeats_walk_forward(
            model, train_data[series_id], test_data[series_id], horizon, lookback
        )
        
        if not np.isnan(mae):
            all_maes.append(mae)
            total_predictions += n_preds
            
        if not np.isnan(mape):
            all_mapes.append(mape)
    
    if len(all_maes) == 0:
        return np.nan, np.nan, 0
    
    return np.mean(all_maes), np.mean(all_mapes) if all_mapes else np.nan, total_predictions

def save_nbeats_model(model, filepath):
    """Save trained N-BEATS model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_nbeats_model(filepath):
    """Load trained N-BEATS model from disk."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def run_nbeats_evaluation(save_weights=False, load_weights=False, model_path="models/nbeats_model.pkl"):
    """Run NBEATS evaluation."""
    print("Running NBEATS evaluation...")
    
    time_series = load_time_series_data('data/CRE.csv')
    
    # For windowing models: exclude last 50 points from ALL columns
    train_data, test_data = temporal_train_test_split(time_series, train_ratio=0.8, exclude_last_n=50)
    
    # Use ALL available training data
    train_df = prepare_nbeats_data(train_data)
    
    model = None
    if load_weights:
        model = load_nbeats_model(model_path)
        if model:
            print("Loaded existing model")
    
    if model is None:
        print("Training NBEATS model...")
        
        model = NeuralForecast(
            models=[
                NBEATS(
                    h=10,
                    input_size=64,
                    max_steps=300,
                    batch_size=64,
                    scaler_type='robust',
                    n_blocks=[3, 3, 3],
                    mlp_units=[[512, 512], [512, 512], [512, 512]],
                    learning_rate=0.001,
                    val_check_steps=25,
                    early_stop_patience_steps=50
                )
            ],
            freq=1
        )
        
        with tqdm(desc="Training model") as pbar:
            model.fit(train_df, val_size=300)
            pbar.update(1)
        
        if save_weights:
            save_nbeats_model(model, model_path)
    
    results = []
    horizons = [1, 2, 3, 5, 7, 10]
    
    for horizon in horizons:
        mae, mape, n_preds = evaluate_multiple_series_nbeats(
            model, train_data, test_data, horizon=horizon, lookback=64
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
            print(f"Horizon {horizon}: FAILED")
    
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/nbeats_mae_results.csv', index=False)
    print("Results saved to results/nbeats_mae_results.csv")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate N-BEATS model for time series forecasting')
    parser.add_argument('--save-weights', action='store_true', 
                        help='Save trained model weights to disk')
    parser.add_argument('--load-weights', action='store_true',
                        help='Load trained model weights from disk (skip training)')
    parser.add_argument('--model-path', type=str, default='models/nbeats_model.pkl',
                        help='Path to save/load model weights (default: models/nbeats_model.pkl)')
    
    args = parser.parse_args()
    
    results = run_nbeats_evaluation(
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        model_path=args.model_path
    )