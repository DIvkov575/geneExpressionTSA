
import pandas as pd
import numpy as np
import os
import warnings
import argparse
import pickle
from hmmlearn.hmm import GaussianHMM
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def train_hmm(data, n_states=4, n_iter=100):
    """Train a Gaussian HMM on the differences of the time series."""
    if len(data) < 2:
        return None
    diffs = np.diff(data).reshape(-1, 1)
    if len(diffs) < n_states:
        return None
    try:
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter)
        model.fit(diffs)
        return model
    except Exception:
        return None

def hmm_forecast(model, history, horizon=1):
    """Forecast using the trained HMM."""
    if model is None or len(history) < 2:
        return np.full(horizon, history[-1])

    diffs = np.diff(history).reshape(-1, 1)
    try:
        hidden_states = model.predict(diffs)
        last_state = hidden_states[-1]
        
        # Predict next state based on transition matrix
        next_state_probs = model.transmat_[last_state]
        next_state = np.argmax(next_state_probs)
        
        # Use the mean of the predicted state's emission as the forecasted difference
        predicted_diff = model.means_[next_state][0]
        
        forecasts = []
        last_value = history[-1]
        for _ in range(horizon):
            next_value = last_value + predicted_diff
            forecasts.append(next_value)
            last_value = next_value
        
        return np.array(forecasts)

    except Exception:
        return np.full(horizon, history[-1]) # Fallback to naive

def evaluate_hmm_walk_forward(train_series, test_series, n_states=4, lookback=50, horizon=1):
    """Walk-forward evaluation for HMM."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    successful_forecasts = 0
    
    for i in range(train_end, len(full_series) - horizon + 1):
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        recent_history = history[-lookback:]
        
        model = train_hmm(recent_history, n_states=n_states)
        
        if model:
            pred = hmm_forecast(model, recent_history, horizon)
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
            
    if len(predictions) == 0:
        return np.nan, np.nan, 0
    
    mae = mean_absolute_error(actuals, predictions)
    
    mape_values = [abs((a - p) / a) * 100 for a, p in zip(actuals, predictions) if abs(a) > 1e-8]
    mape = np.mean(mape_values) if mape_values else np.nan
    
    return mae, mape, successful_forecasts

def evaluate_multiple_series_hmm(train_data, test_data, n_states=4, lookback=50, horizon=1, max_series=10):
    """Evaluate HMM on multiple time series."""
    all_maes = []
    all_mapes = []
    total_predictions = 0
    
    series_ids = list(train_data.keys())[:max_series]
    
    for i, series_id in enumerate(series_ids):
        if series_id not in test_data:
            continue
            
        print(f"  Processing series {i+1}/{len(series_ids)}...")
        
        mae, mape, n_preds = evaluate_hmm_walk_forward(
            train_data[series_id], test_data[series_id], n_states, lookback, horizon
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

def save_hmm_model(model_params, filepath):
    """Save HMM model parameters to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)
    print(f"HMM model parameters saved to {filepath}")

def run_hmm_evaluation(save_weights=False, model_path="models/hmm_model.pkl"):
    """Run HMM evaluation."""
    print("Running HMM evaluation...")
    
    time_series = load_time_series_data('data/CRE.csv')
    train_data, test_data = temporal_train_test_split(time_series, train_ratio=0.8)
    
    results = []
    horizons = [1, 2, 3, 5, 7, 10]
    
    for horizon in horizons:
        print(f"Evaluating horizon {horizon}...")
        
        max_series = 10 if horizon <= 5 else 7
        
        mae, mape, n_preds = evaluate_multiple_series_hmm(
            train_data, test_data, n_states=4, 
            lookback=50, horizon=horizon, max_series=max_series
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
    results_df.to_csv('results/hmm_mae_results.csv', index=False)
    print("Results saved to results/hmm_mae_results.csv")
    
    if save_weights:
        model_params = {
            'model_type': 'hmm',
            'n_states': 4,
            'results': results
        }
        save_hmm_model(model_params, model_path)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate HMM for time series forecasting')
    parser.add_argument('--save-weights', action='store_true', help='Save model parameters')
    parser.add_argument('--model-path', type=str, default='models/hmm_model.pkl', help='Path to save model parameters')
    
    args = parser.parse_args()
    
    run_hmm_evaluation(
        save_weights=args.save_weights,
        model_path=args.model_path
    )
