import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
import numpy as np
import os
import argparse
import pickle
from pathlib import Path
from proper_ts_evaluation import load_time_series_data, temporal_train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Neural ODE Model Definition ---
class ODEFunc(nn.Module):
    """
    Neural network representing the ODE function f(t, y).
    """
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, y):
        return self.net(y)

class ODEBlock(nn.Module):
    """
    Neural ODE block that solves the ODE.
    """
    def __init__(self, ode_func, rtol=1e-3, atol=1e-4, method='dopri5'):
        super(ODEBlock, self).__init__()
        self.ode_func = ode_func
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def forward(self, y0, t_span):
        # odeint solves the ODE from y0 at t_span[0] to t_span[1:]
        # It returns a tensor of shape (len(t_span), *y0.shape)
        return odeint(self.ode_func, y0, t_span, rtol=self.rtol, atol=self.atol, method=self.method)

# --- Forecasting Function ---
def scnode_forecast(model, history, horizon=1, device='cpu'):
    """
    Generates a forecast using the trained Neural ODE model.
    """
    model.eval()
    with torch.no_grad():
        # history is a numpy array, convert to tensor
        y0 = torch.tensor(history[-1:], dtype=torch.float32).to(device) # Use the last point as initial state
        
        # Define time points for forecasting
        # Assuming each step is a unit of time
        t_span = torch.linspace(0, horizon, horizon + 1, dtype=torch.float32).to(device)
        
        # Solve the ODE
        # The output will be (horizon + 1, 1, input_dim)
        # We need the forecasts from t=1 to t=horizon
        predictions = model(y0, t_span)
        
        # Extract the forecasted values (excluding the initial state y0)
        # predictions[1:] gives forecasts for t=1, t=2, ..., t=horizon
        forecast = predictions[1:].squeeze().cpu().numpy()
        
        # If horizon is 1, forecast might be a single value, ensure it's an array
        if forecast.ndim == 0:
            forecast = np.array([forecast])
            
        return forecast

# --- Evaluation Functions (Adapted from ARIMA script) ---
def evaluate_scnode_walk_forward(model, train_series, test_series, lookback=30, horizon=1, device='cpu'):
    """Walk-forward evaluation for scNode."""
    if len(test_series) < horizon:
        return np.nan, np.nan, 0
    
    actuals = []
    predictions = []
    
    full_series = np.concatenate([train_series, test_series])
    train_end = len(train_series)
    
    successful_forecasts = 0
    failed_forecasts = 0
    
    # Evaluate every 5th point to speed up, similar to ARIMA script
    step_size = 5
    
    for i in range(train_end, len(full_series) - horizon + 1, step_size):
        history = full_series[:i]
        
        if len(history) < lookback:
            continue
            
        # Use recent history for fitting (though scNode uses only the last point as y0)
        # The 'lookback' here primarily controls how much history is available for the *initial state*
        # and implicitly, how much data was used to *train* the model up to this point.
        recent_history = history[-lookback:] if len(history) > lookback else history
        
        try:
            pred = scnode_forecast(model, recent_history, horizon, device)
            
            actual = full_series[i:i+horizon]
            
            actuals.extend(actual)
            predictions.extend(pred)
            successful_forecasts += 1
            
        except Exception as e:
            failed_forecasts += 1
            print(f"  Forecast failed: {e}")
            continue
    
    print(f"    Successful: {successful_forecasts}, Failed: {failed_forecasts}")
    
    if len(predictions) == 0:
        return np.nan, np.nan, 0
    
    mae = mean_absolute_error(actuals, predictions)
    
    mape_values = []
    for actual, pred in zip(actuals, predictions):
        if abs(actual) > 1e-8:
            mape_values.append(abs((actual - pred) / actual) * 100)
    
    mape = np.mean(mape_values) if mape_values else np.nan
    
    return mae, mape, len(predictions)

def evaluate_multiple_series_scnode(model, train_data, test_data, lookback=30, horizon=1, max_series=10, device='cpu'):
    """Evaluate scNode on multiple time series."""
    all_maes = []
    all_mapes = []
    total_predictions = 0
    
    series_ids = list(train_data.keys())[:max_series]
    
    for i, series_id in enumerate(series_ids):
        if series_id not in test_data:
            continue
            
        print(f"  Processing series {i+1}/{len(series_ids)}...")
        
        mae, mape, n_preds = evaluate_scnode_walk_forward(
            model, train_data[series_id], test_data[series_id], lookback, horizon, device
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

# --- Training Function ---
def train_scnode_model(train_data, input_dim, hidden_dim, epochs=50, lr=0.01, device='cpu'):
    """
    Trains the Neural ODE model.
    This is a simplified training for demonstration. In a real scenario,
    you'd train on sequences, not just individual points.
    """
    func = ODEFunc(input_dim, hidden_dim).to(device)
    model = ODEBlock(func).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    print(f"Training scNode model on {len(train_data)} series for {epochs} epochs...")

    # For simplicity, we'll train on the last 'lookback' points of each series
    # to learn the dynamics. A more robust approach would involve sequence-to-sequence
    # training or training on multiple segments.
    
    # Collect all training sequences
    all_sequences = []
    for series_id in train_data:
        series = train_data[series_id]
        if len(series) > 1: # Need at least 2 points to define a dynamic
            # Create sequences of (y_t, y_{t+1}) or similar for ODE training
            # Here, we'll just use the series itself and try to predict the next step
            # This is a very basic approach for demonstration.
            for i in range(len(series) - 1):
                all_sequences.append((series[i], series[i+1]))
    
    if not all_sequences:
        raise ValueError("Not enough training data to form sequences for scNode training.")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for y_prev_np, y_next_np in all_sequences:
            optimizer.zero_grad()
            
            y0 = torch.tensor([y_prev_np], dtype=torch.float32).to(device)
            target = torch.tensor([y_next_np], dtype=torch.float32).to(device)
            
            # Predict one step forward
            t_span = torch.tensor([0, 1], dtype=torch.float32).to(device)
            pred_sequence = model(y0, t_span)
            
            # We are interested in the prediction at t=1
            pred = pred_sequence[-1]
            
            loss = loss_func(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(all_sequences):.6f}")
            
    print("Training complete.")
    return model

def save_scnode_model(model, filepath):
    """Save scNode model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"scNode model saved to {filepath}")

def load_scnode_model(filepath, input_dim, hidden_dim, device='cpu'):
    """Load scNode model from disk."""
    if os.path.exists(filepath):
        func = ODEFunc(input_dim, hidden_dim).to(device)
        model = ODEBlock(func).to(device)
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()
        print(f"scNode model loaded from {filepath}")
        return model
    else:
        print(f"scNode model file {filepath} not found")
        return None

def run_scnode_evaluation(save_weights=False, load_weights=False, model_path="models/scnode_model.pth"):
    """Run scNode evaluation."""
    print("Running scNode evaluation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    time_series = load_time_series_data('data/CRE.csv')
    train_data, test_data = temporal_train_test_split(time_series, train_ratio=0.8)
    
    # Determine input dimension from data (assuming univariate for now)
    # If data is multivariate, this needs adjustment
    input_dim = 1 
    hidden_dim = 20 # Hyperparameter for ODEFunc

    model = None
    if load_weights:
        model = load_scnode_model(model_path, input_dim, hidden_dim, device)
    
    if model is None: # Train if not loaded or load failed
        model = train_scnode_model(train_data, input_dim, hidden_dim, epochs=20, lr=0.01, device=device)
        if save_weights:
            save_scnode_model(model, model_path)
    
    results = []
    horizons = [1, 2, 3, 5, 7, 10]
    
    for horizon in horizons:
        print(f"Evaluating horizon {horizon}...")
        
        max_series = 3 if horizon <= 3 else 2 # Limit series for longer horizons
        
        mae, mape, n_preds = evaluate_multiple_series_scnode(
            model, train_data, test_data, lookback=30, horizon=horizon, max_series=max_series, device=device
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
    
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/scnode_mae_results.csv', index=False)
    print("Results saved to results/scnode_mae_results.csv")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate scNode for time series forecasting')
    parser.add_argument('--save-weights', action='store_true', 
                        help='Save trained model weights to disk')
    parser.add_argument('--load-weights', action='store_true',
                        help='Load trained model weights from disk (skip training)')
    parser.add_argument('--model-path', type=str, default='models/scnode_model.pth',
                        help='Path to save/load model weights (default: models/scnode_model.pth)')
    
    args = parser.parse_args()
    
    run_scnode_evaluation(
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        model_path=args.model_path
    )
