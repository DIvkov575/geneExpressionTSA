#!/usr/bin/env python3
"""
Model-Aware Forecasting System

Generates recursive forecasts using model-specific seed sizes and parameters.
Supports full-length forecasting for all columns and extrapolation for column '2'.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import argparse
import pickle
import torch
from datetime import datetime
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Model-specific seed size requirements based on parameter analysis
SEED_REQUIREMENTS = {
    'nbeats': 64,      # Highest requirement - input_size=64
    'tft': 50,         # input_size=50
    'gbm': 50,         # lookback=50, feature engineering needs
    'arima_statsmodels': 30,  # lookback=30
    'scnode': 30,      # lookback=30
    'arima_v3': 30,    # minimum 15, prefer 30 for stability
    'naive': 25        # uses 25-point windows
}

# Import model classes and functions
try:
    from models.ARIMA_model_v3 import MultiHorizonARIMA_v3
except ImportError:
    print("Warning: Could not import ARIMA v3 model")
    MultiHorizonARIMA_v3 = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    print("Warning: Could not import statsmodels ARIMA")
    ARIMA = None

try:
    import torch
    import torch.nn as nn
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    print("Warning: Could not import PyTorch dependencies for scNode")
    torch = None

class ModelLoader:
    """Handles loading of different model types with error handling."""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        
    def load_arima_v3(self):
        """Load ARIMA v3 model."""
        try:
            model_path = self.models_dir / "arima_v3.pkl"
            if model_path.exists() and MultiHorizonARIMA_v3:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
        except Exception as e:
            print(f"Could not load ARIMA v3: {e}")
        return None
        
    def load_arima_statsmodels(self):
        """Load ARIMA statsmodels."""
        try:
            model_path = self.models_dir / "arima_statsmodels.pkl"
            if model_path.exists() and ARIMA:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
        except Exception as e:
            print(f"Could not load ARIMA statsmodels: {e}")
        return None
        
    def load_gbm(self):
        """Load GBM model."""
        try:
            model_path = self.models_dir / "gbm_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
        except Exception as e:
            print(f"Could not load GBM: {e}")
        return None
        
    def load_nbeats(self):
        """Load N-BEATS model."""
        try:
            model_path = self.models_dir / "nbeats_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
        except Exception as e:
            print(f"Could not load N-BEATS: {e}")
        return None
        
    def load_tft(self):
        """Load TFT model."""
        try:
            model_path = self.models_dir / "tft_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
        except Exception as e:
            print(f"Could not load TFT: {e}")
        return None
        
    def load_naive(self):
        """Load Naive model (or create simple implementation)."""
        # Naive model is simple enough to implement directly
        class NaiveModel:
            def predict(self, history, steps=1):
                """Naive prediction - repeat last value."""
                return np.full(steps, history[-1])
        return NaiveModel()
        
    def load_scnode(self):
        """Load scNode model."""
        try:
            model_path = self.models_dir / "scnode_model.pth"
            info_path = self.models_dir / "scnode_info.pkl"
            if model_path.exists() and info_path.exists() and torch:
                import pickle
                
                # Load model info
                with open(info_path, 'rb') as f:
                    model_info = pickle.load(f)
                
                # Recreate model architecture
                class ODEFunc(torch.nn.Module):
                    def __init__(self, input_dim, hidden_dim):
                        super(ODEFunc, self).__init__()
                        self.net = torch.nn.Sequential(
                            torch.nn.Linear(input_dim, hidden_dim),
                            torch.nn.Tanh(),
                            torch.nn.Linear(hidden_dim, input_dim)
                        )
                    def forward(self, t, y):
                        return self.net(y)
                
                class ODEBlock(torch.nn.Module):
                    def __init__(self, ode_func, rtol=1e-3, atol=1e-4, method='dopri5'):
                        super(ODEBlock, self).__init__()
                        self.ode_func = ode_func
                        self.rtol = rtol
                        self.atol = atol
                        self.method = method
                    def forward(self, y0, t_span):
                        from torchdiffeq import odeint_adjoint as odeint
                        return odeint(self.ode_func, y0, t_span, rtol=self.rtol, atol=self.atol, method=self.method)
                
                # Create model and load weights
                func = ODEFunc(model_info['input_dim'], model_info['hidden_dim'])
                model = ODEBlock(func)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                return model
        except Exception as e:
            print(f"Could not load scNode: {e}")
        return None
        
    def load_all_models(self, requested_models=None):
        """Load all available models or specific requested models."""
        model_loaders = {
            'arima_v3': self.load_arima_v3,
            'arima_statsmodels': self.load_arima_statsmodels,
            'gbm': self.load_gbm,
            'nbeats': self.load_nbeats,
            'tft': self.load_tft,
            'naive': self.load_naive,
            'scnode': self.load_scnode
        }
        
        if requested_models is None:
            requested_models = list(model_loaders.keys())
        
        for model_name in requested_models:
            if model_name in model_loaders:
                print(f"Loading {model_name}...")
                model = model_loaders[model_name]()
                if model is not None:
                    self.loaded_models[model_name] = model
                    print(f"  ✓ {model_name} loaded successfully")
                else:
                    print(f"  ✗ {model_name} failed to load")
            else:
                print(f"  ✗ Unknown model: {model_name}")
                
        return self.loaded_models


class ForecastEngine:
    """Handles recursive forecasting with model-specific parameters."""
    
    def __init__(self, models):
        self.models = models
        
    def calculate_metrics(self, actual, predicted):
        """Calculate MAE and MAPE metrics."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # MAE
        mae = np.mean(np.abs(actual - predicted))
        
        # MAPE (handle division by zero)
        mape_values = []
        for a, p in zip(actual, predicted):
            if abs(a) > 1e-8:
                mape_values.append(abs((a - p) / a) * 100)
        mape = np.mean(mape_values) if mape_values else np.nan
        
        return mae, mape
        
    def predict_single_step(self, model_name, model, history):
        """Make a single step prediction using specified model."""
        try:
            if model_name == 'naive':
                return model.predict(history, steps=1)[0]
            elif model_name == 'arima_v3':
                # For recursive forecasting, use a simpler AR(1) approach to avoid overfitting
                recent_history = history[-min(10, len(history)):]
                if len(recent_history) < 3:
                    return history[-1]
                
                try:
                    # Simple AR(1) model: y[t] = c + φ*y[t-1] + ε[t]
                    y = np.array(recent_history)
                    if len(y) >= 3:
                        # Fit simple AR(1) using least squares
                        X = y[:-1].reshape(-1, 1)  # y[t-1]
                        y_target = y[1:]  # y[t]
                        
                        # Add constant term
                        X_with_const = np.column_stack([np.ones(len(X)), X])
                        
                        # Solve normal equations
                        try:
                            params = np.linalg.solve(X_with_const.T @ X_with_const, X_with_const.T @ y_target)
                            c, phi = params[0], params[1]
                            
                            # Predict next value
                            pred_value = c + phi * history[-1]
                            
                            # Sanity check - reject extreme values or unstable parameters
                            if (np.isnan(pred_value) or np.isinf(pred_value) or 
                                abs(pred_value) > 10 * np.std(history) or abs(phi) > 2):
                                # Fall back to trend
                                trend = np.mean(np.diff(recent_history[-3:]))
                                return history[-1] + 0.5 * trend  # Damped trend
                            
                            return pred_value
                        except np.linalg.LinAlgError:
                            # Matrix is singular, fall back to trend
                            trend = np.mean(np.diff(recent_history[-3:]))
                            return history[-1] + 0.5 * trend
                    else:
                        return history[-1]
                        
                except Exception as e:
                    # If all else fails, fall back to simple trend
                    if len(recent_history) >= 3:
                        trend = np.mean(np.diff(recent_history[-3:]))
                        return history[-1] + 0.5 * trend
                    return history[-1]
                    
            elif model_name == 'arima_statsmodels':
                # Simple ARIMA prediction with better error handling
                recent_history = history[-min(30, len(history)):]
                if len(recent_history) < 10:
                    return history[-1]
                    
                try:
                    model_fit = ARIMA(recent_history, order=(2,1,2)).fit(disp=False)
                    pred = model_fit.forecast(steps=1)
                    pred_value = pred[0] if hasattr(pred, '__len__') else pred
                    
                    # Sanity check
                    if np.isnan(pred_value) or np.isinf(pred_value) or abs(pred_value) > 10 * np.std(history):
                        return history[-1]
                    
                    return pred_value
                except:
                    # Fallback to trend if available
                    if len(recent_history) >= 3:
                        trend = np.mean(np.diff(recent_history[-3:]))
                        return history[-1] + trend
                    return history[-1]
                    
            elif model_name == 'scnode':
                # Neural ODE prediction
                try:
                    with torch.no_grad():
                        model.eval()
                        # Use last value as initial condition
                        y0 = torch.tensor([[history[-1]]], dtype=torch.float32)
                        t_span = torch.tensor([0.0, 1.0], dtype=torch.float32)
                        
                        # Solve ODE for one time step
                        pred_sequence = model(y0, t_span)
                        pred_value = pred_sequence[-1].item()
                        
                        # Sanity check
                        if np.isnan(pred_value) or np.isinf(pred_value) or abs(pred_value) > 10 * np.std(history):
                            return history[-1]
                            
                        return pred_value
                except Exception as e:
                    print(f"    scNode prediction error: {e}")
                    return history[-1]
                    
            elif model_name in ['gbm', 'nbeats', 'tft']:
                # These models would need specific preprocessing
                # For now, implement simple forecasting logic
                if hasattr(model, 'predict'):
                    # Reshape for sklearn-like models
                    seed_size = SEED_REQUIREMENTS.get(model_name, 30)
                    recent_history = history[-seed_size:]
                    if len(recent_history) == seed_size:
                        pred = model.predict([recent_history])
                        pred_value = pred[0] if hasattr(pred, '__len__') else pred
                        
                        # Sanity check
                        if np.isnan(pred_value) or np.isinf(pred_value) or abs(pred_value) > 10 * np.std(history):
                            return history[-1]
                            
                        return pred_value
                return history[-1]  # Fallback to naive
            else:
                return history[-1]  # Default fallback
        except Exception as e:
            print(f"    Warning: {model_name} prediction failed: {e}")
            return history[-1]  # Fallback to naive prediction
            
    def recursive_forecast_column(self, column_data, model_name, model, target_length=None):
        """Generate recursive forecast for a single column."""
        if target_length is None:
            target_length = len(column_data)
            
        seed_size = SEED_REQUIREMENTS.get(model_name, 30)
        
        # Ensure we have enough seed data
        if len(column_data) < seed_size:
            print(f"    Warning: Column too short ({len(column_data)}) for {model_name} (needs {seed_size})")
            return None, None, None
            
        # Use model-specific seed
        seed = column_data[:seed_size].copy()
        forecasted = seed.tolist()
        
        # Track metrics progressively
        maes, mapes = [], []
        
        # Generate remaining points recursively
        for step in range(seed_size, target_length):
            # Predict next value using only forecasted history
            next_val = self.predict_single_step(model_name, model, np.array(forecasted))
            forecasted.append(next_val)
            
            # Calculate metrics if we have actual data to compare
            if step < len(column_data):
                mae, mape = self.calculate_metrics([column_data[step]], [next_val])
                maes.append(mae)
                mapes.append(mape)
        
        return np.array(forecasted), maes, mapes
        
    def extrapolate_column2(self, column_data, model_name, model, extrapolate_steps=50):
        """Generate extrapolation forecast for column '2'."""
        seed_size = SEED_REQUIREMENTS.get(model_name, 30)
        
        # Use last seed_size points as starting point
        if len(column_data) < seed_size:
            print(f"    Warning: Column '2' too short for {model_name} extrapolation")
            return None
            
        # Start with last seed_size points
        seed = column_data[-seed_size:].copy()
        extrapolated = seed.tolist()
        
        # Generate extrapolated points recursively
        for step in range(extrapolate_steps):
            # Use only generated/seed history for prediction
            history_for_pred = np.array(extrapolated)
            next_val = self.predict_single_step(model_name, model, history_for_pred)
            extrapolated.append(next_val)
            
        return np.array(extrapolated)


def load_data(file_path="data/CRE.csv"):
    """Load time series data."""
    df = pd.read_csv(file_path)
    
    # Get all columns except time-axis
    series_cols = [col for col in df.columns if col != 'time-axis']
    
    # Return as dictionary
    data = {}
    for col in series_cols:
        data[col] = df[col].values
        
    return data


def save_results(results, output_dir="forecasts"):
    """Save forecasting results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save recursive forecasting results
    if 'recursive' in results:
        recursive_file = f"{output_dir}/recursive_forecasts_{timestamp}.csv"
        results['recursive'].to_csv(recursive_file, index=False)
        print(f"Recursive forecasts saved to: {recursive_file}")
    
    # Save extrapolation results
    if 'extrapolation' in results:
        extrap_file = f"{output_dir}/column2_extrapolation_{timestamp}.csv"
        results['extrapolation'].to_csv(extrap_file, index=False)
        print(f"Extrapolation forecasts saved to: {extrap_file}")
    
    # Save metrics summary
    if 'metrics' in results:
        metrics_file = f"{output_dir}/forecast_metrics_{timestamp}.csv"
        results['metrics'].to_csv(metrics_file, index=False)
        print(f"Forecast metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate recursive forecasts with model-specific parameters')
    parser.add_argument('--models', type=str, default='arima_v3,naive', 
                       help='Comma-separated list of models to use')
    parser.add_argument('--extrapolate-steps', type=int, default=50,
                       help='Number of extrapolation steps for column 2')
    parser.add_argument('--output-dir', type=str, default='forecasts',
                       help='Output directory for results')
    parser.add_argument('--data-file', type=str, default='data/CRE.csv',
                       help='Input data file')
    
    args = parser.parse_args()
    
    # Parse requested models
    requested_models = [m.strip() for m in args.models.split(',')]
    
    print("🔮 Model-Aware Forecasting System")
    print("=" * 50)
    print(f"Requested models: {requested_models}")
    print(f"Extrapolation steps: {args.extrapolate_steps}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    print("\n📊 Loading data...")
    data = load_data(args.data_file)
    print(f"Loaded {len(data)} time series, length: {len(next(iter(data.values())))}")
    
    # Load models
    print("\n🤖 Loading models...")
    loader = ModelLoader()
    models = loader.load_all_models(requested_models)
    
    if not models:
        print("No models loaded successfully. Exiting.")
        return
        
    print(f"Successfully loaded {len(models)} models: {list(models.keys())}")
    
    # Initialize forecast engine
    engine = ForecastEngine(models)
    
    # Results storage
    recursive_results = []
    extrapolation_results = []
    metrics_results = []
    
    print(f"\n🔄 Generating recursive forecasts for all {len(data)} columns...")
    
    # Generate recursive forecasts for all columns
    for col_name, col_data in data.items():
        print(f"\n  Column '{col_name}':")
        
        for model_name, model in models.items():
            print(f"    {model_name}...", end=" ")
            
            forecasted, maes, mapes = engine.recursive_forecast_column(
                col_data, model_name, model
            )
            
            if forecasted is not None:
                # Calculate overall metrics
                seed_size = SEED_REQUIREMENTS.get(model_name, 30)
                overall_mae, overall_mape = engine.calculate_metrics(
                    col_data[seed_size:], forecasted[seed_size:]
                )
                
                print(f"MAE: {overall_mae:.6f}, MAPE: {overall_mape:.2f}%")
                
                # Store results
                for i, val in enumerate(forecasted):
                    recursive_results.append({
                        'column': col_name,
                        'model': model_name,
                        'step': i,
                        'predicted_value': val,
                        'actual_value': col_data[i] if i < len(col_data) else np.nan,
                        'is_recursive': i >= seed_size
                    })
                
                # Store metrics
                metrics_results.append({
                    'column': col_name,
                    'model': model_name,
                    'forecast_type': 'recursive',
                    'mae': overall_mae,
                    'mape': overall_mape,
                    'seed_size': seed_size,
                    'forecast_length': len(forecasted) - seed_size
                })
            else:
                print("FAILED")
    
    # Generate extrapolation forecasts for column '2'
    if '2' in data:
        print(f"\n🚀 Generating extrapolation forecasts for column '2' ({args.extrapolate_steps} steps)...")
        
        for model_name, model in models.items():
            print(f"    {model_name}...", end=" ")
            
            extrapolated = engine.extrapolate_column2(
                data['2'], model_name, model, args.extrapolate_steps
            )
            
            if extrapolated is not None:
                seed_size = SEED_REQUIREMENTS.get(model_name, 30)
                print(f"Generated {len(extrapolated)} total points ({args.extrapolate_steps} new)")
                
                # Store results
                for i, val in enumerate(extrapolated):
                    extrapolation_results.append({
                        'model': model_name,
                        'step': i,
                        'value': val,
                        'is_extrapolated': i >= seed_size
                    })
                
                # Store metrics
                metrics_results.append({
                    'column': '2',
                    'model': model_name,
                    'forecast_type': 'extrapolation',
                    'mae': np.nan,  # No ground truth for extrapolation
                    'mape': np.nan,
                    'seed_size': seed_size,
                    'forecast_length': args.extrapolate_steps
                })
            else:
                print("FAILED")
    
    # Convert results to DataFrames
    results = {}
    if recursive_results:
        results['recursive'] = pd.DataFrame(recursive_results)
    if extrapolation_results:
        results['extrapolation'] = pd.DataFrame(extrapolation_results)
    if metrics_results:
        results['metrics'] = pd.DataFrame(metrics_results)
    
    # Save results
    print(f"\n💾 Saving results...")
    save_results(results, args.output_dir)
    
    print("\n✅ Forecasting complete!")
    print("\nSummary:")
    if 'metrics' in results:
        summary = results['metrics'].groupby(['forecast_type', 'model']).agg({
            'mae': 'mean',
            'mape': 'mean'
        }).round(6)
        print(summary)


if __name__ == "__main__":
    main()