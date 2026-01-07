#!/usr/bin/env python3
"""
Comprehensive forecasting script for all columns in CRE.csv using all available models.
Generates forecasts starting from initial seed and recursively using generated predictions.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore")

# Import forecasting functions from each script
sys.path.append('.')

def import_forecast_functions():
    """Import all forecast functions."""
    functions = {}
    
    # Import naive forecasting
    try:
        from walk_forward_naive import walk_forward_forecast as naive_forecast
        functions['naive'] = naive_forecast
    except ImportError:
        print("Warning: Could not import naive forecasting")
    
    # Import NBEATS forecasting
    try:
        from walk_forward_nbeats import walk_forward_nbeats_forecast, load_nbeats_model
        nbeats_model = load_nbeats_model('../models/nbeats_model.pkl')
        if nbeats_model:
            functions['nbeats'] = lambda data, **kwargs: walk_forward_nbeats_forecast(data, nbeats_model, **kwargs)
    except ImportError:
        print("Warning: Could not import NBEATS forecasting")
    
    # Import GBM forecasting
    try:
        from walk_forward_gbm import walk_forward_gbm_forecast, load_gbm_model
        gbm_model = load_gbm_model('../models/gbm_model.pkl')
        if gbm_model:
            functions['gbm'] = lambda data, **kwargs: walk_forward_gbm_forecast(data, gbm_model, **kwargs)
    except ImportError:
        print("Warning: Could not import GBM forecasting")
    
    # Import TFT forecasting
    try:
        from walk_forward_tft import walk_forward_tft_forecast, load_tft_model
        tft_model = load_tft_model('../models/tft_model.pkl')
        if tft_model:
            functions['tft'] = lambda data, **kwargs: walk_forward_tft_forecast(data, tft_model, **kwargs)
    except ImportError:
        print("Warning: Could not import TFT forecasting")
    
    # Import ARIMA v3 forecasting
    try:
        from walk_forward_arima_v3 import walk_forward_arima_v3_forecast, load_arima_v3_model
        arima_v3_model = load_arima_v3_model('../models/arima_v3_model.pkl')
        if arima_v3_model:
            functions['arima_v3'] = lambda data, **kwargs: walk_forward_arima_v3_forecast(data, arima_v3_model, **kwargs)
    except ImportError:
        print("Warning: Could not import ARIMA v3 forecasting")
    
    # Import ARIMA statsmodels forecasting
    try:
        from walk_forward_arima_statsmodels import walk_forward_arima_statsmodels_forecast
        functions['arima_statsmodels'] = walk_forward_arima_statsmodels_forecast
    except ImportError:
        print("Warning: Could not import ARIMA statsmodels forecasting")
    
    return functions

def forecast_column_model(args):
    """Forecast a single column with a single model - for multiprocessing."""
    model_name, forecast_func, column_data, column_name, min_history = args
    
    try:
        print(f"Starting {model_name} forecasting for column {column_name}")
        
        if model_name in ['arima_statsmodels']:
            # These models have different signatures
            forecasts, actuals, errors, successful = forecast_func(column_data, min_history=min_history)
        else:
            forecasts, actuals, errors, successful = forecast_func(column_data, min_history=min_history)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'step': range(min_history, len(column_data)),
            'actual': actuals,
            'forecast': forecasts,
            'absolute_error': errors
        })
        
        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        # Save results
        output_filename = f"{model_name}_column_{column_name}_results.csv"
        results_df.to_csv(output_filename, index=False)
        
        print(f"Completed {model_name} for column {column_name}: MAE={mae:.6f}, RMSE={rmse:.6f}")
        
        return {
            'model': model_name,
            'column': column_name,
            'mae': mae,
            'rmse': rmse,
            'successful_forecasts': successful,
            'total_forecasts': len(forecasts),
            'filename': output_filename
        }
        
    except Exception as e:
        print(f"Error in {model_name} for column {column_name}: {e}")
        return {
            'model': model_name,
            'column': column_name,
            'error': str(e),
            'filename': None
        }

def main():
    print("Comprehensive Multi-Column Forecasting System")
    print("=" * 60)
    
    # Load data
    data_path = '../data/CRE.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    
    # Get numeric columns (exclude time-axis)
    numeric_columns = [col for col in df.columns if col != 'time-axis']
    print(f"Forecasting columns: {numeric_columns[:5]}...{numeric_columns[-5:]} ({len(numeric_columns)} total)")
    
    # Import forecast functions
    forecast_functions = import_forecast_functions()
    print(f"Available models: {list(forecast_functions.keys())}")
    
    if not forecast_functions:
        print("Error: No forecast functions available")
        return
    
    # Prepare arguments for parallel processing
    min_history = 30
    tasks = []
    
    # Limit to first few columns for testing, then expand
    test_columns = numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns
    
    for column_name in test_columns:
        column_data = df[column_name].values
        print(f"Column {column_name} data range: {column_data.min():.6f} to {column_data.max():.6f}")
        
        for model_name, forecast_func in forecast_functions.items():
            tasks.append((model_name, forecast_func, column_data, column_name, min_history))
    
    print(f"\\nStarting {len(tasks)} forecasting tasks...")
    
    # Process tasks - use fewer workers to avoid memory issues
    results = []
    max_workers = min(4, mp.cpu_count())
    
    # Process sequentially for now to avoid memory issues with large models
    for i, task in enumerate(tasks):
        print(f"Processing task {i+1}/{len(tasks)}")
        result = forecast_column_model(task)
        results.append(result)
    
    # Create summary table
    summary_data = []
    for result in results:
        if 'error' not in result:
            summary_data.append({
                'Model': result['model'],
                'Column': result['column'],
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'Successful_Forecasts': result['successful_forecasts'],
                'Total_Forecasts': result['total_forecasts'],
                'Success_Rate': result['successful_forecasts'] / result['total_forecasts'],
                'Filename': result['filename']
            })
        else:
            summary_data.append({
                'Model': result['model'],
                'Column': result['column'],
                'MAE': np.nan,
                'RMSE': np.nan,
                'Successful_Forecasts': 0,
                'Total_Forecasts': 0,
                'Success_Rate': 0,
                'Error': result['error'],
                'Filename': None
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('forecast_summary_all_columns.csv', index=False)
    
    print("\\n" + "=" * 80)
    print("FORECASTING SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\\nSummary saved to: forecast_summary_all_columns.csv")
    
    # Create model performance comparison
    print("\\nModel Performance Summary:")
    if len(summary_df[summary_df['MAE'].notna()]) > 0:
        model_stats = summary_df[summary_df['MAE'].notna()].groupby('Model').agg({
            'MAE': ['mean', 'std', 'min', 'max'],
            'Success_Rate': 'mean'
        })
        print(model_stats.round(6))

if __name__ == "__main__":
    main()