#!/usr/bin/env python3
"""
Efficient multi-column forecasting script for CRE.csv using fast models.
Focuses on Naive and ARIMA statsmodels to avoid timeout issues.
Generates recursive forecasts using only seed data + model predictions.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.append('.')

def load_fast_models():
    """Load only the fast forecasting models."""
    models = {}
    
    # Import naive forecasting
    try:
        from walk_forward_naive import walk_forward_forecast
        models['naive'] = walk_forward_forecast
        print("✓ Loaded Naive model")
    except ImportError as e:
        print(f"✗ Failed to load Naive model: {e}")
    
    # Import ARIMA statsmodels forecasting  
    try:
        from walk_forward_arima_statsmodels import walk_forward_arima_statsmodels_forecast
        models['arima_statsmodels'] = walk_forward_arima_statsmodels_forecast
        print("✓ Loaded ARIMA Statsmodels model")
    except ImportError as e:
        print(f"✗ Failed to load ARIMA Statsmodels model: {e}")
    
    return models

def forecast_single_column_model(model_name, forecast_func, column_data, column_name, min_history=30):
    """Forecast a single column with a single model."""
    try:
        print(f"  Starting {model_name} for column {column_name}...")
        start_time = datetime.now()
        
        # Perform forecasting
        forecasts, actuals, errors, successful = forecast_func(column_data, min_history=min_history)
        
        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'step': range(min_history, len(column_data)),
            'actual': actuals,
            'forecast': forecasts,
            'absolute_error': errors
        })
        
        # Save results
        output_filename = f"{model_name}_column_{column_name}_results.csv"
        results_df.to_csv(output_filename, index=False)
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"  ✓ Completed {model_name} for column {column_name} in {duration:.1f}s")
        print(f"    MAE: {mae:.6f}, RMSE: {rmse:.6f}, Max Error: {max_error:.6f}")
        
        return {
            'model': model_name,
            'column': column_name,
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'min_error': min_error,
            'successful_forecasts': successful,
            'total_forecasts': len(forecasts),
            'success_rate': successful / len(forecasts) if len(forecasts) > 0 else 0,
            'filename': output_filename,
            'duration_seconds': duration
        }
        
    except Exception as e:
        print(f"  ✗ Error in {model_name} for column {column_name}: {e}")
        return {
            'model': model_name,
            'column': column_name,
            'error': str(e),
            'filename': None,
            'duration_seconds': 0
        }

def main():
    print("🚀 Efficient Multi-Column Forecasting System")
    print("=" * 60)
    
    # Load data
    data_path = '../data/CRE.csv'
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"📊 Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Get numeric columns (exclude time-axis)
    numeric_columns = [col for col in df.columns if col != 'time-axis']
    print(f"📈 Processing {len(numeric_columns)} columns: {numeric_columns[:5]}...{numeric_columns[-5:]}")
    
    # Load fast models
    print("\\n🔧 Loading models...")
    models = load_fast_models()
    
    if not models:
        print("❌ No models available")
        return
    
    print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
    
    # Process all combinations
    min_history = 30
    results = []
    
    total_tasks = len(numeric_columns) * len(models)
    completed_tasks = 0
    
    print(f"\\n📋 Processing {total_tasks} forecasting tasks...")
    print("-" * 60)
    
    for i, column_name in enumerate(numeric_columns):
        column_data = df[column_name].values
        data_range = column_data.max() - column_data.min()
        
        print(f"\\n📊 Column {column_name} ({i+1}/{len(numeric_columns)}):")
        print(f"   Range: [{column_data.min():.6f}, {column_data.max():.6f}], Span: {data_range:.6f}")
        
        for model_name, forecast_func in models.items():
            result = forecast_single_column_model(
                model_name, forecast_func, column_data, column_name, min_history
            )
            results.append(result)
            completed_tasks += 1
            
            progress = (completed_tasks / total_tasks) * 100
            print(f"   Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%)")
    
    # Create summary table
    print("\\n📋 Creating summary table...")
    summary_data = []
    
    for result in results:
        if 'error' not in result:
            summary_data.append({
                'Model': result['model'],
                'Column': result['column'],
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'Max_Error': result['max_error'],
                'Min_Error': result['min_error'],
                'Success_Rate': result['success_rate'],
                'Duration_Seconds': result['duration_seconds'],
                'Filename': result['filename']
            })
        else:
            summary_data.append({
                'Model': result['model'],
                'Column': result['column'],
                'MAE': np.nan,
                'RMSE': np.nan,
                'Max_Error': np.nan,
                'Min_Error': np.nan,
                'Success_Rate': 0.0,
                'Duration_Seconds': result['duration_seconds'],
                'Error': result['error'],
                'Filename': None
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = 'forecast_summary_all_columns.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    print("\\n" + "=" * 80)
    print("📊 FORECASTING SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    successful_forecasts = len(summary_df[summary_df['MAE'].notna()])
    total_attempts = len(summary_df)
    overall_success_rate = (successful_forecasts / total_attempts) * 100
    
    print(f"📈 Overall Results:")
    print(f"   Successful forecasts: {successful_forecasts}/{total_attempts} ({overall_success_rate:.1f}%)")
    
    if successful_forecasts > 0:
        valid_results = summary_df[summary_df['MAE'].notna()]
        total_duration = valid_results['Duration_Seconds'].sum()
        avg_duration = valid_results['Duration_Seconds'].mean()
        
        print(f"   Total processing time: {total_duration:.1f}s")
        print(f"   Average time per forecast: {avg_duration:.1f}s")
        print(f"   Average MAE: {valid_results['MAE'].mean():.6f}")
        print(f"   Average RMSE: {valid_results['RMSE'].mean():.6f}")
    
    # Model performance comparison
    if successful_forecasts > 0:
        print(f"\\n📊 Model Performance Summary:")
        model_stats = valid_results.groupby('Model').agg({
            'MAE': ['count', 'mean', 'std', 'min', 'max'],
            'RMSE': 'mean',
            'Duration_Seconds': 'mean'
        }).round(6)
        print(model_stats)
    
    # Column-wise analysis (top and bottom performers)
    if successful_forecasts > 0:
        print(f"\\n📊 Column Analysis:")
        column_avg = valid_results.groupby('Column')['MAE'].mean().sort_values()
        
        print("   Best performing columns (lowest MAE):")
        for i, (col, mae) in enumerate(column_avg.head().items()):
            print(f"      {i+1}. Column {col}: {mae:.6f}")
        
        print("   Worst performing columns (highest MAE):")
        for i, (col, mae) in enumerate(column_avg.tail().items()):
            print(f"      {i+1}. Column {col}: {mae:.6f}")
    
    print(f"\\n💾 Summary saved to: {summary_filename}")
    print("\\n🎉 Forecasting completed successfully!")

if __name__ == "__main__":
    main()