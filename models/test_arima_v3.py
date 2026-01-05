#!/usr/bin/env python3
"""
Test ARIMA_model_v3 to verify MA component support.
Compare with MultiSeriesARIMA and validate parameter recovery.
"""
import sys
import numpy as np
import pandas as pd
from ARIMA_model_v3 import MultiHorizonARIMA_v3
from ARIMA_model import MultiSeriesARIMA
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TESTING ARIMA_MODEL_V3 WITH MA COMPONENT SUPPORT")
print("="*70)

# Load synthetic data
df = pd.read_csv('synthetic_data/synthetic.csv')
data = df['value'].values

TRUE_AR = 0.7
TRUE_MA = 0.3
TRUE_D = 1

print(f"\nSynthetic Data: ARIMA(1,{TRUE_D},1)")
print(f"  True AR: {TRUE_AR}")
print(f"  True MA: {TRUE_MA}")
print(f"  Sample size: {len(data)}")
print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")

# Test 1: MultiHorizonARIMA_v3
print("\n" + "="*70)
print("TEST 1: MultiHorizonARIMA_v3 (New Implementation)")
print("="*70)

try:
    model_v3 = MultiHorizonARIMA_v3(p=1, d=1, q=1)
    model_v3.fit([data], maxiter=1000)
    
    params_v3 = model_v3.get_params()
    ar_v3 = params_v3['ar_coefs'][0]
    ma_v3 = params_v3['ma_coefs'][0]
    
    print(f"\nEstimated Parameters:")
    print(f"  AR coefficient (φ): {ar_v3:.4f}  [True: {TRUE_AR}]")
    print(f"  MA coefficient (θ): {ma_v3:.4f}  [True: {TRUE_MA}]")
    print(f"  Constant (c):       {params_v3['constant']:.6f}")
    print(f"  Error variance (σ²): {params_v3['sigma2']:.6f}")
    
    ar_error_v3 = abs(ar_v3 - TRUE_AR)
    ma_error_v3 = abs(ma_v3 - TRUE_MA)
    
    print(f"\nParameter Errors:")
    print(f"  AR error: {ar_error_v3:.6f}")
    print(f"  MA error: {ma_error_v3:.6f}")
    print(f"  Total error: {ar_error_v3 + ma_error_v3:.6f}")
    
    if ar_error_v3 < 0.1 and ma_error_v3 < 0.1:
        print("\n✓ Parameters recovered successfully!")
    else:
        print("\n✗ Parameter recovery failed")
    
    # Test forecasting
    print("\nTesting Forecasting:")
    forecast_v3 = model_v3.forecast(data, steps=5)
    print(f"  5-step forecast: {forecast_v3}")
    
    # Test in-sample predictions
    print("\nTesting In-Sample Predictions:")
    predictions_v3 = model_v3.predict_in_sample(data)
    mse_v3 = np.mean((data - predictions_v3)**2)
    print(f"  In-sample MSE: {mse_v3:.6f}")
    
    model_v3.summary()
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Compare with MultiSeriesARIMA
print("\n" + "="*70)
print("TEST 2: MultiSeriesARIMA (Original Implementation)")
print("="*70)

try:
    model_orig = MultiSeriesARIMA(p=1, d=1, q=1)
    model_orig.fit([data], maxiter=1000)
    
    params_orig = model_orig.get_params()
    ar_orig = params_orig['ar_coefs'][0]
    ma_orig = params_orig['ma_coefs'][0]
    
    print(f"\nEstimated Parameters:")
    print(f"  AR coefficient (φ): {ar_orig:.4f}  [True: {TRUE_AR}]")
    print(f"  MA coefficient (θ): {ma_orig:.4f}  [True: {TRUE_MA}]")
    
    ar_error_orig = abs(ar_orig - TRUE_AR)
    ma_error_orig = abs(ma_orig - TRUE_MA)
    
    print(f"\nParameter Errors:")
    print(f"  AR error: {ar_error_orig:.6f}")
    print(f"  MA error: {ma_error_orig:.6f}")
    print(f"  Total error: {ar_error_orig + ma_error_orig:.6f}")
    
    # Test forecasting
    forecast_orig = model_orig.forecast(data, steps=5)
    print(f"\n5-step forecast: {forecast_orig}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")

# Test 3: StatsModels baseline
print("\n" + "="*70)
print("TEST 3: StatsModels ARIMA (Baseline)")
print("="*70)

try:
    model_stats = StatsARIMA(data, order=(1, 1, 1))
    fitted_stats = model_stats.fit()
    
    ar_stats = fitted_stats.arparams[0]
    ma_stats = fitted_stats.maparams[0]
    
    print(f"\nEstimated Parameters:")
    print(f"  AR coefficient (φ): {ar_stats:.4f}  [True: {TRUE_AR}]")
    print(f"  MA coefficient (θ): {ma_stats:.4f}  [True: {TRUE_MA}]")
    
    ar_error_stats = abs(ar_stats - TRUE_AR)
    ma_error_stats = abs(ma_stats - TRUE_MA)
    
    print(f"\nParameter Errors:")
    print(f"  AR error: {ar_error_stats:.6f}")
    print(f"  MA error: {ma_error_stats:.6f}")
    print(f"  Total error: {ar_error_stats + ma_error_stats:.6f}")
    
    # Test forecasting
    forecast_stats = fitted_stats.forecast(steps=5)
    print(f"\n5-step forecast: {forecast_stats.values}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")

# Comparison Summary
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print(f"\n{'Model':<30} {'AR Est':<12} {'MA Est':<12} {'AR Error':<12} {'MA Error':<12}")
print("-"*70)

try:
    print(f"{'MultiHorizonARIMA_v3':<30} {ar_v3:<12.4f} {ma_v3:<12.4f} {ar_error_v3:<12.6f} {ma_error_v3:<12.6f}")
except:
    print(f"{'MultiHorizonARIMA_v3':<30} {'FAILED':<12} {'FAILED':<12} {'N/A':<12} {'N/A':<12}")

try:
    print(f"{'MultiSeriesARIMA':<30} {ar_orig:<12.4f} {ma_orig:<12.4f} {ar_error_orig:<12.6f} {ma_error_orig:<12.6f}")
except:
    print(f"{'MultiSeriesARIMA':<30} {'FAILED':<12} {'FAILED':<12} {'N/A':<12} {'N/A':<12}")

try:
    print(f"{'StatsModels ARIMA':<30} {ar_stats:<12.4f} {ma_stats:<12.4f} {ar_error_stats:<12.6f} {ma_error_stats:<12.6f}")
except:
    print(f"{'StatsModels ARIMA':<30} {'FAILED':<12} {'FAILED':<12} {'N/A':<12} {'N/A':<12}")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)

# Test with different ARIMA orders
print("\n" + "="*70)
print("ADDITIONAL TESTS: Different ARIMA Orders")
print("="*70)

test_orders = [
    (1, 1, 0),  # AR only
    (0, 1, 1),  # MA only
    (2, 1, 1),  # Higher order AR
    (1, 1, 2),  # Higher order MA
]

for p, d, q in test_orders:
    print(f"\nTesting ARIMA({p},{d},{q}):")
    try:
        model_test = MultiHorizonARIMA_v3(p=p, d=d, q=q)
        model_test.fit([data], maxiter=500)
        params_test = model_test.get_params()
        
        print(f"  ✓ Model fitted successfully")
        if p > 0:
            print(f"  AR coefficients: {params_test['ar_coefs']}")
        if q > 0:
            print(f"  MA coefficients: {params_test['ma_coefs']}")
        
        # Test forecast
        forecast_test = model_test.forecast(data, steps=3)
        print(f"  3-step forecast: {forecast_test}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "="*70)
print("ALL TESTS COMPLETE")
print("="*70)
