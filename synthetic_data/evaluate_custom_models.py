#!/usr/bin/env python3
"""
Evaluate custom ARIMA implementations on synthetic data.

Tests:
1. MultiSeriesARIMA (from ARIMA_model.py)
2. DirectARI/MultiHorizonARIMA (from ARIMA_model_v2.py)
3. Standard ARIMA (statsmodels baseline)

Checks if they can recover the true parameters: AR=0.7, MA=0.3, d=1
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from ARIMA_model import MultiSeriesARIMA
from ARIMA_model_v2 import MultiHorizonARIMA
import warnings
warnings.filterwarnings('ignore')

# True parameters
TRUE_AR = 0.7
TRUE_MA = 0.3
TRUE_D = 1

def test_multiseries_arima(data):
    """Test MultiSeriesARIMA model."""
    print("\n" + "="*70)
    print("1. MULTISERIES ARIMA (Custom Implementation)")
    print("="*70)
    
    # MultiSeriesARIMA expects a list of 1D arrays
    data_list = [data]
    
    try:
        # Initialize with p=1, d=1, q=1
        model = MultiSeriesARIMA(p=1, d=1, q=1)
        model.fit(data_list)
        
        # Extract parameters from params_: [c, φ_1,...,φ_p, θ_1,...,θ_q, log(σ²)]
        ar_coef = model.params_[1:1+model.p][0] if model.p > 0 else None
        ma_coef = model.params_[1+model.p:1+model.p+model.q][0] if model.q > 0 else None
        
        print(f"\nEstimated Parameters:")
        print(f"  AR coefficient (φ): {ar_coef:.4f}  [True: {TRUE_AR}]")
        print(f"  MA coefficient (θ): {ma_coef:.4f}  [True: {TRUE_MA}]")
        
        ar_error = abs(ar_coef - TRUE_AR)
        ma_error = abs(ma_coef - TRUE_MA)
        
        print(f"\nParameter Errors:")
        print(f"  AR error: {ar_error:.6f}")
        print(f"  MA error: {ma_error:.6f}")
        print(f"  Total error: {ar_error + ma_error:.6f}")
        
        if ar_error < 0.1 and ma_error < 0.1:
            print("\n✓ Parameters recovered successfully!")
        else:
            print("\n✗ Parameter recovery failed")
            
        return {
            'model': 'MultiSeriesARIMA',
            'ar_est': ar_coef,
            'ma_est': ma_coef,
            'ar_error': ar_error,
            'ma_error': ma_error
        }
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None

def test_direct_ari(data):
    """Test MultiHorizonARIMA model."""
    print("\n" + "="*70)
    print("2. MULTI-HORIZON ARIMA (Direct ARI v2)")
    print("="*70)
    
    try:
        # MultiHorizonARIMA uses p lags and d differencing (no MA component)
        model = MultiHorizonARIMA(p=5, d=1)
        
        # Reshape for MultiHorizonARIMA (expects list of series)
        data_list = [data]
        model.fit(data_list, horizons=[1, 2, 3, 5])
        
        print(f"\nModel Type: AR({model.p}) with differencing order {model.d}")
        print(f"Note: This model does not have MA terms")
        
        # MultiHorizonARIMA trains separate models for each horizon
        print(f"\nTrained {len(model.models)} horizon-specific models")
        print(f"Horizons: {sorted(model.models.keys())}")
        
        # Check if model has coefficients
        if hasattr(model, 'models') and len(model.models) > 0:
            h1_model = model.models.get(1)
            if h1_model is not None and hasattr(h1_model, 'coef_'):
                print(f"\nHorizon-1 AR coefficients (first 5): {h1_model.coef_[:5]}")
        
        print("\n⚠ MultiHorizonARIMA is an AR-only model, cannot compare MA parameters")
        
        return {
            'model': 'MultiHorizonARIMA',
            'ar_est': None,
            'ma_est': None,
            'ar_error': None,
            'ma_error': None,
            'note': 'AR-only model'
        }
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_statsmodels_arima(data):
    """Test statsmodels ARIMA (baseline)."""
    print("\n" + "="*70)
    print("3. STATSMODELS ARIMA (Baseline)")
    print("="*70)
    
    try:
        model = StatsARIMA(data, order=(1, 1, 1))
        fitted = model.fit()
        
        ar_est = fitted.arparams[0]
        ma_est = fitted.maparams[0]
        
        print(f"\nEstimated Parameters:")
        print(f"  AR coefficient (φ): {ar_est:.4f}  [True: {TRUE_AR}]")
        print(f"  MA coefficient (θ): {ma_est:.4f}  [True: {TRUE_MA}]")
        
        ar_error = abs(ar_est - TRUE_AR)
        ma_error = abs(ma_est - TRUE_MA)
        
        print(f"\nParameter Errors:")
        print(f"  AR error: {ar_error:.6f}")
        print(f"  MA error: {ma_error:.6f}")
        print(f"  Total error: {ar_error + ma_error:.6f}")
        
        print(f"\nModel Diagnostics:")
        print(f"  AIC: {fitted.aic:.2f}")
        print(f"  BIC: {fitted.bic:.2f}")
        
        if ar_error < 0.1 and ma_error < 0.1:
            print("\n✓ Parameters recovered successfully!")
        else:
            print("\n✗ Parameter recovery failed")
            
        return {
            'model': 'StatsARIMA',
            'ar_est': ar_est,
            'ma_est': ma_est,
            'ar_error': ar_error,
            'ma_error': ma_error
        }
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None

if __name__ == "__main__":
    # Load synthetic data
    df = pd.read_csv('synthetic.csv')
    data = df['value'].values
    
    print("="*70)
    print("CUSTOM ARIMA MODEL EVALUATION ON SYNTHETIC DATA")
    print("="*70)
    print(f"\nTrue parameters: ARIMA(1,{TRUE_D},1)")
    print(f"  AR coefficient: {TRUE_AR}")
    print(f"  MA coefficient: {TRUE_MA}")
    print(f"  Differencing order: {TRUE_D}")
    print(f"\nData: {len(data)} observations")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Test all models
    results = []
    
    # Test MultiSeriesARIMA
    result1 = test_multiseries_arima(data)
    if result1:
        results.append(result1)
    
    # Test DirectARI
    result2 = test_direct_ari(data)
    if result2:
        results.append(result2)
    
    # Test statsmodels baseline
    result3 = test_statsmodels_arima(data)
    if result3:
        results.append(result3)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<25} {'AR Est':<12} {'MA Est':<12} {'AR Error':<12} {'MA Error':<12}")
    print("-"*70)
    
    for r in results:
        ar_str = f"{r['ar_est']:.4f}" if r['ar_est'] is not None else "N/A"
        ma_str = f"{r['ma_est']:.4f}" if r['ma_est'] is not None else "N/A"
        ar_err_str = f"{r['ar_error']:.6f}" if r['ar_error'] is not None else "N/A"
        ma_err_str = f"{r['ma_error']:.6f}" if r['ma_error'] is not None else "N/A"
        
        print(f"{r['model']:<25} {ar_str:<12} {ma_str:<12} {ar_err_str:<12} {ma_err_str:<12}")
    
    print("\n" + "="*70)
