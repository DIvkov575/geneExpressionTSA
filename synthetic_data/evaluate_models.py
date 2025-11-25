#!/usr/bin/env python3
"""
Evaluate different ARIMA models on synthetic data to see if they can recover the true parameters.

This script:
1. Loads the synthetic ARIMA(1,1,1) data (true params: AR=0.7, MA=0.3, d=1)
2. Fits multiple ARIMA models with different orders
3. Compares estimated parameters to the true values
4. Reports which model best recovers the ground truth
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# True parameters used to generate the data
TRUE_AR = [0.7]
TRUE_MA = [0.3]
TRUE_D = 1

def fit_and_evaluate(data, order):
    """Fit ARIMA model and return estimated parameters."""
    p, d, q = order
    try:
        model = ARIMA(data, order=order)
        fitted = model.fit()
        
        # Extract parameters
        ar_params = fitted.arparams if p > 0 else []
        ma_params = fitted.maparams if q > 0 else []
        
        # Calculate AIC/BIC
        aic = fitted.aic
        bic = fitted.bic
        
        return {
            'order': order,
            'ar_params': ar_params,
            'ma_params': ma_params,
            'aic': aic,
            'bic': bic,
            'converged': fitted.mle_retvals['converged']
        }
    except Exception as e:
        return {
            'order': order,
            'ar_params': None,
            'ma_params': None,
            'aic': np.inf,
            'bic': np.inf,
            'converged': False,
            'error': str(e)
        }

def calculate_parameter_error(estimated, true):
    """Calculate RMSE between estimated and true parameters."""
    if estimated is None or len(estimated) == 0:
        return np.inf
    if len(estimated) != len(true):
        return np.inf
    return np.sqrt(np.mean((np.array(estimated) - np.array(true))**2))

if __name__ == "__main__":
    # Load synthetic data
    df = pd.read_csv('synthetic.csv')
    data = df['value'].values
    
    print("="*70)
    print("ARIMA MODEL EVALUATION ON SYNTHETIC DATA")
    print("="*70)
    print(f"\nTrue parameters: ARIMA({len(TRUE_AR)},{TRUE_D},{len(TRUE_MA)})")
    print(f"  AR coefficients: {TRUE_AR}")
    print(f"  MA coefficients: {TRUE_MA}")
    print(f"  Differencing order: {TRUE_D}")
    print(f"\nData: {len(data)} observations")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    print("\n" + "="*70)
    
    # Test different ARIMA orders
    orders_to_test = [
        (0, 1, 0),  # Random walk
        (1, 1, 0),  # ARIMA(1,1,0)
        (0, 1, 1),  # ARIMA(0,1,1)
        (1, 1, 1),  # ARIMA(1,1,1) - TRUE MODEL
        (2, 1, 1),  # ARIMA(2,1,1)
        (1, 1, 2),  # ARIMA(1,1,2)
        (2, 1, 2),  # ARIMA(2,1,2)
    ]
    
    results = []
    for order in orders_to_test:
        result = fit_and_evaluate(data, order)
        results.append(result)
    
    # Print results
    print("\nMODEL COMPARISON")
    print("-"*70)
    print(f"{'Order':<12} {'AIC':>10} {'BIC':>10} {'AR Params':<20} {'MA Params':<20}")
    print("-"*70)
    
    best_aic_idx = np.argmin([r['aic'] for r in results])
    best_bic_idx = np.argmin([r['bic'] for r in results])
    
    for i, result in enumerate(results):
        order_str = f"({result['order'][0]},{result['order'][1]},{result['order'][2]})"
        
        ar_str = ""
        if result['ar_params'] is not None and len(result['ar_params']) > 0:
            ar_str = ", ".join([f"{x:.3f}" for x in result['ar_params']])
        
        ma_str = ""
        if result['ma_params'] is not None and len(result['ma_params']) > 0:
            ma_str = ", ".join([f"{x:.3f}" for x in result['ma_params']])
        
        marker = ""
        if i == best_aic_idx:
            marker += " [Best AIC]"
        if i == best_bic_idx:
            marker += " [Best BIC]"
        if result['order'] == (len(TRUE_AR), TRUE_D, len(TRUE_MA)):
            marker += " [TRUE MODEL]"
        
        print(f"{order_str:<12} {result['aic']:>10.2f} {result['bic']:>10.2f} {ar_str:<20} {ma_str:<20} {marker}")
    
    # Evaluate parameter recovery for the true model
    print("\n" + "="*70)
    print("PARAMETER RECOVERY ANALYSIS")
    print("="*70)
    
    true_model_result = results[3]  # ARIMA(1,1,1)
    
    if true_model_result['ar_params'] is not None:
        ar_error = calculate_parameter_error(true_model_result['ar_params'], TRUE_AR)
        print(f"\nAR Parameter Recovery:")
        print(f"  True:      {TRUE_AR}")
        print(f"  Estimated: {[f'{x:.4f}' for x in true_model_result['ar_params']]}")
        print(f"  RMSE:      {ar_error:.6f}")
    
    if true_model_result['ma_params'] is not None:
        ma_error = calculate_parameter_error(true_model_result['ma_params'], TRUE_MA)
        print(f"\nMA Parameter Recovery:")
        print(f"  True:      {TRUE_MA}")
        print(f"  Estimated: {[f'{x:.4f}' for x in true_model_result['ma_params']]}")
        print(f"  RMSE:      {ma_error:.6f}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    best_model = results[best_bic_idx]
    if best_model['order'] == (len(TRUE_AR), TRUE_D, len(TRUE_MA)):
        print("\n✓ Model selection (BIC) correctly identified the true model order!")
    else:
        print(f"\n✗ Model selection chose ARIMA{best_model['order']} instead of the true model")
    
    if true_model_result['ar_params'] is not None and true_model_result['ma_params'] is not None:
        total_error = ar_error + ma_error
        if total_error < 0.1:
            print("✓ Parameter estimates are very close to true values (RMSE < 0.1)")
        elif total_error < 0.5:
            print("~ Parameter estimates are reasonably close to true values (RMSE < 0.5)")
        else:
            print("✗ Parameter estimates differ significantly from true values")
    
    print("\n" + "="*70)
