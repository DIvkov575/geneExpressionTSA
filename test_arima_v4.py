import numpy as np
import pandas as pd
from ARIMA_model_v4 import MultiHorizonARIMAX
import unittest

class TestARIMAX(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
    def generate_arimax_data(self, n=200, phi=0.5, beta=2.0, sigma=0.1):
        """
        Generate ARIMAX(1,0,0) data:
        y[t] = phi * y[t-1] + beta * x[t] + e[t]
        """
        x = np.random.randn(n) # Exogenous variable
        e = np.random.normal(0, sigma, n)
        y = np.zeros(n)
        
        for t in range(1, n):
            y[t] = phi * y[t-1] + beta * x[t] + e[t]
            
        return y, x

    def test_parameter_recovery(self):
        print("\nTesting Parameter Recovery (ARIMAX)...")
        TRUE_PHI = 0.5
        TRUE_BETA = 2.0
        
        y, x = self.generate_arimax_data(n=500, phi=TRUE_PHI, beta=TRUE_BETA)
        
        # Reshape x for model
        x = x.reshape(-1, 1)
        
        model = MultiHorizonARIMAX(p=1, d=0, q=0, exog_dim=1)
        model.fit([y], exog_list=[x])
        
        params = model.get_params()
        est_phi = params['ar_coefs'][0]
        est_beta = params['exog_coefs'][0]
        
        print(f"True Phi: {TRUE_PHI}, Estimated: {est_phi:.4f}")
        print(f"True Beta: {TRUE_BETA}, Estimated: {est_beta:.4f}")
        
        self.assertAlmostEqual(est_phi, TRUE_PHI, delta=0.05)
        self.assertAlmostEqual(est_beta, TRUE_BETA, delta=0.05)
        
    def test_forecast_shape(self):
        print("\nTesting Forecast Shape...")
        y, x = self.generate_arimax_data(n=100)
        x = x.reshape(-1, 1)
        
        model = MultiHorizonARIMAX(p=1, d=0, q=0, exog_dim=1)
        model.fit([y], exog_list=[x])
        
        # Forecast 5 steps ahead
        # Need 5 steps of future exogenous data
        x_future = np.random.randn(5).reshape(-1, 1)
        
        forecast = model.forecast(y, exog_history=x, exog_future=x_future, steps=5)
        
        self.assertEqual(len(forecast), 5)
        print("Forecast shape correct.")
        
    def test_no_exog_compatibility(self):
        print("\nTesting Backward Compatibility (No Exog)...")
        # Generate simple AR(1) data
        y, _ = self.generate_arimax_data(n=100, beta=0.0)
        
        model = MultiHorizonARIMAX(p=1, d=0, q=0, exog_dim=0)
        model.fit([y]) # No exog_list needed
        
        params = model.get_params()
        self.assertEqual(len(params['exog_coefs']), 0)
        
        forecast = model.forecast(y, steps=5)
        self.assertEqual(len(forecast), 5)
        print("Backward compatibility confirmed.")

if __name__ == '__main__':
    unittest.main()
