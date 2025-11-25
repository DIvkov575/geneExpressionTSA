# Implementation Plan - ARIMA v4 (ARIMAX)

## Goal
Create `ARIMA_model_v4.py` which extends the v3 implementation to support exogenous variables (ARIMAX). This allows the model to include external regressors (e.g., weather, holidays, economic indicators) in the ARIMA equation.

## User Review Required
> [!IMPORTANT]
> The `fit` and `forecast` methods will now require exogenous data if `exog_dim > 0`. This changes the API signature compared to v3.

## Proposed Changes

### [New File] ARIMA_model_v4.py

Create a new class `MultiHorizonARIMAX` inheriting from or based on `MultiHorizonARIMA_v3`.

#### Key Modifications:

1.  **`__init__(self, p, d, q, exog_dim=0)`**
    *   Add `exog_dim` to store the number of exogenous variables.

2.  **`compute_residuals(self, z, exog, params)`**
    *   Update equation: `e[t] = z[t] - (c + AR_term + MA_term + Exog_term)`
    *   `Exog_term = dot(beta, exog[t])`

3.  **`neg_log_likelihood(self, params, series_list, exog_list)`**
    *   Update parameter extraction to include `beta` coefficients.
    *   Pass `exog` to `compute_residuals`.

4.  **`fit(self, series_list, exog_list=None, ...)`**
    *   Validate `exog_list` matches `series_list` in length and time steps.
    *   Initialize `beta` parameters (can use 0.0 or simple regression).
    *   Pass `exog_list` to the optimizer.

5.  **`forecast(self, series, exog_history=None, exog_future=None, steps=1, ...)`**
    *   Require `exog_future` if `exog_dim > 0`.
    *   Use `exog_history` for residual computation.
    *   Use `exog_future` for generating forecasts.

6.  **`get_params(self)`**
    *   Include `exog_coefs` in the returned dictionary.

## Verification Plan

### Automated Tests (`test_arima_v4.py`)
1.  **Parameter Recovery (Synthetic ARIMAX)**:
    *   Generate synthetic data: `y[t] = 0.5*y[t-1] + 2.0*x[t] + e[t]`
    *   Fit v4 model and check if it recovers `beta=2.0` and `phi=0.5`.
2.  **Forecasting with Exog**:
    *   Verify forecast shape and values using known exogenous inputs.
3.  **API Compatibility**:
    *   Ensure it works like v3 when `exog_dim=0`.

### Manual Verification
*   Run the test script and check output.
