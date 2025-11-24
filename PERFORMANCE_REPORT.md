# MultiSeriesARIMA Performance Report

## Model Configuration
- **Model**: ARIMA(1,1,1)
- **Training Samples**: 1,000 windows
- **Test Samples**: 23,080 windows
- **Window Size**: 25 (24 training points + 1 test point)

## Fitted Parameters
- **Constant**: -0.000007
- **AR Coefficient (φ₁)**: 0.976588
- **MA Coefficient (θ₁)**: 0.194453
- **Error Variance (σ²)**: 0.000004

## Performance Metrics

### Normalized Metrics
| Metric | ARIMA | Naive Baseline |
|--------|-------|----------------|
| MAPE   | 2.44% | 50.19%        |
| NRMSE  | 0.05% | 0.23%         |

### Absolute Metrics
| Metric | ARIMA    | Naive Baseline |
|--------|----------|----------------|
| MSE    | 0.000004 | 0.000097      |
| RMSE   | 0.002052 | 0.009836      |
| MAE    | 0.000451 | 0.004436      |

## Summary
- **ARIMA achieves 2.44% MAPE** - on average, predictions are off by only 2.44%
- **95.1% improvement** over the Naive baseline
- Model successfully captures temporal patterns in the data
- No data leakage detected (verified through code audit)

## Date
2025-11-24
