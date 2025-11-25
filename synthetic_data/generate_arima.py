# synthetic_data/generate_arima.py
"""Generate a synthetic time series using statsmodels ARIMA.
This script generates an ARIMA(p,d,q) series and writes it to synthetic.csv.
"""
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def generate_arima_series(
    n_samples: int = 100,
    ar_params: list = [0.7],   # AR coefficients
    ma_params: list = [0.3],   # MA coefficients
    d: int = 1,                # Differencing order
    sigma: float = 1.0,        # Noise std dev
    seed: int = 42,
) -> pd.DataFrame:
    """Generate an ARIMA(p,d,q) series using statsmodels.

    Parameters
    ----------
    n_samples : int
        Length of the final series.
    ar_params : list
        AR coefficients (e.g., [0.7] for AR(1)).
    ma_params : list
        MA coefficients (e.g., [0.3] for MA(1)).
    d : int
        Differencing order.
    sigma : float
        Standard deviation of white noise.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    
    # statsmodels expects sign-reversed AR coefficients with leading 1
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]
    
    # Create ARMA process
    arma = ArmaProcess(ar, ma)
    
    # Generate the series (add extra samples for differencing)
    raw = arma.generate_sample(nsample=n_samples + d, scale=sigma)
    
    # Apply differencing if d > 0
    if d > 0:
        # Integrate the series d times (cumsum)
        series = raw
        for _ in range(d):
            series = np.cumsum(series)
    else:
        series = raw
    
    # Trim to requested length
    series = series[:n_samples]
    
    # Create DataFrame
    df = pd.DataFrame({
        "time": np.arange(len(series)),
        "value": series
    })
    return df


if __name__ == "__main__":
    # Generate ARIMA(1,1,1) series
    df = generate_arima_series(
        n_samples=100,
        ar_params=[0.7],
        ma_params=[0.3],
        d=1,
        sigma=1.0,
        seed=42,
    )
    
    # Save to CSV
    out_path = os.path.join(os.path.dirname(__file__), "synthetic.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic ARIMA series written to {out_path}")
    print(f"Generated {len(df)} samples")
    print(f"Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")
