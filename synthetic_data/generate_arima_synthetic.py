import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def generate_arima_data(n_samples=1000, ar_params=None, ma_params=None, d=1, 
                        mean=0, std=1, seed=42):
    """
    Generate synthetic time series data using a pre-determined ARIMA model.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points to generate
    ar_params : list or None
        AR coefficients (excluding the leading 1). For ARIMA(1,1,1), use [0.5] for AR(1)
    ma_params : list or None
        MA coefficients (excluding the leading 1). For ARIMA(1,1,1), use [0.5] for MA(1)
    d : int
        Degree of differencing (1 for ARIMA(1,1,1))
    mean : float
        Mean of the series
    std : float
        Standard deviation of the noise
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame : DataFrame with generated time series
    """
    np.random.seed(seed)
    
    # Default parameters for ARIMA(1,1,1) if not provided
    if ar_params is None:
        ar_params = [0.7]  # AR(1) coefficient
    if ma_params is None:
        ma_params = [0.3]  # MA(1) coefficient
    
    # Create ARMA process (before differencing)
    # Note: statsmodels expects coefficients with opposite sign for AR
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]
    
    # Generate ARMA process
    arma_process = ArmaProcess(ar, ma)
    
    # Generate more samples to account for differencing
    extra_samples = d * 10
    arma_data = arma_process.generate_sample(nsample=n_samples + extra_samples, scale=std)
    
    # Apply differencing d times (integrate)
    integrated_data = arma_data.copy()
    for _ in range(d):
        integrated_data = np.cumsum(integrated_data)
    
    # Add mean and take the last n_samples
    integrated_data = integrated_data[-n_samples:] + mean
    
    return integrated_data

def create_synthetic_dataset(n_series=5, n_samples=1000, output_file='synthetic_arima_data.csv',
                            ar_params=None, ma_params=None, d=1, seed=42):
    """
    Create a synthetic dataset with multiple time series columns.
    
    Parameters:
    -----------
    n_series : int
        Number of time series to generate
    n_samples : int
        Number of data points per series
    output_file : str
        Output CSV filename
    ar_params : list or None
        AR coefficients for the ARIMA model
    ma_params : list or None
        MA coefficients for the ARIMA model
    d : int
        Degree of differencing
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame : DataFrame with all generated series
    """
    np.random.seed(seed)
    
    data = {}
    
    for i in range(1, n_series + 1):
        # Vary parameters slightly for each series to add diversity
        series_seed = seed + i
        series_mean = np.random.uniform(-10, 10)
        series_std = np.random.uniform(0.5, 2.0)
        
        # Generate series
        series_data = generate_arima_data(
            n_samples=n_samples,
            ar_params=ar_params,
            ma_params=ma_params,
            d=d,
            mean=series_mean,
            std=series_std,
            seed=series_seed
        )
        
        data[str(i)] = series_data
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {n_series} synthetic time series with {n_samples} samples each")
    print(f"✓ Saved to: {output_file}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def visualize_synthetic_data(df, n_plots=5, output_file='synthetic_data_plot.png'):
    """
    Visualize the generated synthetic time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with synthetic time series
    n_plots : int
        Number of series to plot
    output_file : str
        Output filename for the plot
    """
    n_plots = min(n_plots, len(df.columns))
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, col in enumerate(df.columns[:n_plots]):
        axes[i].plot(df[col], linewidth=1.5, color='steelblue')
        axes[i].set_title(f'Synthetic Time Series - Column {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    # Configuration matching your ARIMA(1,1,1) model
    print("="*60)
    print("  SYNTHETIC DATA GENERATION USING ARIMA(1,1,1)")
    print("="*60)
    print()
    
    # Pre-determined ARIMA parameters
    AR_PARAMS = [0.7]   # AR(1) coefficient
    MA_PARAMS = [0.3]   # MA(1) coefficient
    D = 1               # First-order differencing
    
    # Dataset configuration
    N_SERIES = 5        # Number of time series to generate
    N_SAMPLES = 1000    # Number of data points per series
    SEED = 42           # Random seed for reproducibility
    
    OUTPUT_FILE = 'synthetic_data/synthetic_arima_data.csv'
    PLOT_FILE = 'synthetic_data/synthetic_data_plot.png'
    
    print(f"Model Configuration:")
    print(f"  - ARIMA Order: (1, {D}, 1)")
    print(f"  - AR Parameters: {AR_PARAMS}")
    print(f"  - MA Parameters: {MA_PARAMS}")
    print(f"\nDataset Configuration:")
    print(f"  - Number of series: {N_SERIES}")
    print(f"  - Samples per series: {N_SAMPLES}")
    print(f"  - Random seed: {SEED}")
    print()
    print("-"*60)
    print()
    
    # Generate synthetic dataset
    df = create_synthetic_dataset(
        n_series=N_SERIES,
        n_samples=N_SAMPLES,
        output_file=OUTPUT_FILE,
        ar_params=AR_PARAMS,
        ma_params=MA_PARAMS,
        d=D,
        seed=SEED
    )
    
    # Visualize the data
    visualize_synthetic_data(df, n_plots=N_SERIES, output_file=PLOT_FILE)
    
    print()
    print("="*60)
    print("  GENERATION COMPLETE")
    print("="*60)
