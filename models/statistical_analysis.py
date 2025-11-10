import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import List, Dict, Tuple
import os
import warnings

# Suppress specific warnings that might clutter output
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning, module='statsmodels')

def load_data(file_path: str = '../data/CRE.csv') -> List[TimeSeries]:
    """
    Loads data from a CSV file into a list of Darts TimeSeries.
    Converts data to float32 for compatibility.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.astype('float32')
        print(f"Successfully loaded data from {file_path}")
        return [TimeSeries.from_series(df[col]) for col in df.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'. Please ensure the path is correct.")
        return []
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return []

def get_descriptive_stats(series: TimeSeries) -> pd.Series:
    """
    Returns descriptive statistics for a given TimeSeries.
    """
    return series.to_series().describe()

def perform_adf_test(series: TimeSeries) -> Dict:
    """
    Performs the Augmented Dickey-Fuller test on a TimeSeries.
    Returns a dictionary of test results.
    """
    results = {"ADF Statistic": np.nan, "p-value": np.nan, "Critical Values": {}}
    try:
        if series.to_series().nunique() > 1 and len(series) > 8: # ADF needs enough data points
            adf_result = adfuller(series.values().flatten())
            results["ADF Statistic"] = adf_result[0]
            results["p-value"] = adf_result[1]
            results["Critical Values"] = adf_result[4]
            results["Conclusion"] = "Stationary (reject H0)" if adf_result[1] <= 0.05 else "Non-stationary (fail to reject H0)"
        else:
            results["Conclusion"] = "Cannot perform ADF test: Series is constant or too short."
    except Exception as e:
        results["Conclusion"] = f"Error during ADF test: {e}"
    return results

def create_acf_pacf_plots(series: TimeSeries, series_name: str, save_dir: str = 'plots'):
    """
    Generates and saves ACF and PACF plots for a given TimeSeries.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{series_name}_acf_pacf.png')
    
    try:
        if len(series) > 2 and series.to_series().nunique() > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            max_lags = min(len(series) // 2 - 1, 40)
            
            plot_acf(series.values().flatten(), ax=axes[0], lags=max_lags, title=f'ACF for {series_name}')
            plot_pacf(series.values().flatten(), ax=axes[1], lags=max_lags, title=f'PACF for {series_name}')
            
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close(fig)
            print(f"  ACF/PACF plots saved to {file_path}")
        else:
            print(f"  Cannot generate ACF/PACF plots for {series_name}: Not enough data or series is constant.")
    except Exception as e:
        print(f"  Error generating ACF/PACF plots for {series_name}: {e}")

def perform_analysis_and_report():
    # Create a 'plots' directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_abs_path = os.path.join(script_dir, 'plots')
    os.makedirs(plots_abs_path, exist_ok=True)

    all_series = load_data()

    if not all_series:
        print("No data loaded. Aborting analysis.")
        return

    print(f"Performing descriptive statistics for {len(all_series)} series.")
    for i, series in enumerate(all_series):
        print(f"\n--- Series {i+1} (Column: {series.columns[0]}) ---")
        print("Descriptive Statistics:")
        print(get_descriptive_stats(series))

    print(f"\nPerforming ADF tests for {len(all_series)} series.")
    for i, series in enumerate(all_series):
        print(f"\n--- Series {i+1} (Column: {series.columns[0]}) ---")
        adf_results = perform_adf_test(series)
        for key, value in adf_results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.2f}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    print(f"\nGenerating ACF/PACF plots for {len(all_series)} series. Plots will be saved in 'models/plots/'.")
    for i, series in enumerate(all_series):
        series_name = series.columns[0].replace(' ', '_').replace('/', '_')
        create_acf_pacf_plots(series, series_name, save_dir=plots_abs_path)

    print("\nComprehensive statistical analysis complete. Review the output and generated plots in 'models/plots/'.")

if __name__ == '__main__':
    perform_analysis_and_report()
