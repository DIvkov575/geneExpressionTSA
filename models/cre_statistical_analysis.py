
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import os

# --- 0. Setup Paths ---
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_path = os.path.join(project_root, 'data', 'CRE.csv')
plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)


# --- 1. Data Loading and Preparation ---
# Load the dataset
df = pd.read_csv(data_path)

# Convert 'time-axis' to datetime objects
df['time-axis'] = pd.to_datetime(df['time-axis'], unit='s')

# Set 'time-axis' as the index
df.set_index('time-axis', inplace=True)

# Select the time series to analyze (column '1')
# We are assuming the analysis is for the first series.
time_series = df['1'].dropna()

print("--- Data Loading and Preparation ---")
print("Dataset loaded successfully.")
print(f"Time series range: {time_series.index.min()} to {time_series.index.max()}")
print(f"Number of observations: {len(time_series)}")
print("\n")


# --- 2. Descriptive Statistics ---
print("--- 2. Descriptive Statistics ---")
print(time_series.describe())
print("\n")


# --- 3. Stationarity Test (Augmented Dickey-Fuller Test) ---
print("--- 3. Stationarity Test (ADF) ---")
adf_result = adfuller(time_series)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value}')

if adf_result[1] > 0.05:
    print("Result: The series is likely non-stationary (p-value > 0.05).")
else:
    print("Result: The series is likely stationary (p-value <= 0.05).")
print("\n")


# --- 4. Time Series Decomposition ---
# We need to determine the seasonal period. Let's assume a period based on data frequency or inspection.
# For this generic analysis, we'll try a period of 12, a common starting point.
# A more advanced approach would involve spectral analysis or inspecting the ACF plot.
print("--- 4. Time Series Decomposition ---")
decomposition = sm.tsa.seasonal_decompose(time_series, model='additive', period=12)

fig = decomposition.plot()
plt.suptitle('Time Series Decomposition', y=1.02)
fig.set_size_inches(10, 8)
plt.tight_layout()
decomposition_plot_path = os.path.join(plot_dir, 'cre_decomposition.png')
plt.savefig(decomposition_plot_path)
plt.close()
print(f"Decomposition plot saved to '{decomposition_plot_path}'")
print("\n")


# --- 5. Autocorrelation and Partial Autocorrelation ---
print("--- 5. Autocorrelation Analysis ---")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
sm.graphics.tsa.plot_acf(time_series, lags=40, ax=ax1)
sm.graphics.tsa.plot_pacf(time_series, lags=40, ax=ax2)
plt.tight_layout()
acf_pacf_plot_path = os.path.join(plot_dir, 'cre_acf_pacf.png')
plt.savefig(acf_pacf_plot_path)
plt.close()
print(f"ACF and PACF plots saved to '{acf_pacf_plot_path}'")
print("\n")


# --- 6. Normality Test (Shapiro-Wilk Test) ---
# This tests if the data is drawn from a normal distribution.
# Often applied to the residuals of a model, but can be checked on the series itself.
print("--- 6. Normality Test (Shapiro-Wilk) ---")
shapiro_stat, shapiro_p = shapiro(time_series)
print(f"Shapiro-Wilk Statistic: {shapiro_stat}")
print(f"P-value: {shapiro_p}")

if shapiro_p > 0.05:
    print("Result: The data appears to be normally distributed (p-value > 0.05).")
else:
    print("Result: The data does not appear to be normally distributed (p-value <= 0.05).")
print("\n")

print("--- Analysis Complete ---")
