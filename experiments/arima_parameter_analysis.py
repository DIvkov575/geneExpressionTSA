import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Configuration ---
# Define the path to your time series data
# Based on your project structure, it could be 'data/CRE.csv' or 'data/FRAM.csv'
DATA_PATH = 'data/CRE.csv'
DATE_COLUMN = 'time-axis'  # Name of the column containing date/time information
VALUE_COLUMN = '1' # Name of the column containing the time series values

# --- 1. Load and Prepare Data ---
def load_data(path, date_col, value_col):
    """Loads data, sets date as index, and returns the time series."""
    try:
        df = pd.read_csv(path, index_col=date_col)
        series = df[value_col]
        series.index = pd.RangeIndex(start=0, stop=len(series), step=1) # Reset to integer index
        print(f"Data loaded successfully from {path}. Shape: {df.shape}")
        return series
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None
    except KeyError as e:
        print(f"Error: Missing expected column in data: {e}")
        return None

# --- 2. Visualize Time Series and Decomposition ---
def visualize_series(series):
    """Plots the time series and its seasonal decomposition."""
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel(VALUE_COLUMN)
    plt.grid(True)
    plt.show()

    # Perform seasonal decomposition (additive or multiplicative)
    # Adjust model='additive' or 'multiplicative' based on your data's characteristics
    # For example, if seasonality increases with the level of the series, use 'multiplicative'.
    # You might need to specify a period if your data doesn't have a default frequency set.
    # try:
    #     decomposition = seasonal_decompose(series, model='additive', period=365) # Assuming yearly seasonality for daily data
    #     decomposition.plot()
    #     plt.suptitle('Time Series Decomposition', y=1.02)
    #     plt.tight_layout()
    #     plt.show()
    # except Exception as e:
    #     print(f"Could not perform seasonal decomposition. Ensure data frequency is set and period is appropriate. Error: {e}")


# --- 3. Determine 'd' (Differencing Order) ---
def plot_differenced_series(series, lags=1):
    """Plots the differenced series to help determine 'd'."""
    differenced_series = series.diff(lags).dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(differenced_series)
    plt.title(f'Differenced Time Series (Lag={lags})')
    plt.xlabel('Date')
    plt.ylabel(f'{VALUE_COLUMN} Differenced')
    plt.grid(True)
    plt.show()
    return differenced_series

# --- 4. Determine 'p' and 'q' (AR and MA Orders) ---
def plot_acf_pacf(series, lags=40):
    """Plots ACF and PACF to determine p and q orders."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(series, lags=lags, ax=axes[0], title='Autocorrelation Function (ACF)')
    plot_pacf(series, lags=lags, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.show()
    print("Examine ACF and PACF plots to determine p and q:")
    print("  - p (AR order): Look for the lag where PACF cuts off or drops significantly.")
    print("  - q (MA order): Look for the lag where ACF cuts off or drops significantly.")

# --- 5. ARIMA Model Fitting and Evaluation (Example) ---
def fit_and_evaluate_arima(series, order=(5,1,0), train_size_ratio=0.8):
    """
    Fits an ARIMA model and evaluates its performance.
    This is a basic example; for production, consider more robust cross-validation.
    """
    train_size = int(len(series) * train_size_ratio)
    train, test = series[0:train_size], series[train_size:]

    print(f"\nAttempting to fit ARIMA model with order {order}...")
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        print(model_fit.summary())

        # Forecast
        forecast_steps = len(test)
        forecast = model_fit.predict(start=len(train), end=len(train) + forecast_steps - 1)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"RMSE for ARIMA{order}: {rmse:.3f}")

        # Plot forecast vs actual
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Actual Test')
        plt.plot(test.index, forecast, label='Forecast')
        plt.title(f'ARIMA{order} Forecast vs Actual')
        plt.xlabel('Date')
        plt.ylabel(VALUE_COLUMN)
        plt.legend()
        plt.grid(True)
        plt.show()

        return model_fit, rmse
    except Exception as e:
        print(f"Error fitting ARIMA model with order {order}: {e}")
        return None, None

# --- 6. Grid Search for Optimal Parameters (Conceptual) ---
def grid_search_arima_parameters(series, p_values, d_values, q_values, train_size_ratio=0.8):
    """
    Performs a conceptual grid search for optimal ARIMA parameters (p, d, q).
    This can be computationally intensive.
    """
    best_score, best_order = float("inf"), None
    results = []

    train_size = int(len(series) * train_size_ratio)
    train, test = series[0:train_size], series[train_size:]

    print("\nStarting Grid Search for ARIMA parameters...")
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model = ARIMA(train, order=order)
                    model_fit = model.fit()
                    # Using AIC as a metric for model selection during grid search
                    aic = model_fit.aic
                    results.append({'order': order, 'aic': aic})

                    if aic < best_score:
                        best_score, best_order = aic, order
                    print(f'ARIMA{order} AIC: {aic:.3f}')
                except Exception as e:
                    print(f'ARIMA{order} failed: {e}')
                    continue
    print(f'\nBest ARIMA order: {best_order} with AIC: {best_score:.3f}')
    return best_order, results

# --- Main Execution ---
if __name__ == "__main__":
    # Load data
    time_series = load_data(DATA_PATH, DATE_COLUMN, VALUE_COLUMN)

    if time_series is not None:
        # 1. Visualize the raw time series
        visualize_series(time_series)

        # 2. Determine 'd' (differencing order)
        # Plot ACF/PACF of the original series to check for stationarity
        print("\n--- Analyzing Original Series for Stationarity ---")
        plot_acf_pacf(time_series)
        print("If ACF decays slowly, differencing is likely needed.")

        # Apply differencing and re-check stationarity
        # Start with d=1, then d=2 if needed.
        print("\n--- Analyzing Differenced Series (d=1) ---")
        differenced_series_1 = plot_differenced_series(time_series, lags=1)
        if differenced_series_1 is not None and len(differenced_series_1) > 0:
            plot_acf_pacf(differenced_series_1)
            print("After differencing, if ACF drops quickly to zero, the series is likely stationary.")

        # 3. Determine 'p' and 'q' from the (stationarized) differenced series
        # Based on the ACF/PACF plots of the differenced series, estimate p and q.
        # Example: If PACF cuts off at lag 2 and ACF tails off, p=2. If ACF cuts off at lag 1 and PACF tails off, q=1.
        print("\n--- Estimating p and q from Differenced Series ACF/PACF ---")
        # Manually inspect the plots from step 2 and update these values
        # For example, if you observe a significant spike at lag 2 in PACF and then it drops, p=2.
        # If you observe a significant spike at lag 1 in ACF and then it drops, q=1.
        suggested_p = 5 # Placeholder - update based on your PACF analysis
        suggested_d = 1 # Placeholder - update based on your differencing analysis
        suggested_q = 0 # Placeholder - update based on your ACF analysis

        print(f"Suggested ARIMA order (p,d,q) based on visual inspection: ({suggested_p}, {suggested_d}, {suggested_q})")

        # 4. Fit and evaluate an example ARIMA model with the visually suggested parameters
        # You can iterate on these parameters based on evaluation metrics and plot.
        print("\n--- Fitting and Evaluating Example ARIMA Model ---")
        example_order = (suggested_p, suggested_d, suggested_q)
        model_fit, rmse = fit_and_evaluate_arima(time_series, order=example_order)

        # 5. (Optional) Perform a grid search for a range of parameters
        # Be cautious with the range of p, d, q values as it can be very time-consuming.
        # Smaller ranges are recommended for initial exploration.
        print("\n--- Running Conceptual Grid Search (can be time-consuming) ---")
        p_values = range(0, 6)  # Example: 0 to 5
        d_values = range(0, 3)  # Example: 0 to 2
        q_values = range(0, 6)  # Example: 0 to 5

        best_arima_order, grid_search_results = grid_search_arima_parameters(time_series, p_values, d_values, q_values)

        print("\n--- Analysis Complete ---")
        print("Review the plots and model summaries to determine the optimal ARIMA parameters.")
        print("Consider AIC/BIC values from model summaries for comparison across different orders.")
