
import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel
import matplotlib.pyplot as plt
import os

def plot_lightgbm_forecasts():
    """
    Loads the best LightGBM model, generates forecasts for a series,
    and plots them against the true values.
    """
    # --- 1. Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'CRE.csv')
    
    # Path to the saved LightGBM model
    lightgbm_model_path = os.path.join(script_dir, 'darts_logs', 'lightgbm_cre_model_tuned', 'best_lightgbm_model.pth.tar')
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # --- 2. Load Data ---
    print("--- Loading data ---")
    df = pd.read_csv(data_path)
    df = df.astype('float32')

    # Select the first series (column '1') for plotting
    series_to_plot_name = '1'
    series_to_plot = TimeSeries.from_series(df[series_to_plot_name])
    
    # Split into training and validation sets (consistent with training)
    split_fraction = 0.8
    train_series, val_series = series_to_plot.split_before(split_fraction)
    print(f"Series '{series_to_plot_name}' loaded. Train length: {len(train_series)}, Val length: {len(val_series)}")
    print("\n")

    # --- 3. Load Model ---
    print(f"--- Loading LightGBM model from {lightgbm_model_path} ---")
    try:
        model = LightGBMModel.load(lightgbm_model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {lightgbm_model_path}. Please ensure 'training_lightgbm.py' was run successfully.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("\n")

    # --- 4. Generate Forecasts ---
    print("--- Generating forecasts ---")
    # LightGBMModel's predict method needs the series it was trained on
    # It also needs the input_chunk_length from the model
    
    # For LightGBM, we need to provide the entire series up to the point of prediction.
    # The model was trained on multiple series, but for prediction, we provide a single series.
    
    # The LightGBM model in Darts is a GlobalForecastingModel.
    # It expects a series to predict from.
    
    # Ensure the training series is long enough for the model's lags
    # The lags parameter is equivalent to input_chunk_length for LightGBMModel
    # For LightGBMModel, model.lags can be a dict of lags for different components.
    # We need to find the maximum absolute lag to determine the effective input_chunk_length.
    if isinstance(model.lags, dict):
        # Assuming 'target' is the key for the main series lags
        lags_list = model.lags.get('target', [])
        if lags_list:
            input_chunk_length = max(abs(lag) for lag in lags_list)
        else:
            input_chunk_length = 1 # Default or handle error
    elif isinstance(model.lags, (list, int)):
        # Handle cases where lags might be a list of ints or a single int
        if isinstance(model.lags, list):
            input_chunk_length = max(abs(lag) for lag in model.lags)
        else: # int
            input_chunk_length = abs(model.lags)
    else:
        input_chunk_length = 1 # Default or handle error
    if len(train_series) < input_chunk_length:
        print(f"Warning: Training series length ({len(train_series)}) is less than model's input_chunk_length ({input_chunk_length}). Forecast might be unreliable.")
        # For plotting, we might need to pad the training series or adjust the prediction strategy.
        # For now, we'll proceed, but this is a potential issue.

    forecast = model.predict(n=len(val_series), series=train_series)
    print("Forecasts generated.")
    print("\n")

    # --- 5. Plot ---
    print("--- Plotting forecasts ---")
    plt.figure(figsize=(12, 6))
    series_to_plot.plot(label='True Values (Full Series)')
    train_series.plot(label='Training Data')
    val_series.plot(label='True Values (Validation)')
    forecast.plot(label='LightGBM Forecast')
    plt.title(f'LightGBM Forecast vs. True Values for Series {series_to_plot_name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plot_path = os.path.join(plot_dir, f'lightgbm_forecast_vs_true_series_{series_to_plot_name}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Forecast plot saved to '{plot_path}'")
    print("\n--- Plotting complete ---")


if __name__ == '__main__':
    plot_lightgbm_forecasts()
