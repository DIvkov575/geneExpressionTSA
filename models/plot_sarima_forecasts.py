
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA
import matplotlib.pyplot as plt
import os

def plot_sarima_forecasts():
    """
    Loads a trained SARIMA model for a specific series, generates forecasts,
    and plots them against the true values.
    """
    # --- 1. Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'CRE.csv')
    
    # Directory where the best SARIMA models are saved
    sarima_models_dir = os.path.join(script_dir, 'darts_logs', 'arima_cre_model_tuned', 'best_models')
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
    # Construct the path to the specific model for this series
    model_path = os.path.join(sarima_models_dir, f"model_series_{series_to_plot_name}.pth.tar")
    
    print(f"--- Loading SARIMA model from {model_path} ---")
    try:
        model = ARIMA.load(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure 'training_darts_arima.py' was run successfully.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("\n")

    # --- 4. Generate Forecasts ---
    print("--- Generating forecasts ---")
    # ARIMA model's predict method needs the series it was trained on
    forecast = model.predict(n=len(val_series), series=train_series)
    print("Forecasts generated.")
    print("\n")

    # --- 5. Plot ---
    print("--- Plotting forecasts ---")
    plt.figure(figsize=(12, 6))
    series_to_plot.plot(label='True Values (Full Series)')
    train_series.plot(label='Training Data')
    val_series.plot(label='True Values (Validation)')
    forecast.plot(label='SARIMA Forecast')
    plt.title(f'SARIMA Forecast vs. True Values for Series {series_to_plot_name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plot_path = os.path.join(plot_dir, f'sarima_forecast_vs_true_series_{series_to_plot_name}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Forecast plot saved to '{plot_path}'")
    print("\n--- Plotting complete ---")


if __name__ == '__main__':
    plot_sarima_forecasts()
