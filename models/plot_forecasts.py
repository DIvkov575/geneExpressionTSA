
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
import matplotlib.pyplot as plt
import os

def plot_nbeats_forecasts():
    """
    Loads the best N-BEATS model, generates forecasts for a series,
    and plots them against the true values.
    """
    # --- 1. Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'CRE.csv')
    model_path = os.path.join(script_dir, 'darts_logs', 'nbeats_cre_baseline', '_model.pth.tar')
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # --- 2. Load Data ---
    print("--- Loading data ---")
    df = pd.read_csv(data_path)
    df = df.astype('float32')

    # Select the first series (column '1') for plotting
    series_to_plot = TimeSeries.from_series(df['1'])
    
    # Split into training and validation sets
    split_fraction = 0.8
    train_series, val_series = series_to_plot.split_before(split_fraction)
    print(f"Series '1' loaded. Train length: {len(train_series)}, Val length: {len(val_series)}")
    print("\n")

    # --- 3. Load Model ---
    print(f"--- Loading N-BEATS model from {model_path} ---")
    model = NBEATSModel.load(model_path)
    print("Model loaded successfully.")
    print("\n")

    # --- 4. Generate Forecasts ---
    print("-- - Generating forecasts ---")
    # Get input_chunk_length and output_chunk_length from the loaded model
    # These are stored in the model's _model_params attribute
    input_chunk_length = model.input_chunk_length
    output_chunk_length = model.output_chunk_length

    # Predict the validation series.
    # We need to provide the training series to the predict method.
    # The forecast horizon should be the length of the validation series.
    
    # For NBEATS, we can use historical_forecasts for a more robust evaluation
    # or simply predict from the end of the training series.
    # Let's predict from the end of the training series for simplicity in plotting.
    
    # The NBEATS model was trained on multiple series. When predicting a single series,
    # it expects a single series as input.
    
    # The model was trained with input_chunk_length=24, output_chunk_length=12
    # We need to predict for the length of the validation series.
    
    # Darts models predict `n` steps into the future from the end of the input series.
    # The input series for prediction should be the training series.
    
    # Ensure the training series is long enough for the input_chunk_length
    if len(train_series) < input_chunk_length:
        print(f"Warning: Training series length ({len(train_series)}) is less than model's input_chunk_length ({input_chunk_length}). Forecast might be unreliable.")
        # For plotting, we might need to pad the training series or adjust the prediction strategy.
        # For now, we'll proceed, but this is a potential issue.

    # Generate forecasts
    forecast = model.predict(n=len(val_series), series=train_series)
    print("Forecasts generated.")
    print("\n")

    # --- 5. Plot ---
    print("--- Plotting forecasts ---")
    plt.figure(figsize=(12, 6))
    series_to_plot.plot(label='True Values (Full Series)')
    train_series.plot(label='Training Data')
    val_series.plot(label='True Values (Validation)')
    forecast.plot(label='N-BEATS Forecast')
    plt.title(f'N-BEATS Forecast vs. True Values for Series 1 (MAPE: {model.last_epoch_val_loss:.2f}%)') # Assuming last_epoch_val_loss is MAPE
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plot_path = os.path.join(plot_dir, 'nbeats_forecast_vs_true.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Forecast plot saved to '{plot_path}'")
    print("\n--- Plotting complete ---")


if __name__ == '__main__':
    plot_nbeats_forecasts()
