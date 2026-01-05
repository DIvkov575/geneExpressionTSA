import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NHiTSModel
import warnings
import os

warnings.filterwarnings("ignore")

def load_data(file_path, window_size=25):
    """Load data and create sliding windows."""
    df = pd.read_csv(file_path)
    series_cols = [col for col in df.columns if col != 'time-axis']
    all_windows = []
    
    for col in series_cols:
        series = df[col].values.astype(np.float32)
        if len(series) < window_size:
            continue
        for i in range(len(series) - window_size + 1):
            all_windows.append(series[i : i + window_size])
            
    return np.array(all_windows)

if __name__ == "__main__":
    # Configuration
    FILE_PATH = 'data/CRE.csv'
    WINDOW_SIZE = 25
    HORIZONS = [1, 2, 3, 5, 7, 10]
    MAX_HORIZON = max(HORIZONS)
    INPUT_CHUNK_LENGTH = WINDOW_SIZE - MAX_HORIZON
    OUTPUT_CHUNK_LENGTH = MAX_HORIZON
    
    print("="*70)
    print("  N-HITS MODEL TRAINING AND FORECASTING (using Darts)")
    print("="*70)

    windows = load_data(FILE_PATH, WINDOW_SIZE)
    
    TRAIN_SIZE = int(0.8 * len(windows))
    
    np.random.seed(42)
    np.random.shuffle(windows)
    
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:]

    # Downsample for speed
    MAX_TRAIN = 2000
    if len(train_windows) > MAX_TRAIN:
        print(f"Downsampling training data from {len(train_windows)} to {MAX_TRAIN} windows...")
        train_windows = train_windows[:MAX_TRAIN]

    print(f"Train windows: {len(train_windows)}")
    print(f"Test windows: {len(test_windows)}")

    # Prepare training data for Darts
    train_series = [TimeSeries.from_values(window) for window in train_windows]

    # Define the N-HiTS model
    model = NHiTSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        n_epochs=50,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "auto"}
    )
    
    print("\nTraining N-HiTS model...")
    model.fit(train_series, verbose=False)
    
    print("\nGenerating forecasts on the test set...")
    results = []
    
    for h in HORIZONS:
        all_actuals = []
        all_predictions = []
        
        for window in test_windows:
            history_size = len(window) - h
            if history_size < INPUT_CHUNK_LENGTH:
                continue

            initial_history = window[:history_size].astype(np.float32)
            actual_future = window[history_size:].astype(np.float32)

            # Create TimeSeries for prediction
            history_series = TimeSeries.from_values(initial_history)
            
            # Generate forecast
            try:
                forecast = model.predict(n=h, series=history_series)
                pred = forecast.values().flatten()
            except Exception as e:
                print(f"Warning: Forecasting failed for a window. Error: {e}")
                pred = np.full(h, np.nan)

            all_actuals.extend(actual_future)
            all_predictions.extend(pred)

        results.append({
            'horizon': h,
            'actuals': all_actuals,
            'predictions': all_predictions
        })

    # Save results to a file
    output_data = []
    for res in results:
        h = res['horizon']
        min_len = min(len(res['actuals']), len(res['predictions']))
        for i in range(min_len):
            output_data.append({'horizon': h, 'actual': res['actuals'][i], 'prediction': res['predictions'][i]})
            
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'nhits_results.csv')
    pd.DataFrame(output_data).to_csv(output_path, index=False)
    
    print(f"\n✓ N-HiTS forecasts saved to '{output_path}'")
    print("\nTraining and forecasting complete!")