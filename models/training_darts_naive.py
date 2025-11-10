import pandas as pd
from darts import TimeSeries
from darts.models import NaiveSeasonal
from darts.metrics import mape
import os
import numpy as np

def train_baseline_naive():
    """
    Trains and evaluates a baseline NaiveSeasonal model on the CRE dataset.
    """
    # --- 1. Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'CRE.csv')
    model_dir = os.path.join(script_dir, 'darts_logs', 'naive_cre_model_baseline')
    os.makedirs(model_dir, exist_ok=True)

    print("--- Loading and preparing data ---")
    df = pd.read_csv(data_path)
    df = df.astype('float32')

    # Convert all columns (excluding 'time-axis') to TimeSeries objects
    full_series_list = [TimeSeries.from_series(df[col]) for col in df.columns if col != 'time-axis']

    # Split into training and validation sets
    train_series_list = []
    val_series_list = []
    split_fraction = 0.8
    for series in full_series_list:
        train, val = series.split_before(split_fraction)
        train_series_list.append(train)
        val_series_list.append(val)
    
    print(f"Loaded {len(full_series_list)} series.")
    print(f"Training on {len(train_series_list)} series, validating on {len(val_series_list)} series.")
    print("\n")

    # --- 2. Model Definition and Evaluation ---
    # NaiveSeasonal(K=1) predicts the last observed value
    # NaiveSeasonal(K=12) predicts the value from 12 periods ago (seasonal naive)
    
    # Let's evaluate both and report the better one, or just use K=12 if seasonality is expected.
    # Given ARIMA used seasonal_s_param = 12, let's stick with K=12 for seasonal naive.
    
    model = NaiveSeasonal(K=12) # Assuming a seasonality of 12

    total_mape = 0
    num_series_evaluated = 0
    
    print(f"--- Evaluating NaiveSeasonal(K={model.K}) model ---")

    for i, train_series in enumerate(train_series_list):
        val_series = val_series_list[i]
        
        try:
            # Naive models need to be 'fit' on the series they are predicting from
            model.fit(train_series)
            prediction = model.predict(
                n=len(val_series)
            )
            current_mape = mape(val_series, prediction)
            
            if np.isnan(current_mape) or np.isinf(current_mape):
                print(f"MAPE is NaN or Inf for series {i+1}. Skipping this series.")
                continue # Skip this series if MAPE is invalid

            total_mape += current_mape
            num_series_evaluated += 1
        except Exception as e:
            print(f"Evaluation failed for series {i+1} with error: {e}. Skipping this series.")
            continue # Skip this series if evaluation fails

    if num_series_evaluated > 0:
        average_mape = total_mape / num_series_evaluated
        print(f"Average MAPE across {num_series_evaluated} series: {average_mape:.2f}%")
    else:
        average_mape = "N/A"
        print("Evaluation could not be completed for any series.")
    print("\n")

    # --- 3. Save Results ---
    results_summary = {
        "model_type": "NaiveSeasonal",
        "seasonal_period_K": model.K,
        "average_mape": average_mape,
        "num_series_evaluated": num_series_evaluated
    }
    summary_path = os.path.join(model_dir, "results_summary.txt")
    with open(summary_path, 'w') as f:
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Results summary saved to {summary_path}")
    print("\n--- Baseline NaiveSeasonal training complete ---")


if __name__ == '__main__':
    train_baseline_naive()
