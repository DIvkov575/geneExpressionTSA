import pandas as pd
from darts import TimeSeries
from darts.models import NaiveMean
from darts.metrics import mae
import os
import numpy as np

def train_baseline_naive_mean():
    """
    Trains and evaluates a NaiveMean model on the CRE dataset.
    """
    # --- 1. Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'CRE.csv')
    model_dir = os.path.join(script_dir, 'darts_logs', 'naive_mean_cre_model_baseline')
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
    model = NaiveMean()

    total_mae = 0
    num_series_evaluated = 0
    
    print(f"--- Evaluating NaiveMean model ---")

    for i, train_series in enumerate(train_series_list):
        val_series = val_series_list[i]
        
        try:
            # Naive models need to be 'fit' on the series they are predicting from
            model.fit(train_series)
            prediction = model.predict(
                n=len(val_series)
            )
            current_mae = mae(val_series, prediction)
            
            if np.isnan(current_mae) or np.isinf(current_mae):
                print(f"MAE is NaN or Inf for series {i+1}. Skipping this series.")
                continue

            total_mae += current_mae
            num_series_evaluated += 1
        except Exception as e:
            print(f"Evaluation failed for series {i+1} with error: {e}. Skipping this series.")
            continue

    if num_series_evaluated > 0:
        average_mae = total_mae / num_series_evaluated
        print(f"Average MAE across {num_series_evaluated} series: {average_mae:.4f}")
    else:
        average_mae = "N/A"
        print("Evaluation could not be completed for any series.")
    print("\n")

    # --- 3. Save Results ---
    results_summary = {
        "model_type": "NaiveMean",
        "average_mae": average_mae,
        "num_series_evaluated": num_series_evaluated
    }
    summary_path = os.path.join(model_dir, "results_summary.txt")
    with open(summary_path, 'w') as f:
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Results summary saved to {summary_path}")
    print("\n--- Baseline NaiveMean training complete ---")


if __name__ == '__main__':
    train_baseline_naive_mean()
