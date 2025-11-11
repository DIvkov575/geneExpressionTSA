
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae
import torch
import os

def train_baseline_nbeats():
    """
    Trains and evaluates a baseline N-BEATS model on the CRE dataset.
    """
    # --- 1. Setup Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'CRE.csv')
    model_dir = os.path.join(script_dir, 'darts_logs', 'nbeats_cre_model_baseline')
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df.astype('float32')
    full_series_list = [TimeSeries.from_series(df[col]) for col in df.columns if col != 'time-axis']
    train_series_list = []
    val_series_list = []
    split_fraction = 0.8

    for series in full_series_list:
        train, val = series.split_before(split_fraction)
        train_series_list.append(train)
        val_series_list.append(val)
    
    input_chunk_length = 24
    output_chunk_length = 12
    n_epochs = 50

    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=n_epochs,
        random_state=42,
        model_name="nbeats_cre_baseline",
        save_checkpoints=True,
        force_reset=True,
        work_dir=os.path.dirname(model_dir),
    )

    model.fit(series=train_series_list, verbose=True)

    # evaluation
    total_mae = 0
    num_series_evaluated = 0

    for i, train_series in enumerate(train_series_list):
        val_series = val_series_list[i]
        try:
            prediction = model.historical_forecasts(
                val_series,
                start=val_series.start_time(),
                forecast_horizon=1,
                stride=1,
                retrain=False,
                verbose=False
            )
            current_mae = mae(val_series, prediction)
            total_mae += current_mae
            num_series_evaluated += 1
        except Exception as e:
            print(f"Evaluation failed for series {i} with error: {e}")

    if num_series_evaluated > 0:
        average_mae = total_mae / num_series_evaluated
        print(f"Average MAE across {num_series_evaluated} series: {average_mae:.4f}")
    else:
        print("Evaluation could not be completed for any series.")
    print("\n")

    # Save the Final Model and Results
    model_path = os.path.join(model_dir, "model.pth.tar")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    results_summary = {
        "average_mae": average_mae if num_series_evaluated > 0 else "N/A",
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "n_epochs": n_epochs,
        "num_series_evaluated": num_series_evaluated
    }
    summary_path = os.path.join(model_dir, "results_summary.txt")
    with open(summary_path, 'w') as f:
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Results summary saved to {summary_path}")
    print("\n--- Baseline training complete ---")


if __name__ == '__main__':
    train_baseline_nbeats()
