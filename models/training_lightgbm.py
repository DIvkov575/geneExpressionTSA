import pandas as pd
from darts import TimeSeries
from typing import List
import torch
from darts.models import LightGBMModel
from darts.metrics import mae
import optuna
import os # Added os import

_best_lightgbm_model = None
_current_best_mae = float('inf')

def train_and_validate_lightgbm(trial):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DEFAULT_DATA_PATH = os.path.join(project_root, 'data', 'CRE.csv')
    df = pd.read_csv(DEFAULT_DATA_PATH)
    df = df.astype('float32')
    full_series_sequence = [TimeSeries.from_series(df[col]) for col in df.columns]

    train_series_list = []
    val_series_list = []
    split_fraction = 0.8
    for series in full_series_sequence:
        train, val = series.split_before(split_fraction)
        train_series_list.append(train)
        val_series_list.append(val)

    # 1. Define the hyperparameter search space
    lags = trial.suggest_int("lags", 24, 48)
    output_chunk_length = trial.suggest_int("output_chunk_length", 6, 24)
    n_estimators = trial.suggest_int("n_estimators", 200, 800)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # 2. Create the model with the suggested hyperparameters
    model = LightGBMModel(
        lags=lags,
        output_chunk_length=output_chunk_length,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
    )

    # 3. Train and evaluate the model
    model.fit(series=train_series_list)

    total_mae = 0
    # Using a smaller subset for faster tuning, you can use the full list for final tuning
    series_to_evaluate = train_series_list[:20] 
    for i, train_series in enumerate(series_to_evaluate):
        val_series = val_series_list[i]
        
        try:
            prediction = model.historical_forecasts(
                val_series,
                start=val_series.start_time(),
                forecast_horizon=1,
                stride=1,
                retrain=False,
                verbose=False  # Turning off verbose for cleaner tuning logs
            )
            current_mae = mae(val_series, prediction)
            total_mae += current_mae
        except Exception as e:
            print(f"Evaluation failed for trial {trial.number} with error: {e}")
            # Return a high value to penalize failing trials
            return float('inf')


    average_mae = total_mae / len(series_to_evaluate)
    
    # Store the model if it's the best so far
    global _best_lightgbm_model, _current_best_mae
    if average_mae < _current_best_mae:
        _current_best_mae = average_mae
        _best_lightgbm_model = model # Store the trained model

    # 4. Return the evaluation metric
    return average_mae

if __name__ == '__main__':
    # Create a study object and specify the direction is to minimize the metric
    study = optuna.create_study(direction="minimize")
    
    # Start the optimization
    study.optimize(train_and_validate_lightgbm, n_trials=50) # Using 50 trials for demonstration

    print("\nHyperparameter tuning complete.")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (MAE): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best model
    if _best_lightgbm_model is not None:
        model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'darts_logs', 'lightgbm_cre_model_tuned')
        os.makedirs(model_save_dir, exist_ok=True)
        best_model_path = os.path.join(model_save_dir, "best_lightgbm_model.pth.tar")
        _best_lightgbm_model.save(best_model_path)
        print(f"\nBest LightGBM model saved to {best_model_path}")
    else:
        print("\nNo best LightGBM model was saved (possibly due to all trials failing).")
