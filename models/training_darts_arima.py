
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA
from darts.metrics import mae
import os
import optuna
import numpy as np

# Global variables for data (loaded once)
_train_series_list = None
_val_series_list = None
_model_dir = None
_best_trial_params = None # To store best parameters for retraining

def load_and_prepare_data():
    """
    Loads and prepares the data, making it available globally.
    """
    global _train_series_list, _val_series_list, _model_dir

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'CRE.csv')
    _model_dir = os.path.join(script_dir, 'darts_logs', 'arima_cre_model_tuned')
    os.makedirs(_model_dir, exist_ok=True)

    print("--- Loading and preparing data ---")
    df = pd.read_csv(data_path)
    df = df.astype('float32')

    # Convert all columns (excluding 'time-axis') to TimeSeries objects
    full_series_list = [TimeSeries.from_series(df[col]) for col in df.columns if col != 'time-axis']

    # Split into training and validation sets
    _train_series_list = []
    _val_series_list = []
    split_fraction = 0.8
    for series in full_series_list:
        train, val = series.split_before(split_fraction)
        _train_series_list.append(train)
        _val_series_list.append(val)
    
    print(f"Loaded {len(full_series_list)} series.")
    print(f"Training on {len(_train_series_list)} series, validating on {len(_val_series_list)} series.")
    print("\n")

def train_and_validate_arima(trial):
    global _train_series_list, _val_series_list, _model_dir

    p_param = trial.suggest_int("p", 0, 10)
    d_param = 1
    q_param = trial.suggest_int("q", 0, 10)
    seasonal_p_param = trial.suggest_int("seasonal_p", 0, 2)
    seasonal_d_param = 0 # No seasonal differencing for now
    seasonal_q_param = trial.suggest_int("seasonal_q", 0, 2)
    seasonal_s_param = 12 # Assuming a seasonality of 12 (e.g., monthly data, or 12 steps in a cycle)

    # Darts ARIMA trains a separate model for each series.
    trained_models = []
    
    for i, train_series in enumerate(_train_series_list):
        model = ARIMA(p=p_param, d=d_param, q=q_param,
                      seasonal_order=(seasonal_p_param, seasonal_d_param, seasonal_q_param, seasonal_s_param))
        try:
            model.fit(train_series)
            trained_models.append(model)
        except Exception as e:
            # print(f"Trial {trial.number}: Failed to train ARIMA for series {i+1} with error: {e}. Returning inf.")
            return float('inf') # Penalize trials that fail to train

    total_mae = 0
    num_series_evaluated = 0

    for i, model in enumerate(trained_models):
        train_series = _train_series_list[i]
        val_series = _val_series_list[i]
        
        try:
            prediction = model.predict(
                n=len(val_series),
                series=train_series
            )
            current_mae = mae(val_series, prediction)
            # Check for NaN or inf in MAPE, which can happen with bad predictions
            if np.isnan(current_mae) or np.isinf(current_mae):
                # print(f"Trial {trial.number}: MAPE is NaN or Inf for series {i+1}. Returning inf.")
                return float('inf')
            total_mae += current_mae
            num_series_evaluated += 1
        except Exception as e:
            # print(f"Trial {trial.number}: Evaluation failed for series {i+1} with error: {e}. Returning inf.")
            return float('inf') # Penalize trials that fail to evaluate

    if num_series_evaluated > 0:
        average_mae = total_mae / num_series_evaluated
    else:
        # print(f"Trial {trial.number}: No series evaluated. Returning inf.")
        return float('inf') # No successful evaluations

    return average_mae


def retrain_and_save_best_models(best_params):
    """
    Retrains ARIMA models with the best hyperparameters found by Optuna
    and saves them.
    """
    global _train_series_list, _model_dir

    print("\n--- Retraining and saving models with best parameters ---")
    best_models_save_dir = os.path.join(_model_dir, "best_models")
    os.makedirs(best_models_save_dir, exist_ok=True)

    p_param = best_params.get("p")
    d_param = 1 # Forced differencing order
    q_param = best_params.get("q")
    seasonal_p_param = best_params.get("seasonal_p")
    seasonal_d_param = 0 # No seasonal differencing for now
    seasonal_q_param = best_params.get("seasonal_q")
    seasonal_s_param = 12 # Fixed seasonal period

    for i, train_series in enumerate(_train_series_list):
        model = ARIMA(p=p_param, d=d_param, q=q_param,
                      seasonal_order=(seasonal_p_param, seasonal_d_param, seasonal_q_param, seasonal_s_param))
        try:
            model.fit(train_series)
            model_path = os.path.join(best_models_save_dir, f"model_series_{i+1}.pth.tar")
            model.save(model_path)
            print(f"Saved model for series {i+1} to {model_path}")
        except Exception as e:
            print(f"Failed to retrain and save model for series {i+1} with error: {e}")
    print("--- Retraining and saving complete ---")


if __name__ == '__main__':
    load_and_prepare_data()

    # Create a study object and specify the direction is to minimize the metric
    study = optuna.create_study(direction="minimize")
    
    # Start the optimization
    # Using a small number of trials for demonstration. Increase for better results.
    n_trials = 50 
    print(f"--- Starting Optuna hyperparameter tuning for ARIMA ({n_trials} trials) ---")
    study.optimize(train_and_validate_arima, n_trials=n_trials)

    print("\n--- Hyperparameter tuning complete ---")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Average MAE): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters and results
    results_summary = {
        "best_average_mae": trial.value,
        "best_p_param": trial.params.get("p"),
        "best_d_param": trial.params.get("d"),
        "best_q_param": trial.params.get("q"),
        "best_seasonal_p_param": trial.params.get("seasonal_p"),
        "best_seasonal_d_param": trial.params.get("seasonal_d"),
        "best_seasonal_q_param": trial.params.get("seasonal_q"),
        "seasonal_s_param": 12, # Fixed seasonal period
        "num_trials_run": n_trials
    }
    summary_path = os.path.join(_model_dir, "tuned_results_summary.txt")
    with open(summary_path, 'w') as f:
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Tuning results summary saved to {summary_path}")

    # Retrain and save models with best parameters
    retrain_and_save_best_models(trial.params)