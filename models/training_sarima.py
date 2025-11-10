import pandas as pd
from darts import TimeSeries
from typing import List
import torch
from darts.models import ARIMA
from darts.metrics import mape

def train_and_validate_sarima(): # Renamed function
    DEFAULT_DATA_PATH = '../data/CRE.csv'
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

    total_mape = 0
    num_evaluated_series = 0 # Keep track of how many series were successfully evaluated

    for i, train_series in enumerate(train_series_list):
        val_series = val_series_list[i]

        try:
            # Instantiate and train the ARIMA model for the current series
            model = ARIMA(
                p=1, d=1, q=1,
                seasonal_order=(1, 0, 0, 12),
                random_state=42,
            )
            model.fit(series=train_series)

            prediction = model.historical_forecasts(
                val_series,
                start=val_series.start_time(),
                forecast_horizon=1,
                stride=1,
                retrain=False, # Retrain is False because we fit it once per series
                verbose=False
            )
            current_mape = mape(val_series, prediction)
            total_mape += current_mape
            num_evaluated_series += 1

        except Exception as e:
            print(f"Evaluation failed for series {i} with error: {e}")
            continue

    if num_evaluated_series > 0:
        average_mape = total_mape / num_evaluated_series
        print(f"\nValidation complete.")
        print(f"Average MAPE for SARIMA across all {num_evaluated_series} validation series: {average_mape:.2f}%")
    else:
        print("\nNo series were successfully evaluated.")

if __name__ == '__main__':
    train_and_validate_sarima()
