import pandas as pd
from darts import TimeSeries
from typing import List
import torch
from darts.models import LightGBMModel
from darts.metrics import mape

def train_and_validate_lightgbm():
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

    model = LightGBMModel(
        lags=36,
        output_chunk_length=12,
        random_state=42,
    )

    model.fit(series=train_series_list, val_series=val_series_list)

    total_mape = 0
    for i, train_series in enumerate(train_series_list):
        val_series = val_series_list[i]
        prediction = model.historical_forecasts(
            val_series,
            start=val_series.start_time(),
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=True
        )
        
        current_mape = mape(val_series, prediction)
        total_mape += current_mape

    average_mape = total_mape / len(train_series_list)
    print(f"\nValidation complete.")
    print(f"Average MAPE for LightGBM across all {len(val_series_list)} validation series: {average_mape:.2f}%")

if __name__ == '__main__':
    train_and_validate_lightgbm()