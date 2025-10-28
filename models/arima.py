import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA
from sklearn.metrics import mean_squared_error

from train_model import create_sliding_windows, train_val_test_split

def train_and_predict_arima(X_train, y_train, X_test, y_test):
    train_series = TimeSeries.from_values(y_train)
    test_series = TimeSeries.from_values(y_test)

    model = ARIMA(d=1)
    model.fit(train_series)

    predictions = model.predict(len(test_series))
    return test_series, predictions

if __name__ == "__main__":
    df = pd.read_csv('data/CRE.csv')
    series_cols = [col for col in df.columns if col != 'time-axis']
    data = df[series_cols].values
    X, y, series_ids = create_sliding_windows(data)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y, series_ids)

    test_arima, predictions_arima = train_and_predict_arima(X_train, y_train, X_test, y_test)
    rmse_arima = np.sqrt(mean_squared_error(test_arima.to_dataframe(), predictions_arima.to_dataframe()))

    print(f'ARIMA Average RMSE: {rmse_arima}')
