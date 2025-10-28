import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

from train_model import create_sliding_windows, train_val_test_split

def train_and_predict_es(X_train, y_train, X_test, y_test):
    """
    Trains an Exponential Smoothing model and returns predictions.
    """
    # Create TimeSeries objects from the numpy arrays
    train_series = TimeSeries.from_values(y_train)
    test_series = TimeSeries.from_values(y_test)

    # Train the model
    model = ExponentialSmoothing()
    model.fit(train_series)

    # Make predictions
    predictions = model.predict(len(test_series))

    return test_series, predictions

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/CRE.csv')

    # Get the series columns
    series_cols = [col for col in df.columns if col != 'time-axis']

    # Prepare data for sliding windows
    data = df[series_cols].values

    # Create sliding windows
    X, y, series_ids = create_sliding_windows(data)

    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y, series_ids)

    # Train and predict using Exponential Smoothing model
    test_es, predictions_es = train_and_predict_es(X_train, y_train, X_test, y_test)
    rmse_es = np.sqrt(mean_squared_error(test_es.to_dataframe(), predictions_es.to_dataframe()))

    print(f'ExponentialSmoothing Average RMSE: {rmse_es}')
