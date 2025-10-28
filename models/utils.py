import numpy as np
import pandas as pd

def create_sliding_windows(data, window=10, horizon=1):
    """
    Converts a multi-series DataFrame/array into sliding windows for global model training.
    
    Parameters:
        data: pd.DataFrame or np.ndarray, shape (time_steps, n_series)
        window: int, number of past steps as input
        horizon: int, number of steps to forecast ahead
    Returns:
        X: np.ndarray, shape (num_samples, window)
        y: np.ndarray, shape (num_samples,)
        series_ids: np.ndarray, which series the sample comes from
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_timesteps, n_series = data.shape
    X, y, series_ids = [], [], []

    for series_id in range(n_series):
        series = data[:, series_id]
        for t in range(window, n_timesteps - horizon + 1):
            X.append(series[t-window:t])
            y.append(series[t:t+horizon].squeeze())
            series_ids.append(series_id)

    X = np.array(X)
    y = np.array(y)
    series_ids = np.array(series_ids)
    return X, y, series_ids


def train_test_split(X, y, series_ids, test_ratio=0.1):
    """
    Splits sliding window dataset into train/test preserving temporal order per series.

    Parameters:
        X, y: np.ndarray of sliding windows
        series_ids: np.ndarray mapping each sample to its series
        test_ratio: fraction of time for test
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    for series_id in np.unique(series_ids):
        idx = np.where(series_ids == series_id)[0]
        n = len(idx)
        test_size = int(n * test_ratio)
        train_size = n - test_size

        train_idx = idx[:train_size]
        test_idx = idx[train_size:]

        X_train.append(X[train_idx])
        y_train.append(y[train_idx])
        X_test.append(X[test_idx])
        y_test.append(y[test_idx])

    return (np.vstack(X_train), np.vstack(y_train)), \
        (np.vstack(X_test), np.vstack(y_test))