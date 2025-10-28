import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

from train_model import create_sliding_windows, train_val_test_split

def train_and_predict_ar(X_train, y_train, X_test, y_test):
    """
    Trains an AR model and returns predictions.
    """
    # Train the model
    # The 'lags' parameter needs to be chosen carefully. Using a default of 10 for now.
    # AutoReg expects a 1D array for endog, so we use y_train directly
    model = AutoReg(y_train, lags=10)
    model_fit = model.fit()

    # Make predictions
    # For AR model, predictions are made sequentially. This is a simplified approach.
    # In a real scenario, you would need to feed previous predictions back into the model.
    predictions = []
    current_window = X_test[0]
    for _ in range(len(y_test)):
        pred = model_fit.predict(start=len(y_train) + len(predictions), end=len(y_train) + len(predictions))
        predictions.append(pred[0])

    return y_test, np.array(predictions)

if __name__ == "__main__":
    df = pd.read_csv('data/CRE.csv')
    series_cols = [col for col in df.columns if col != 'time-axis']
    data = df[series_cols].values
    X, y, series_ids = create_sliding_windows(data)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y, series_ids)

    test_ar, predictions_ar = train_and_predict_ar(X_train, y_train, X_test, y_test)
    rmse_ar = np.sqrt(mean_squared_error(test_ar, predictions_ar))

    print(f'AR Model Average RMSE: {rmse_ar}')
