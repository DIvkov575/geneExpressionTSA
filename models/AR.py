import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from utils import create_sliding_windows, train_test_split

def train_xgboost_model(data_path, window=10, horizon=1, test_ratio=0.1):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['time-axis'])
    X, y, series_ids = create_sliding_windows(df, window=window, horizon=horizon)

    if horizon > 1:
        print("Warning: XGBoostRegressor is a single-output model. "
              "Only the first step of the horizon will be used as target.")
        y = y[:, 0] # Take only the first step for prediction

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, series_ids, test_ratio=test_ratio)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test.ravel(), predictions))

    metrics = {"rmse": rmse}
    print(f"Test RMSE: {rmse}")

    return model, metrics

if __name__ == "__main__":
    # Example usage
    data_file = "/Users/dmitriyivkov/programming/protien-tsa/data/CRE.csv"
    trained_model, evaluation_metrics = train_xgboost_model(data_file, window=20, horizon=1)
    print("Model training complete.")
    print(f"Evaluation Metrics: {evaluation_metrics}")

    # You can save the model here if needed
    # trained_model.save_model("xgboost_cre_model.json")
