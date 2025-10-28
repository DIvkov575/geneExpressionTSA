import pandas as pd
from darts import TimeSeries
from darts.models import NaiveMean
from darts.metrics import rmse

def train_naive_mean_model(data_path, forecast_horizon=1, test_size=0.1):
    """
    Trains a Darts NaiveMean model for time series forecasting.

    Args:
        data_path (str): Path to the CSV data file.
        forecast_horizon (int): Number of steps to forecast ahead.
        test_size (float): Fraction of data to use for testing.

    Returns:
        darts.models.NaiveMean: Trained NaiveMean model.
        dict: Dictionary containing evaluation metrics.
    """
    # 1. Load data
    df = pd.read_csv(data_path)

    # Prepare data for Darts
    # Convert 'time-axis' to datetime and set as index
    df['time'] = pd.to_datetime(df['time-axis'], unit='s')
    df = df.set_index('time')
    df = df.drop(columns=['time-axis'])

    # Calculate average frequency and create Timedelta
    time_axis_numeric = pd.to_numeric(pd.read_csv(data_path)['time-axis'])
    time_diffs = time_axis_numeric.diff().dropna()
    average_diff_seconds = time_diffs.mean()
    freq_timedelta = pd.to_timedelta(average_diff_seconds, unit='s')

    # Resample the DataFrame to a regular frequency
    df_resampled = df.resample(freq_timedelta).mean()

    all_series = []
    for col in df_resampled.columns:
        series = TimeSeries.from_dataframe(df_resampled, value_cols=[col], freq=freq_timedelta)
        all_series.append(series)

    # Manual splitting into train and test sets
    train_series_list = []
    test_series_list = []
    for series in all_series:
        split_point = int(len(series) * (1 - test_size))
        train = series.drop_after(series.time_index[split_point - 1])
        test = series.drop_before(series.time_index[split_point])
        train_series_list.append(train)
        test_series_list.append(test)

    # 5. Train NaiveMean model
    model = NaiveMean()

    # 6. Forecast and Evaluate
    total_rmse = 0
    num_series = len(all_series)
    for i in range(num_series):
        model.fit(train_series_list[i])
        forecast = model.predict(len(test_series_list[i]))
        current_rmse = rmse(test_series_list[i], forecast)
        total_rmse += current_rmse

    average_rmse = total_rmse / num_series

    metrics = {"average_rmse": average_rmse}
    print(f"Average Test RMSE: {average_rmse}")

    return model, metrics

if __name__ == "__main__":
    data_file = "/Users/dmitriyivkov/programming/protien-tsa/data/CRE.csv"
    trained_model, evaluation_metrics = train_naive_mean_model(data_file, forecast_horizon=1)
    print("NaiveMean model training and evaluation complete.")
    print(f"Evaluation Metrics: {evaluation_metrics}")
