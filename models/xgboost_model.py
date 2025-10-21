
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the data
try:
    data = pd.read_csv('../data/CRE.csv')
except FileNotFoundError:
    data = pd.read_csv('data/CRE.csv')

data = data.set_index('time-axis')

# Create lagged features
LAG = 5
for lag in range(1, LAG + 1):
    for col in data.columns:
        data[f'{col}_lag_{lag}'] = data[col].shift(lag)

# Drop rows with NaN values created by lagging
data = data.dropna()

# Split data into X and y
y = data[data.columns[:40]]
X = data[data.columns[40:]]

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train an XGBoost model for each target variable
predictions = pd.DataFrame()
for i, col in enumerate(y.columns):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train[col])
    preds = model.predict(X_test)
    predictions[col] = preds

# Evaluate the model
mse = mean_squared_error(y_test.values, predictions.values)
print(f'Mean Squared Error: {mse}')

# Calculate and print MSE for each column
for i, col in enumerate(y.columns):
    mse_col = mean_squared_error(y_test[col], predictions[col])
    print(f'MSE for {col}: {mse_col}')
