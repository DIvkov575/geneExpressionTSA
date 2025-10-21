
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
try:
    data = pd.read_csv('../data/CRE.csv')
except FileNotFoundError:
    data = pd.read_csv('data/CRE.csv')

# Set the 'time-axis' as the index
data = data.set_index('time-axis')

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]

# Fit the VAR model
model = VAR(train)
try:
    results = model.fit(maxlags=10, ic='aic')
except ValueError as e:
    print(f"Error fitting VAR model: {e}")
    # A simple fallback if lag selection fails
    results = model.fit(1) 

# Make predictions
lag_order = results.k_ar
forecast_input = train.values[-lag_order:]
predictions = results.forecast(y=forecast_input, steps=len(test))

# Evaluate the model
mse = mean_squared_error(test.values, predictions)
print(f'Mean Squared Error: {mse}')

# Calculate and print MSE for each column
for i, col in enumerate(data.columns):
    mse_col = mean_squared_error(test[col], predictions[:, i])
    print(f'MSE for {col}: {mse_col}')
