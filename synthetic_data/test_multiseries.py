import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '..')
from ARIMA_model import MultiSeriesARIMA

df = pd.read_csv('synthetic.csv')
data = df['value'].values

print(f"Data length: {len(data)}")
print(f"Data type: {type(data)}")

# Test with p=1, d=1, q=1
model = MultiSeriesARIMA(p=1, d=1, q=1)
print(f"\nModel requires: d + max(p,q) = {model.d} + max({model.p},{model.q}) = {model.d + max(model.p, model.q)}")
print(f"Series length: {len(data)}")
print(f"Check: {len(data)} > {model.d + max(model.p, model.q)} ? {len(data) > model.d + max(model.p, model.q)}")

try:
    model.fit([data])
    print("\n✓ Fit succeeded!")
except Exception as e:
    print(f"\n✗ Fit failed: {e}")
