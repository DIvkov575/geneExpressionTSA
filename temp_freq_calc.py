import pandas as pd
df = pd.read_csv("/Users/dmitriyivkov/programming/protien-tsa/data/CRE.csv")
time_axis = pd.to_numeric(df['time-axis'])
time_diffs = time_axis.diff().dropna()
average_diff = time_diffs.mean()
print(f"Average time difference: {average_diff} seconds")