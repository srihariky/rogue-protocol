import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)
df['ts'] = pd.to_datetime(df['timestamp'])

print("Earliest timestamp:", df['ts'].min())
print("Latest timestamp:  ", df['ts'].max())

# Group by day
df['day'] = df['ts'].dt.date
daily = df.groupby('day')['is_true_fault'].agg(['count', 'mean'])
print("\nDaily fault rate across all 22 days:")
print(daily)

# Group by hour
df['hour'] = df['ts'].dt.hour
hourly = df.groupby('hour')['is_true_fault'].agg(['count', 'mean'])
print("\nHourly fault rate:")
print(hourly)

