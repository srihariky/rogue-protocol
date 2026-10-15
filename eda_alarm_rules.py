import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

nominal_ranges = {
    'Lab-Power': (10.74, 13.34),
    'Life-Support': (18.96, 22.90),
    'Antenna': (31.92, 51.27),
    'Notice-Bus': (239.72, 374.64)
}

def is_out_of_bounds(row):
    mod = row['module']
    r = row['reading']
    if pd.isna(r):
        return 1
    low, high = nominal_ranges[mod]
    return int(r < low or r > high)

train['out_of_bounds'] = train.apply(is_out_of_bounds, axis=1)

print("Crosstab of (status in [ERROR, CRIT]) vs out_of_bounds vs is_true_fault:")
train['is_alarm'] = train['status'].isin(['ERROR', 'CRIT']).astype(int)

for alarm in [0, 1]:
    for oob in [0, 1]:
        subset = train[(train['is_alarm'] == alarm) & (train['out_of_bounds'] == oob)]
        print(f"Alarm={alarm}, OutOfBounds={oob} -> Count={len(subset)}, FaultRate={subset['is_true_fault'].mean():.4f}")

