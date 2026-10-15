import pandas as pd

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

train['ts'] = pd.to_datetime(train['timestamp'])
val['ts'] = pd.to_datetime(val['timestamp'])
test['ts'] = pd.to_datetime(test['timestamp'])

print("Train time range:", train['ts'].min(), "to", train['ts'].max())
print("Val time range:  ", val['ts'].min(), "to", val['ts'].max())
print("Test time range: ", test['ts'].min(), "to", test['ts'].max())

# Are the splits random or chronological?
print("\nLog IDs:")
print("Train:", train['log_id'].iloc[0], "to", train['log_id'].iloc[-1])
print("Val:  ", val['log_id'].iloc[0], "to", val['log_id'].iloc[-1])
print("Test: ", test['log_id'].iloc[0], "to", test['log_id'].iloc[-1])

