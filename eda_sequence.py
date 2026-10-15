import pandas as pd

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

all_logs = pd.concat([train.assign(split='train'), val.assign(split='val'), test.assign(split='test')])
all_logs['ts'] = pd.to_datetime(all_logs['timestamp'])
all_logs = all_logs.sort_values('ts').reset_index(drop=True)

print("Total logs in combined timeline:", len(all_logs))
print("Time diff describe (seconds):")
all_logs['time_diff'] = all_logs['ts'].diff().dt.total_seconds()
print(all_logs['time_diff'].describe())

# Check time diff per module
all_logs['mod_time_diff'] = all_logs.groupby('module')['ts'].diff().dt.total_seconds()
print("\nTime diff per module (seconds):")
print(all_logs['mod_time_diff'].describe())

# Check time diff per node
all_logs['node_time_diff'] = all_logs.groupby('node')['ts'].diff().dt.total_seconds()
print("\nTime diff per node (seconds):")
print(all_logs['node_time_diff'].describe())

