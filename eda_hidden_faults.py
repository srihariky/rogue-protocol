import pandas as pd
import re

train = pd.read_csv('train.csv')

def parse_msg(row):
    msg = str(row['message'])
    b_match = re.search(r'\[(.*?)\]', msg)
    bracket = b_match.group(1) if b_match else ""
    b_parts = bracket.split()
    b_mod = b_parts[0] if len(b_parts) > 0 else ""
    return b_mod != row['module']

train['mod_mismatch'] = train.apply(parse_msg, axis=1)

print("Alarm=0 and mod_mismatch=1:")
sub = train[(train['status'].isin(['OK', 'WARN'])) & (train['mod_mismatch'] == 1)]
print(f"Count: {len(sub)}, Fault rate: {sub['is_true_fault'].mean():.4f}")

print("Alarm=0 and mod_mismatch=0:")
sub = train[(train['status'].isin(['OK', 'WARN'])) & (train['mod_mismatch'] == 0)]
print(f"Count: {len(sub)}, Fault rate: {sub['is_true_fault'].mean():.4f}")

