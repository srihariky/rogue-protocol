import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')

# Let's inspect messages
print("Sample messages:")
for i, row in train.head(30).iterrows():
    print(f"ID: {row['log_id']} | Fault: {row['is_true_fault']} | Mod: {row['module']} | Stat: {row['status']} | Read: {row['reading']} | Msg: {row['message']}")

# Let's extract bracket content: [Module Reading Unit]
def parse_bracket(msg):
    match = re.search(r'\[(.*?)\]', str(msg))
    if match:
        return match.group(1)
    return None

train['bracket'] = train['message'].apply(parse_bracket)
print("\nSample brackets:")
print(train[['module', 'reading', 'bracket', 'is_true_fault']].head(20))

# Check mismatch between module and bracket module
def parse_bracket_parts(b):
    if not b or pd.isna(b):
        return None, None, None
    parts = b.split()
    if len(parts) == 2:
        return parts[0], parts[1], None
    elif len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return parts[0], None, None

parsed = train['bracket'].apply(parse_bracket_parts)
train['msg_module'] = [p[0] if p else None for p in parsed]
train['msg_reading'] = [p[1] if p else None for p in parsed]
train['msg_unit'] = [p[2] if p else None for p in parsed]

train['module_mismatch'] = (train['module'] != train['msg_module']).astype(int)
print("\nModule mismatch vs is_true_fault:")
print(pd.crosstab(train['module_mismatch'], train['is_true_fault'], normalize='index'))
print(train['module_mismatch'].value_counts())

# Check reading difference
def get_num(val):
    try:
        return float(val)
    except:
        return np.nan

train['msg_reading_num'] = train['msg_reading'].apply(get_num)
train['reading_diff'] = np.abs(train['reading'] - train['msg_reading_num'])
train['reading_mismatch'] = (train['reading_diff'] > 1e-4).astype(int)
train['reading_is_redacted'] = train['message'].str.contains('REDACTED', na=False).astype(int)

print("\nReading mismatch vs is_true_fault:")
print(pd.crosstab(train['reading_mismatch'], train['is_true_fault'], normalize='index'))
print(train['reading_mismatch'].value_counts())

print("\nRedacted reading vs is_true_fault:")
print(pd.crosstab(train['reading_is_redacted'], train['is_true_fault'], normalize='index'))

# Check message prefixes / phrases
train['msg_prefix'] = train['message'].apply(lambda s: s.split('.')[0] if '.' in str(s) else s)
print("\nPrefix vs is_true_fault (top 15):")
print(pd.crosstab(train['msg_prefix'], train['is_true_fault'], normalize='index'))

