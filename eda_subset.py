import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

nominal_ranges = {
    'Lab-Power': (10.74, 13.34),
    'Life-Support': (18.96, 22.90),
    'Antenna': (31.92, 51.27),
    'Notice-Bus': (239.72, 374.64)
}

def parse_bracket(msg):
    m = re.search(r'\[(.*?)\]', str(msg))
    if not m:
        return "", "", ""
    parts = m.group(1).split()
    mod = parts[0] if len(parts) > 0 else ""
    r = parts[1] if len(parts) > 1 else ""
    u = parts[2] if len(parts) > 2 else ""
    return mod, r, u

brackets = [parse_bracket(m) for m in df['message']]
df['b_mod'] = [b[0] for b in brackets]
df['mod_mismatch'] = (df['b_mod'] != df['module']).astype(int)

def get_oob(row):
    r = row['reading']
    if pd.isna(r):
        return 1
    low, high = nominal_ranges[row['module']]
    return int(r < low or r > high)

df['reading_oob'] = df.apply(get_oob, axis=1)

PREFIXES = ['crew need not worry', 'all is well', 'nothing to see', 'logging', 'noted']
def get_event(msg):
    m = re.sub(r'^NINE:\s*', '', str(msg)).strip()
    body = re.sub(r'\[.*?\]', '', m).strip()
    for p in PREFIXES:
        if body.lower().startswith(p + "."):
            return body[len(p)+1:].strip()
        elif body.lower().startswith(p):
            return body[len(p):].strip()
    return body

df['event'] = df['message'].apply(get_event)

subset = df[(df['status'].isin(['CRIT', 'ERROR'])) & (df['reading_oob'] == 1) & (df['mod_mismatch'] == 0)].copy()

print(f"Subset size: {len(subset)}, overall fault rate: {subset['is_true_fault'].mean():.4f}")

print("\nFault rate by event in this subset:")
print(subset.groupby('event')['is_true_fault'].agg(['count', 'mean']).sort_values('mean', ascending=False))

print("\nFault rate by retry_count:")
print(subset.groupby('retry_count')['is_true_fault'].agg(['count', 'mean']))

print("\nDelay_ms: faults vs non-faults:")
print("Faults delay_ms:", subset[subset['is_true_fault'] == 1]['delay_ms'].describe())
print("Non-faults delay_ms:", subset[subset['is_true_fault'] == 0]['delay_ms'].describe())

