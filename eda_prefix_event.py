import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

PREFIXES = ['crew need not worry', 'all is well', 'nothing to see', 'logging', 'noted']

def decompose_msg(msg):
    # remove "NINE: "
    m = re.sub(r'^NINE:\s*', '', str(msg)).strip()
    # bracket
    b_match = re.search(r'\[(.*?)\]', m)
    bracket = b_match.group(1) if b_match else ""
    body = re.sub(r'\[.*?\]', '', m).strip()
    
    prefix = "none"
    event = body
    for p in PREFIXES:
        if body.lower().startswith(p + "."):
            prefix = p
            event = body[len(p)+1:].strip()
            break
        elif body.lower().startswith(p):
            prefix = p
            event = body[len(p):].strip()
            break
            
    b_parts = bracket.split()
    b_mod = b_parts[0] if len(b_parts) > 0 else ""
    b_read = b_parts[1] if len(b_parts) > 1 else ""
    b_unit = b_parts[2] if len(b_parts) > 2 else ""
    
    return prefix, event, b_mod, b_read, b_unit

parsed = [decompose_msg(m) for m in df['message']]
df['prefix'] = [p[0] for p in parsed]
df['event'] = [p[1] for p in parsed]
df['b_mod'] = [p[2] for p in parsed]
df['b_read'] = [p[3] for p in parsed]
df['b_unit'] = [p[4] for p in parsed]

print("--- Event Statistics across 7,000 logs ---")
ev_stats = df.groupby('event')['is_true_fault'].agg(['count', 'mean']).sort_values('mean', ascending=False)
print(ev_stats)

print("\n--- Prefix Statistics across 7,000 logs ---")
pref_stats = df.groupby('prefix')['is_true_fault'].agg(['count', 'mean']).sort_values('mean', ascending=False)
print(pref_stats)

print("\n--- Event x Prefix fault rate matrix ---")
pivot = df.pivot_table(index='event', columns='prefix', values='is_true_fault', aggfunc='mean')
print(pivot)

