import pandas as pd
import numpy as np
import re

# Load train and val
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

# Let's inspect messages and labels where reading is completely normal vs completely out of bounds
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
df['b_read'] = [b[1] for b in brackets]
df['b_unit'] = [b[2] for b in brackets]
df['mod_mismatch'] = (df['b_mod'] != df['module']).astype(int)

# Let's check: what are ALL unique text phrases before the bracket?
def get_phrase(msg):
    # remove "NINE: "
    m = re.sub(r'^NINE:\s*', '', str(msg))
    # remove bracket
    m = re.sub(r'\[.*?\]', '', m).strip()
    return m

df['phrase'] = df['message'].apply(get_phrase)

print("Number of unique phrases:", df['phrase'].nunique())
phrase_stats = df.groupby('phrase')['is_true_fault'].agg(['count', 'mean']).sort_values('count', ascending=False)
print("\nTop 30 most frequent phrases:")
print(phrase_stats.head(30))

print("\nPhrases where fault rate is 100%:")
p_100 = phrase_stats[phrase_stats['mean'] == 1.0]
print(f"Total phrases with 100% fault rate: {len(p_100)} (covering {p_100['count'].sum()} rows)")
print(p_100.head(20))

print("\nPhrases where fault rate is 0%:")
p_0 = phrase_stats[phrase_stats['mean'] == 0.0]
print(f"Total phrases with 0% fault rate: {len(p_0)} (covering {p_0['count'].sum()} rows)")
print(p_0.head(20))

