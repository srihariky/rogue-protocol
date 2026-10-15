import pandas as pd
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
df = pd.concat([train, val], ignore_index=True)

EVENT_MODULE_MAP = {
    # Antenna events
    'carrier drift beyond tolerance': 'Antenna',
    'dish actuator stalled': 'Antenna',
    'phase lock lost': 'Antenna',
    # Lab-Power events
    'overvolt on rail B': 'Lab-Power',
    'cascade brownout detected': 'Lab-Power',
    'breaker tripped twice': 'Lab-Power',
    # Life-Support events
    'seal breach suspected': 'Life-Support',
    'O2 partial pressure falling': 'Life-Support',
    'scrubber saturation': 'Life-Support',
    # Notice-Bus events
    'queue overflow, packets dropped': 'Notice-Bus',
    'heartbeat missing from 3 nodes': 'Notice-Bus',
    'bus arbitration deadlock': 'Notice-Bus'
}

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
df['event_mod'] = df['event'].map(EVENT_MODULE_MAP)

# Check when event has an intrinsic module:
has_event_mod = df.dropna(subset=['event_mod']).copy()
has_event_mod['event_mod_mismatch'] = (has_event_mod['event_mod'] != has_event_mod['module']).astype(int)

print(f"Total rows with specific failure event: {len(has_event_mod)}")
print("Event module mismatch vs is_true_fault:")
print(pd.crosstab(has_event_mod['event_mod_mismatch'], has_event_mod['is_true_fault'], normalize='index'))
print(has_event_mod['event_mod_mismatch'].value_counts())

