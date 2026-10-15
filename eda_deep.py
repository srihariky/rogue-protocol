import pandas as pd
import numpy as np
import re
from collections import Counter

train = pd.read_csv('train.csv')

# Look at the structure of messages:
# e.g., "NINE: <prefix>. <middle> [<bracket>]"
def analyze_msg(msg):
    # remove bracket
    s = re.sub(r'\[.*?\]', '', msg).strip()
    return s

train['msg_no_bracket'] = train['message'].apply(analyze_msg)
print("Unique message templates (top 40):")
tmpl_counts = train.groupby('msg_no_bracket')['is_true_fault'].agg(['count', 'mean']).sort_values(by='count', ascending=False)
print(tmpl_counts.head(40))

print("\nTemplates with highest fault rate (min count 10):")
print(tmpl_counts[tmpl_counts['count'] >= 10].sort_values(by='mean', ascending=False).head(30))

print("\nTemplates with lowest fault rate (min count 10):")
print(tmpl_counts[tmpl_counts['count'] >= 10].sort_values(by='mean', ascending=True).head(30))

