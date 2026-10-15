import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('train.csv')

# Let's inspect the message parts:
# Usually: NINE: [prefix.] [event] [module reading unit]
# Let's see what parts exist
def split_msg(msg):
    # remove "NINE: "
    m = re.sub(r'^NINE:\s*', '', str(msg))
    # extract bracket
    bracket = ""
    b_match = re.search(r'\[(.*?)\]', m)
    if b_match:
        bracket = b_match.group(1)
        m = re.sub(r'\[.*?\]', '', m).strip()
    return m, bracket

parts = train['message'].apply(split_msg)
train['body'] = [p[0] for p in parts]
train['bracket'] = [p[1] for p in parts]

# Count vectorizer on body
vec = CountVectorizer(ngram_range=(1, 3), min_df=5)
X = vec.fit_transform(train['body'])
feature_names = vec.get_feature_names_out()

faults = train['is_true_fault'].values
p_fault = X[faults == 1].sum(axis=0) / faults.sum()
p_nofault = X[faults == 0].sum(axis=0) / (len(faults) - faults.sum())

diff = np.array(p_fault - p_nofault).flatten()
ratio = np.array((p_fault + 1e-4) / (p_nofault + 1e-4)).flatten()

top_fault_indices = np.argsort(diff)[::-1][:25]
top_nofault_indices = np.argsort(diff)[:25]

print("Top n-grams associated with TRUE FAULTS (highest difference):")
for idx in top_fault_indices:
    print(f"  {feature_names[idx]:<35} | p(fault)={p_fault[0, idx]:.3f} | p(nofault)={p_nofault[0, idx]:.3f} | diff={diff[idx]:.3f} | ratio={ratio[idx]:.2f}")

print("\nTop n-grams associated with LIES / NOISE (lowest difference):")
for idx in top_nofault_indices:
    print(f"  {feature_names[idx]:<35} | p(fault)={p_fault[0, idx]:.3f} | p(nofault)={p_nofault[0, idx]:.3f} | diff={diff[idx]:.3f} | ratio={ratio[idx]:.2f}")


