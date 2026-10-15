import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

for mod in train['module'].unique():
    print(f"\n================ Module: {mod} ================")
    sub = train[train['module'] == mod]
    # Check readings by percentiles
    print("Normal (0) percentiles (1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%):")
    print(np.percentile(sub[sub['is_true_fault'] == 0]['reading'].dropna(), [1, 5, 10, 25, 50, 75, 90, 95, 99]))
    print("Fault (1) percentiles:")
    print(np.percentile(sub[sub['is_true_fault'] == 1]['reading'].dropna(), [1, 5, 10, 25, 50, 75, 90, 95, 99]))

    # Let's see the fault rate by reading bins!
    sub_clean = sub.dropna(subset=['reading']).copy()
    sub_clean['reading_bin'] = pd.qcut(sub_clean['reading'], q=10, duplicates='drop')
    ct = sub_clean.groupby('reading_bin', observed=False)['is_true_fault'].agg(['count', 'mean'])
    print("\nFault rate by deciles of reading:")
    print(ct)

