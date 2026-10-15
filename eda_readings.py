import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

for mod in train['module'].unique():
    sub = train[train['module'] == mod]
    print(f"\n================ Module: {mod} ================")
    print("Reading summary for non-faults (0):")
    print(sub[sub['is_true_fault'] == 0]['reading'].describe())
    print("\nReading summary for true faults (1):")
    print(sub[sub['is_true_fault'] == 1]['reading'].describe())
    
    print("\nDelay_ms summary:")
    print("Non-faults delay_ms:", sub[sub['is_true_fault'] == 0]['delay_ms'].describe())
    print("Faults delay_ms:", sub[sub['is_true_fault'] == 1]['delay_ms'].describe())

    print("\nRetry_count summary:")
    print(pd.crosstab(sub['retry_count'], sub['is_true_fault'], normalize='index'))

