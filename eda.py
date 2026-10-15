import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}")
print(f"Train fault rate: {train['is_true_fault'].mean():.4f}")
print(f"Val fault rate:   {val['is_true_fault'].mean():.4f}")

print("\n--- Missing values ---")
print("Train missing:")
print(train.isnull().sum()[train.isnull().sum() > 0])
print("Val missing:")
print(val.isnull().sum()[val.isnull().sum() > 0])
print("Test missing:")
print(test.isnull().sum()[test.isnull().sum() > 0])

print("\n--- Categorical columns value counts ---")
for col in ['module', 'mode', 'node', 'status']:
    print(f"\nCol: {col}")
    ct = pd.crosstab(train[col], train['is_true_fault'], normalize='index')
    ct['count'] = train[col].value_counts()
    print(ct)

