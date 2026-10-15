import pandas as pd
import numpy as np

# Load val set and see errors
import test_features

val = pd.read_csv('val.csv')
y_val = val['is_true_fault'].values
val_prob = test_features.val_prob
pred = (val_prob >= 0.24).astype(int)

val['prob'] = val_prob
val['pred'] = pred
val['correct'] = (val['pred'] == val['is_true_fault']).astype(int)

fp = val[(val['pred'] == 1) & (val['is_true_fault'] == 0)]
fn = val[(val['pred'] == 0) & (val['is_true_fault'] == 1)]

print(f"Total Val: {len(val)}, FP: {len(fp)}, FN: {len(fn)}")

print("\n--- SAMPLE FALSE POSITIVES (Model said FAULT, but crew said LIE/0) ---")
for i, r in fp.head(15).iterrows():
    print(f"[{r['module']}] Status: {r['status']} | Read: {r['reading']} | Delay: {r['delay_ms']} | Prob: {r['prob']:.3f} | Msg: {r['message']}")

print("\n--- SAMPLE FALSE NEGATIVES (Model said NOISE, but crew said FAULT/1) ---")
for i, r in fn.head(15).iterrows():
    print(f"[{r['module']}] Status: {r['status']} | Read: {r['reading']} | Delay: {r['delay_ms']} | Prob: {r['prob']:.3f} | Msg: {r['message']}")

