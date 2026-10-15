import pandas as pd
import numpy as np

# Load test predictions from generate_submission.py
import generate_submission

test_blend = generate_submission.test_blend
oof_blend = generate_submission.oof_blend
y_full = generate_submission.y_full_train

print(f"Total test rows: {len(test_blend)}")

print("\n--- Flagged count vs Threshold on test.csv ---")
for th in np.arange(0.25, 0.50, 0.01):
    cnt = (test_blend >= th).sum()
    pct = cnt / len(test_blend)
    # calculate OOF F1 at this threshold
    from sklearn.metrics import f1_score, precision_score, recall_score
    oof_f1 = f1_score(y_full, (oof_blend >= th).astype(int))
    oof_p = precision_score(y_full, (oof_blend >= th).astype(int))
    oof_r = recall_score(y_full, (oof_blend >= th).astype(int))
    print(f"Threshold: {th:.2f} -> Flagged on Test: {cnt:>4} ({pct:.1%}) | OOF F1: {oof_f1:.4f} | Prec: {oof_p:.4f} | Rec: {oof_r:.4f}")

# Find threshold that gives closest to 718
target = 718
diffs = [abs((test_blend >= th).sum() - target) for th in np.arange(0.20, 0.50, 0.001)]
best_idx = np.argmin(diffs)
target_th = np.arange(0.20, 0.50, 0.001)[best_idx]
print(f"\nExact threshold for ~718 faults: {target_th:.4f} (gives {(test_blend >= target_th).sum()} faults)")

