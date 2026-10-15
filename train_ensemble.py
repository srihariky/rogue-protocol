import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import train_advanced

X_tr = train_advanced.X_tr
X_val = train_advanced.X_val
X_te = train_advanced.X_te
y_tr = train_advanced.y_tr
y_val = train_advanced.y_val

# LightGBM predictions from previous run
lgb_val_preds = train_advanced.val_preds

# Now let's train HistGradientBoosting
print("Training HistGradientBoostingClassifier...")
hgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.04,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42
)
hgb.fit(X_tr, y_tr)
hgb_val_preds = hgb.predict_proba(X_val)[:, 1]
print(f"HGB Val ROC-AUC: {roc_auc_score(y_val, hgb_val_preds):.4f}")

best_hgb_f1, best_hgb_th = 0, 0.5
for th in np.arange(0.15, 0.85, 0.01):
    f = f1_score(y_val, (hgb_val_preds >= th).astype(int))
    if f > best_hgb_f1:
        best_hgb_f1 = f
        best_hgb_th = th
print(f"Best HGB Val F1: {best_hgb_f1:.4f} at {best_hgb_th:.2f}")

# Train Random Forest / Extra Trees
print("\nTraining ExtraTreesClassifier...")
et = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
et.fit(X_tr.fillna(0), y_tr)
et_val_preds = et.predict_proba(X_val.fillna(0))[:, 1]
print(f"ET Val ROC-AUC: {roc_auc_score(y_val, et_val_preds):.4f}")

# Ensemble weights grid search
print("\n--- Grid search ensemble weights ---")
best_ens_f1 = 0
best_w = None
best_th = 0.5

for w_lgb in np.linspace(0.4, 0.9, 11):
    for w_hgb in np.linspace(0.1, 0.5, 9):
        w_et = max(0, 1.0 - w_lgb - w_hgb)
        if w_et < 0:
            continue
        blend = w_lgb * lgb_val_preds + w_hgb * hgb_val_preds + w_et * et_val_preds
        for th in np.arange(0.25, 0.55, 0.02):
            f = f1_score(y_val, (blend >= th).astype(int))
            if f > best_ens_f1:
                best_ens_f1 = f
                best_w = (w_lgb, w_hgb, w_et)
                best_th = th

print(f"Best Ensemble Val F1: {best_ens_f1:.4f} at Threshold {best_th:.2f}")
print(f"Weights (LGB, HGB, ET): {best_w}")

blend = best_w[0] * lgb_val_preds + best_w[1] * hgb_val_preds + best_w[2] * et_val_preds
pred_val = (blend >= best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, pred_val).ravel()
print(f"Ensemble Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision: {precision_score(y_val, pred_val):.4f}, Recall: {recall_score(y_val, pred_val):.4f}")

