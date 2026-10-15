import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

FEATURES = ['reading', 'delay_ms', 'retry_count']

model = make_pipeline(
    make_column_transformer((make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), FEATURES)),
    LogisticRegression(),
)

model.fit(train, train['is_true_fault'])
val_prob = model.predict_proba(val)[:, 1]

for thresh in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    verdict = (val_prob >= thresh).astype(int)
    f1 = f1_score(val['is_true_fault'], verdict)
    p = precision_score(val['is_true_fault'], verdict, zero_division=0)
    r = recall_score(val['is_true_fault'], verdict)
    print(f"Threshold: {thresh:.2f} | F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")

auc = roc_auc_score(val['is_true_fault'], val_prob)
print(f"ROC-AUC: {auc:.4f}")

