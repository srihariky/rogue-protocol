import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

def extract_features(df_list):
    # Combine to extract time/sequence features cleanly
    all_df = pd.concat([d.assign(_orig_idx=d.index, _split=i) for i, d in enumerate(df_list)], ignore_index=True)
    all_df['ts'] = pd.to_datetime(all_df['timestamp'])
    all_df = all_df.sort_values('ts').reset_index(drop=True)
    
    # Time diffs
    all_df['time_diff_prev'] = all_df['ts'].diff().dt.total_seconds().fillna(9999)
    all_df['time_diff_next'] = (-all_df['ts'].diff(-1).dt.total_seconds()).fillna(9999)
    
    all_df['mod_time_diff_prev'] = all_df.groupby('module')['ts'].diff().dt.total_seconds().fillna(9999)
    all_df['mod_time_diff_next'] = (-all_df.groupby('module')['ts'].diff(-1).dt.total_seconds()).fillna(9999)
    
    all_df['node_time_diff_prev'] = all_df.groupby('node')['ts'].diff().dt.total_seconds().fillna(9999)
    all_df['node_time_diff_next'] = (-all_df.groupby('node')['ts'].diff(-1).dt.total_seconds()).fillna(9999)

    # Sort back to original order
    all_df = all_df.sort_values(['_split', '_orig_idx']).reset_index(drop=True)
    
    feats = []
    
    # Expected units per module
    expected_units = {
        'Lab-Power': 'V',
        'Life-Support': '%O2',
        'Antenna': 'dBm',
        'Notice-Bus': 'msg/s'
    }

    # Nominal reading stats (calculated per module)
    mod_stats = {
        'Lab-Power': {'mean': 12.19, 'std': 1.05},
        'Life-Support': {'mean': 21.21, 'std': 1.67},
        'Antenna': {'mean': 43.30, 'std': 7.95},
        'Notice-Bus': {'mean': 315.28, 'std': 56.31}
    }

    def parse_msg(row):
        msg = str(row['message'])
        mod = row['module']
        reading = row['reading']
        
        # Bracket parsing
        b_match = re.search(r'\[(.*?)\]', msg)
        bracket = b_match.group(1) if b_match else ""
        body = re.sub(r'\[.*?\]', '', msg).strip()
        body_clean = re.sub(r'^NINE:\s*', '', body).strip()
        
        # Extract parts from bracket
        b_parts = bracket.split()
        b_mod = b_parts[0] if len(b_parts) > 0 else ""
        b_read_str = b_parts[1] if len(b_parts) > 1 else ""
        b_unit = b_parts[2] if len(b_parts) > 2 else ""
        
        b_is_redacted = int('REDACTED' in bracket)
        b_read_val = np.nan
        if not b_is_redacted and b_read_str:
            try:
                b_read_val = float(b_read_str)
            except:
                pass
                
        # Module mismatch
        mod_mismatch = int(b_mod != mod)
        
        # Unit mismatch
        exp_unit = expected_units.get(mod, '')
        unit_mismatch = int(b_unit != exp_unit and not b_is_redacted)
        
        # Reading diff
        read_is_na = int(pd.isna(reading))
        read_diff = abs(reading - b_read_val) if not pd.isna(reading) and not pd.isna(b_read_val) else np.nan
        read_mismatch = int(read_diff > 1e-4) if not pd.isna(read_diff) else 0
        
        # Specific phrases in body
        body_lower = body_clean.lower()
        has_all_is_well = int('all is well' in body_lower)
        has_crew_no_worry = int('crew need not worry' in body_lower)
        has_nothing_to_see = int('nothing to see' in body_lower)
        has_logging = int('logging' in body_lower)
        has_noted = int('noted' in body_lower)
        
        # Known severe failure phrases
        severe_phrases = [
            'carrier drift', 'actuator stalled', 'breaker tripped', 'seal breach',
            'heartbeat missing', 'pressure falling', 'lock lost', 'overflow',
            'brownout', 'saturation', 'overvolt'
        ]
        has_severe_fault_msg = int(any(p in body_lower for p in severe_phrases))
        
        # Known benign / noise phrases
        benign_phrases = [
            'duplicate of earlier', 'sensor warming up', 'self-test pattern',
            'simulated drill', 'scheduled calibration', 'recovered after retry'
        ]
        has_benign_msg = int(any(p in body_lower for p in benign_phrases))
        
        # Deceptive calm: NINE says calm prefix while message contains severe phrase or status is CRIT/ERROR
        is_deceptive_calm = int((has_all_is_well or has_crew_no_worry or has_nothing_to_see) and has_severe_fault_msg)
        is_false_calm = int(has_all_is_well or has_crew_no_worry or has_nothing_to_see)

        # Z-score of reading
        stat = mod_stats.get(mod, {'mean': 0, 'std': 1})
        reading_z = (reading - stat['mean']) / stat['std'] if not pd.isna(reading) else 0.0
        reading_z_abs = abs(reading_z)
        
        msg_read_z = (b_read_val - stat['mean']) / stat['std'] if not pd.isna(b_read_val) else 0.0
        msg_read_z_abs = abs(msg_read_z)

        return {
            'mod_mismatch': mod_mismatch,
            'unit_mismatch': unit_mismatch,
            'read_is_na': read_is_na,
            'b_is_redacted': b_is_redacted,
            'read_diff': 0.0 if pd.isna(read_diff) else read_diff,
            'read_mismatch': read_mismatch,
            'has_all_is_well': has_all_is_well,
            'has_crew_no_worry': has_crew_no_worry,
            'has_nothing_to_see': has_nothing_to_see,
            'has_logging': has_logging,
            'has_noted': has_noted,
            'has_severe_fault_msg': has_severe_fault_msg,
            'has_benign_msg': has_benign_msg,
            'is_deceptive_calm': is_deceptive_calm,
            'is_false_calm': is_false_calm,
            'reading_z': reading_z,
            'reading_z_abs': reading_z_abs,
            'msg_read_z': msg_read_z,
            'msg_read_z_abs': msg_read_z_abs,
            'clean_body': body_clean
        }

    parsed_rows = [parse_msg(row) for _, row in all_df.iterrows()]
    parsed_df = pd.DataFrame(parsed_rows)
    
    # Status encoding
    status_map = {'OK': 0, 'WARN': 1, 'ERROR': 2, 'CRIT': 3}
    all_df['status_num'] = all_df['status'].map(status_map).fillna(0)
    
    # Mode one-hot
    mode_dummies = pd.get_dummies(all_df['mode'], prefix='mode', drop_first=True)
    # Module one-hot
    module_dummies = pd.get_dummies(all_df['module'], prefix='mod', drop_first=True)
    # Node one-hot
    node_dummies = pd.get_dummies(all_df['node'], prefix='node', drop_first=True)
    # Status one-hot
    status_dummies = pd.get_dummies(all_df['status'], prefix='stat', drop_first=True)

    # Combine tabular features
    X_num = pd.DataFrame({
        'reading': all_df['reading'].fillna(all_df.groupby('module')['reading'].transform('median')),
        'delay_ms': all_df['delay_ms'],
        'log_delay': np.log1p(all_df['delay_ms']),
        'retry_count': all_df['retry_count'],
        'status_num': all_df['status_num'],
        'time_diff_prev': all_df['time_diff_prev'],
        'time_diff_next': all_df['time_diff_next'],
        'mod_time_diff_prev': all_df['mod_time_diff_prev'],
        'mod_time_diff_next': all_df['mod_time_diff_next'],
        'node_time_diff_prev': all_df['node_time_diff_prev'],
        'node_time_diff_next': all_df['node_time_diff_next'],
    })
    
    # Interaction: status is OK/WARN but severe fault message or false calm
    X_num['inverted_status_crit'] = ((all_df['status'] == 'CRIT') & (parsed_df['has_benign_msg'] == 1)).astype(int)
    X_num['inverted_status_ok'] = ((all_df['status'] == 'OK') & ((parsed_df['is_false_calm'] == 1) | (parsed_df['has_severe_fault_msg'] == 1))).astype(int)
    
    # Merge parsed features
    drop_cols = ['clean_body']
    X_parsed = parsed_df.drop(columns=drop_cols)
    
    X_all = pd.concat([X_num, X_parsed, mode_dummies, module_dummies, node_dummies, status_dummies], axis=1)
    
    # Return split datasets
    out_dfs = []
    start_idx = 0
    for d in df_list:
        n = len(d)
        out_dfs.append((X_all.iloc[start_idx:start_idx+n].reset_index(drop=True), 
                        parsed_df['clean_body'].iloc[start_idx:start_idx+n].reset_index(drop=True)))
        start_idx += n
        
    return out_dfs

(X_tr_num, tr_text), (X_val_num, val_text), (X_te_num, te_text) = extract_features([train, val, test])

# Add TF-IDF features
tfidf = TfidfVectorizer(max_features=150, ngram_range=(1, 2), min_df=3)
tfidf_tr = tfidf.fit_transform(tr_text).toarray()
tfidf_val = tfidf.transform(val_text).toarray()
tfidf_te = tfidf.transform(te_text).toarray()

tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_tr.shape[1])]
X_tr = pd.concat([X_tr_num, pd.DataFrame(tfidf_tr, columns=tfidf_cols)], axis=1)
X_val = pd.concat([X_val_num, pd.DataFrame(tfidf_val, columns=tfidf_cols)], axis=1)
X_te = pd.concat([X_te_num, pd.DataFrame(tfidf_te, columns=tfidf_cols)], axis=1)

print(f"X_tr shape: {X_tr.shape}, X_val shape: {X_val.shape}, X_te shape: {X_te.shape}")

# Train LightGBM
y_tr = train['is_true_fault']
y_val = val['is_true_fault']

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

clf = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

val_prob = clf.predict(X_val)
best_f1, best_th = 0, 0.5
for th in np.arange(0.1, 0.9, 0.02):
    pred = (val_prob >= th).astype(int)
    f = f1_score(y_val, pred)
    if f > best_f1:
        best_f1 = f
        best_th = th

print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")
print(f"Best Validation F1:  {best_f1:.4f} at Threshold {best_th:.2f}")

pred = (val_prob >= best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"Precision: {precision_score(y_val, pred):.4f}, Recall: {recall_score(y_val, pred):.4f}")

# Feature importances
importances = pd.Series(clf.feature_importance(importance_type='gain'), index=X_tr.columns).sort_values(ascending=False)
print("\nTop 20 Most Important Features (Gain):")
print(importances.head(20))

