import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

nominal_ranges = {
    'Lab-Power': (10.74, 13.34),
    'Life-Support': (18.96, 22.90),
    'Antenna': (31.92, 51.27),
    'Notice-Bus': (239.72, 374.64)
}

expected_units = {
    'Lab-Power': 'V',
    'Life-Support': '%O2',
    'Antenna': 'dBm',
    'Notice-Bus': 'msg/s'
}

mod_stats = {
    'Lab-Power': {'mean': 12.19, 'std': 1.05, 'median': 12.18},
    'Life-Support': {'mean': 21.21, 'std': 1.67, 'median': 21.11},
    'Antenna': {'mean': 43.30, 'std': 7.95, 'median': 43.13},
    'Notice-Bus': {'mean': 315.28, 'std': 56.31, 'median': 313.09}
}

severe_phrases = [
    'carrier drift', 'dish actuator stalled', 'breaker tripped', 'seal breach',
    'heartbeat missing', 'o2 partial pressure', 'phase lock lost', 'queue overflow',
    'cascade brownout', 'scrubber saturation', 'overvolt on rail'
]

benign_phrases = [
    'duplicate of earlier', 'sensor warming up', 'self-test pattern',
    'simulated drill', 'scheduled calibration', 'recovered after retry'
]

reassuring_prefixes = [
    'all is well', 'crew need not worry', 'nothing to see'
]

def build_features(df_list):
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
    
    records = []
    text_clean = []
    
    for _, row in all_df.iterrows():
        mod = row['module']
        reading = row['reading']
        status = row['status']
        delay = row['delay_ms']
        retry = row['retry_count']
        msg = str(row['message'])
        
        # Parse bracket
        b_match = re.search(r'\[(.*?)\]', msg)
        bracket = b_match.group(1) if b_match else ""
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
                
        # Parse text body
        body = re.sub(r'\[.*?\]', '', msg).strip()
        body_clean = re.sub(r'^NINE:\s*', '', body).strip()
        body_lower = body_clean.lower()
        text_clean.append(body_clean)
        
        # Mismatches
        mod_mismatch = int(bool(b_mod) and b_mod != mod)
        exp_unit = expected_units.get(mod, '')
        unit_mismatch = int(bool(b_unit) and b_unit != exp_unit and not b_is_redacted)
        
        read_is_na = int(pd.isna(reading))
        read_diff = abs(reading - b_read_val) if not pd.isna(reading) and not pd.isna(b_read_val) else (0.0 if not read_is_na and not b_is_redacted else np.nan)
        read_mismatch = int(read_diff > 1e-4) if not pd.isna(read_diff) else 0
        read_diff_val = 0.0 if pd.isna(read_diff) else min(read_diff, 100.0)
        
        # Out of bounds check for row reading
        low, high = nominal_ranges[mod]
        if pd.isna(reading):
            reading_oob = 1
            reading_low_viol = 0.0
            reading_high_viol = 0.0
            reading_viol_mag = 1.0
        else:
            reading_oob = int(reading < low or reading > high)
            reading_low_viol = max(0.0, low - reading)
            reading_high_viol = max(0.0, reading - high)
            reading_viol_mag = reading_low_viol + reading_high_viol
            
        # Out of bounds check for bracket reading
        bracket_mod = b_mod if b_mod in nominal_ranges else mod
        b_low, b_high = nominal_ranges[bracket_mod]
        if pd.isna(b_read_val):
            b_read_oob = 1 if b_is_redacted else 0
            b_viol_mag = 1.0 if b_is_redacted else 0.0
        else:
            b_read_oob = int(b_read_val < b_low or b_read_val > b_high)
            b_viol_mag = max(0.0, b_low - b_read_val) + max(0.0, b_read_val - b_high)
            
        # Z-scores
        stat = mod_stats[mod]
        reading_val = stat['median'] if pd.isna(reading) else reading
        reading_z = (reading_val - stat['mean']) / stat['std']
        reading_z_abs = abs(reading_z)
        
        # Phrase flags
        has_reassurance = int(any(p in body_lower for p in reassuring_prefixes))
        has_severe = int(any(p in body_lower for p in severe_phrases))
        has_benign = int(any(p in body_lower for p in benign_phrases))
        has_logging = int('logging' in body_lower)
        has_noted = int('noted' in body_lower)
        has_status_changed = int('status changed' in body_lower)
        has_threshold_check = int('threshold check fired' in body_lower)
        has_anomaly_flag = int('anomaly flag raised' in body_lower)
        has_periodic = int('periodic report' in body_lower)
        has_nominal_window = int('reading outside nominal window' in body_lower)

        # Deception / Inversion patterns:
        is_alarm = int(status in ['ERROR', 'CRIT'])
        is_calm = int(status in ['OK', 'WARN'])
        
        # False alarm: NINE raises alarm on normal system
        suspicious_false_alarm = int(is_alarm and (reading_oob == 0) and (mod_mismatch == 0) and (read_mismatch == 0) and (has_severe == 0))
        
        # Hidden fault: NINE hides fault under calm status
        hidden_fault_mismatch = int(is_calm and mod_mismatch)
        hidden_fault_severe = int(is_calm and has_severe)
        hidden_fault_oob = int(is_calm and reading_oob and has_reassurance)
        
        # Confirmed true fault signature: Alarm + OutOfBounds/Mismatch
        confirmed_alarm_fault = int(is_alarm and (reading_oob or mod_mismatch or read_mismatch or has_severe))
        
        # Deceptive reassurance: says "all is well" / "nothing to see" while something is broken
        deceptive_reassurance = int(has_reassurance and (reading_oob or mod_mismatch or read_mismatch or has_severe))

        records.append({
            'mod_mismatch': mod_mismatch,
            'unit_mismatch': unit_mismatch,
            'read_is_na': read_is_na,
            'b_is_redacted': b_is_redacted,
            'read_mismatch': read_mismatch,
            'read_diff_val': read_diff_val,
            'reading_oob': reading_oob,
            'reading_low_viol': reading_low_viol,
            'reading_high_viol': reading_high_viol,
            'reading_viol_mag': reading_viol_mag,
            'b_read_oob': b_read_oob,
            'b_viol_mag': b_viol_mag,
            'reading_z': reading_z,
            'reading_z_abs': reading_z_abs,
            'has_reassurance': has_reassurance,
            'has_severe': has_severe,
            'has_benign': has_benign,
            'has_logging': has_logging,
            'has_noted': has_noted,
            'has_status_changed': has_status_changed,
            'has_threshold_check': has_threshold_check,
            'has_anomaly_flag': has_anomaly_flag,
            'has_periodic': has_periodic,
            'has_nominal_window': has_nominal_window,
            'suspicious_false_alarm': suspicious_false_alarm,
            'hidden_fault_mismatch': hidden_fault_mismatch,
            'hidden_fault_severe': hidden_fault_severe,
            'hidden_fault_oob': hidden_fault_oob,
            'confirmed_alarm_fault': confirmed_alarm_fault,
            'deceptive_reassurance': deceptive_reassurance,
            'delay_ms': delay,
            'log_delay': np.log1p(delay),
            'delay_over_500': int(delay > 500),
            'delay_over_1000': int(delay > 1000),
            'retry_count': retry,
            'retry_high': int(retry >= 3),
            'retry_delay_mult': retry * np.log1p(delay),
            'time_diff_prev': row['time_diff_prev'],
            'time_diff_next': row['time_diff_next'],
            'mod_time_diff_prev': row['mod_time_diff_prev'],
            'mod_time_diff_next': row['mod_time_diff_next'],
            'node_time_diff_prev': row['node_time_diff_prev'],
            'node_time_diff_next': row['node_time_diff_next'],
        })
        
    df_feats = pd.DataFrame(records)
    
    # Categoricals
    status_map = {'OK': 0, 'WARN': 1, 'ERROR': 2, 'CRIT': 3}
    df_feats['status_num'] = all_df['status'].map(status_map).fillna(0)
    
    mode_dummies = pd.get_dummies(all_df['mode'], prefix='mode', drop_first=True)
    module_dummies = pd.get_dummies(all_df['module'], prefix='mod', drop_first=True)
    node_dummies = pd.get_dummies(all_df['node'], prefix='node', drop_first=True)
    status_dummies = pd.get_dummies(all_df['status'], prefix='stat', drop_first=True)
    
    df_all = pd.concat([df_feats, mode_dummies, module_dummies, node_dummies, status_dummies], axis=1)
    
    out_dfs = []
    start_idx = 0
    for d in df_list:
        n = len(d)
        out_dfs.append((df_all.iloc[start_idx:start_idx+n].reset_index(drop=True),
                        text_clean[start_idx:start_idx+n]))
        start_idx += n
        
    return out_dfs

(X_tr_num, tr_text), (X_val_num, val_text), (X_te_num, te_text) = build_features([train, val, test])

# TF-IDF
tfidf = TfidfVectorizer(max_features=120, ngram_range=(1, 2), min_df=3)
tfidf_tr = tfidf.fit_transform(tr_text).toarray()
tfidf_val = tfidf.transform(val_text).toarray()
tfidf_te = tfidf.transform(te_text).toarray()

tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_tr.shape[1])]
X_tr = pd.concat([X_tr_num, pd.DataFrame(tfidf_tr, columns=tfidf_cols)], axis=1)
X_val = pd.concat([X_val_num, pd.DataFrame(tfidf_val, columns=tfidf_cols)], axis=1)
X_te = pd.concat([X_te_num, pd.DataFrame(tfidf_te, columns=tfidf_cols)], axis=1)

y_tr = train['is_true_fault'].values
y_val = val['is_true_fault'].values

print(f"X_tr features: {X_tr.shape[1]}")

# 5-fold CV with LightGBM
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
val_preds = np.zeros(len(val))
test_preds = np.zeros(len(test))

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 25,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'random_state': 42,
    'verbose': -1
}

models = []
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_tr, y_tr)):
    X_f_tr, y_f_tr = X_tr.iloc[trn_idx], y_tr[trn_idx]
    X_f_va, y_f_va = X_tr.iloc[val_idx], y_tr[val_idx]
    
    dtrain = lgb.Dataset(X_f_tr, label=y_f_tr)
    dval = lgb.Dataset(X_f_va, label=y_f_va, reference=dtrain)
    
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    models.append(model)
    oof_preds[val_idx] = model.predict(X_f_va)
    val_preds += model.predict(X_val) / 5.0
    test_preds += model.predict(X_te) / 5.0

print(f"\n--- OOF CV Results (Train 6,000 logs) ---")
print(f"OOF ROC-AUC: {roc_auc_score(y_tr, oof_preds):.4f}")
best_oof_f1, best_oof_th = 0, 0.5
for th in np.arange(0.15, 0.85, 0.01):
    f = f1_score(y_tr, (oof_preds >= th).astype(int))
    if f > best_oof_f1:
        best_oof_f1 = f
        best_oof_th = th
print(f"Best OOF F1: {best_oof_f1:.4f} at threshold {best_oof_th:.2f}")

print(f"\n--- Validation Results (Val 1,000 logs) ---")
print(f"Val ROC-AUC: {roc_auc_score(y_val, val_preds):.4f}")
val_f1_at_oof_th = f1_score(y_val, (val_preds >= best_oof_th).astype(int))
print(f"Val F1 at OOF-chosen threshold ({best_oof_th:.2f}): {val_f1_at_oof_th:.4f}")

best_val_f1, best_val_th = 0, 0.5
for th in np.arange(0.15, 0.85, 0.01):
    f = f1_score(y_val, (val_preds >= th).astype(int))
    if f > best_val_f1:
        best_val_f1 = f
        best_val_th = th
print(f"Best Val F1: {best_val_f1:.4f} at threshold {best_val_th:.2f}")

pred_val = (val_preds >= best_val_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, pred_val).ravel()
print(f"Val Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision: {precision_score(y_val, pred_val):.4f}, Recall: {recall_score(y_val, pred_val):.4f}")

