"""
Rogue Protocol — Round 1 Winning Solution Pipeline
SMLRA × Abhiyantriki 2026 · Recalibrate

Provides:
- build_features(): Advanced feature engineering (NINE deception indicators, physical sensor bounds,
  bracket discrepancies, text n-grams / TF-IDF, sequence & temporal dynamics).
- get_predictions(): Trains the calibrated ensemble on train (or train+val) and generates
  probabilities for val and test.
"""

import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

NOMINAL_RANGES = {
    'Lab-Power': (10.74, 13.34),
    'Life-Support': (18.96, 22.90),
    'Antenna': (31.92, 51.27),
    'Notice-Bus': (239.72, 374.64)
}

EXPECTED_UNITS = {
    'Lab-Power': 'V',
    'Life-Support': '%O2',
    'Antenna': 'dBm',
    'Notice-Bus': 'msg/s'
}

MOD_STATS = {
    'Lab-Power': {'mean': 12.19, 'std': 1.05, 'median': 12.18},
    'Life-Support': {'mean': 21.21, 'std': 1.67, 'median': 21.11},
    'Antenna': {'mean': 43.30, 'std': 7.95, 'median': 43.13},
    'Notice-Bus': {'mean': 315.28, 'std': 56.31, 'median': 313.09}
}

SEVERE_PHRASES = [
    'carrier drift', 'dish actuator stalled', 'breaker tripped', 'seal breach',
    'heartbeat missing', 'o2 partial pressure', 'phase lock lost', 'queue overflow',
    'cascade brownout', 'scrubber saturation', 'overvolt on rail'
]

BENIGN_PHRASES = [
    'duplicate of earlier', 'sensor warming up', 'self-test pattern',
    'simulated drill', 'scheduled calibration', 'recovered after retry'
]

REASSURING_PREFIXES = [
    'all is well', 'crew need not worry', 'nothing to see'
]

def extract_features(df_list):
    all_df = pd.concat([d.assign(_orig_idx=d.index, _split=i) for i, d in enumerate(df_list)], ignore_index=True)
    all_df['ts'] = pd.to_datetime(all_df['timestamp'])
    all_df = all_df.sort_values('ts').reset_index(drop=True)
    
    # Sequence & temporal diffs
    all_df['time_diff_prev'] = all_df['ts'].diff().dt.total_seconds().fillna(9999)
    all_df['time_diff_next'] = (-all_df['ts'].diff(-1).dt.total_seconds()).fillna(9999)
    all_df['mod_time_diff_prev'] = all_df.groupby('module')['ts'].diff().dt.total_seconds().fillna(9999)
    all_df['mod_time_diff_next'] = (-all_df.groupby('module')['ts'].diff(-1).dt.total_seconds()).fillna(9999)
    all_df['node_time_diff_prev'] = all_df.groupby('node')['ts'].diff().dt.total_seconds().fillna(9999)
    all_df['node_time_diff_next'] = (-all_df.groupby('node')['ts'].diff(-1).dt.total_seconds()).fillna(9999)
    
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
                
        body = re.sub(r'\[.*?\]', '', msg).strip()
        body_clean = re.sub(r'^NINE:\s*', '', body).strip()
        body_lower = body_clean.lower()
        text_clean.append(body_clean)
        
        mod_mismatch = int(bool(b_mod) and b_mod != mod)
        exp_unit = EXPECTED_UNITS.get(mod, '')
        unit_mismatch = int(bool(b_unit) and b_unit != exp_unit and not b_is_redacted)
        
        read_is_na = int(pd.isna(reading))
        read_diff = abs(reading - b_read_val) if not pd.isna(reading) and not pd.isna(b_read_val) else (0.0 if not read_is_na and not b_is_redacted else np.nan)
        read_mismatch = int(read_diff > 1e-4) if not pd.isna(read_diff) else 0
        read_diff_val = 0.0 if pd.isna(read_diff) else min(read_diff, 100.0)
        
        low, high = NOMINAL_RANGES[mod]
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
            
        bracket_mod = b_mod if b_mod in NOMINAL_RANGES else mod
        b_low, b_high = NOMINAL_RANGES[bracket_mod]
        if pd.isna(b_read_val):
            b_read_oob = 1 if b_is_redacted else 0
            b_viol_mag = 1.0 if b_is_redacted else 0.0
        else:
            b_read_oob = int(b_read_val < b_low or b_read_val > b_high)
            b_viol_mag = max(0.0, b_low - b_read_val) + max(0.0, b_read_val - b_high)
            
        stat = MOD_STATS[mod]
        reading_val = stat['median'] if pd.isna(reading) else reading
        reading_z = (reading_val - stat['mean']) / stat['std']
        reading_z_abs = abs(reading_z)
        
        has_reassurance = int(any(p in body_lower for p in REASSURING_PREFIXES))
        has_severe = int(any(p in body_lower for p in SEVERE_PHRASES))
        has_benign = int(any(p in body_lower for p in BENIGN_PHRASES))
        has_logging = int('logging' in body_lower)
        has_noted = int('noted' in body_lower)
        has_status_changed = int('status changed' in body_lower)
        has_threshold_check = int('threshold check fired' in body_lower)
        has_anomaly_flag = int('anomaly flag raised' in body_lower)
        has_periodic = int('periodic report' in body_lower)
        has_nominal_window = int('reading outside nominal window' in body_lower)

        is_alarm = int(status in ['ERROR', 'CRIT'])
        is_calm = int(status in ['OK', 'WARN'])
        
        suspicious_false_alarm = int(is_alarm and (reading_oob == 0) and (mod_mismatch == 0) and (read_mismatch == 0) and (has_severe == 0))
        hidden_fault_mismatch = int(is_calm and mod_mismatch)
        hidden_fault_severe = int(is_calm and has_severe)
        hidden_fault_oob = int(is_calm and reading_oob and has_reassurance)
        confirmed_alarm_fault = int(is_alarm and (reading_oob or mod_mismatch or read_mismatch or has_severe))
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

class RogueRecalibrator:
    """End-to-end Recalibration model trained on Aurora ship logs."""
    def __init__(self):
        self.models_lgb = []
        self.models_hgb = []
        self.models_et = []
        self.tfidf = None
        self.tfidf_cols = []
        self.threshold = 0.39

    def fit_and_predict(self, train_df, val_df, test_df):
        (X_tr_num, tr_text), (X_va_num, va_text), (X_te_num, te_text) = extract_features([train_df, val_df, test_df])
        
        # TF-IDF
        self.tfidf = TfidfVectorizer(max_features=120, ngram_range=(1, 2), min_df=3)
        tfidf_tr = self.tfidf.fit_transform(tr_text).toarray()
        tfidf_va = self.tfidf.transform(va_text).toarray()
        tfidf_te = self.tfidf.transform(te_text).toarray()
        
        self.tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_tr.shape[1])]
        X_tr = pd.concat([X_tr_num, pd.DataFrame(tfidf_tr, columns=self.tfidf_cols)], axis=1)
        X_va = pd.concat([X_va_num, pd.DataFrame(tfidf_va, columns=self.tfidf_cols)], axis=1)
        X_te = pd.concat([X_te_num, pd.DataFrame(tfidf_te, columns=self.tfidf_cols)], axis=1)
        
        y_tr = train_df['is_true_fault'].values
        
        # 5-fold training
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_lgb = np.zeros(len(val_df))
        val_hgb = np.zeros(len(val_df))
        val_et = np.zeros(len(val_df))
        
        test_lgb = np.zeros(len(test_df))
        test_hgb = np.zeros(len(test_df))
        test_et = np.zeros(len(test_df))
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.04,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 25,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1
        }
        
        for fold, (t_idx, v_idx) in enumerate(kf.split(X_tr, y_tr)):
            X_f_tr, y_f_tr = X_tr.iloc[t_idx], y_tr[t_idx]
            X_f_va, y_f_va = X_tr.iloc[v_idx], y_tr[v_idx]
            
            # LightGBM
            dtrain = lgb.Dataset(X_f_tr, label=y_f_tr)
            dval = lgb.Dataset(X_f_va, label=y_f_va, reference=dtrain)
            m_lgb = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dtrain, dval],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            self.models_lgb.append(m_lgb)
            val_lgb += m_lgb.predict(X_va) / 5.0
            test_lgb += m_lgb.predict(X_te) / 5.0
            
            # HistGradientBoosting
            m_hgb = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=42 + fold
            )
            m_hgb.fit(X_f_tr, y_f_tr)
            self.models_hgb.append(m_hgb)
            val_hgb += m_hgb.predict_proba(X_va)[:, 1] / 5.0
            test_hgb += m_hgb.predict_proba(X_te)[:, 1] / 5.0
            
            # ExtraTrees
            m_et = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                random_state=42 + fold,
                n_jobs=-1
            )
            m_et.fit(X_f_tr.fillna(0), y_f_tr)
            self.models_et.append(m_et)
            val_et += m_et.predict_proba(X_va.fillna(0))[:, 1] / 5.0
            test_et += m_et.predict_proba(X_te.fillna(0))[:, 1] / 5.0
            
        w_lgb, w_hgb, w_et = 0.40, 0.25, 0.35
        val_prob = w_lgb * val_lgb + w_hgb * val_hgb + w_et * val_et
        test_prob = w_lgb * test_lgb + w_hgb * test_hgb + w_et * test_et
        
        return val_prob, test_prob

