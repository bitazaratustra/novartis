# src/model_utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb

# ============================
# PREPROCESSING
# ============================

def preprocess_data(X_train, X_test):
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()

    nan_cols = X_train.columns[X_train.isna().any()].tolist()

    for col in nan_cols:
        X_train_clean[f"{col}_missing"] = X_train_clean[col].isna().astype(int)
        X_test_clean[f"{col}_missing"] = X_test_clean[col].isna().astype(int)

    imputer = SimpleImputer(strategy='median')
    X_train_clean[nan_cols] = imputer.fit_transform(X_train_clean[nan_cols])
    X_test_clean[nan_cols] = imputer.transform(X_test_clean[nan_cols])

    X_train_clean["missing_rate"] = X_train_clean[[f"{c}_missing" for c in nan_cols]].mean(axis=1)
    X_test_clean["missing_rate"] = X_test_clean[[f"{c}_missing" for c in nan_cols]].mean(axis=1)

    return X_train_clean, X_test_clean

# ============================
# METRICAS
# ============================

def find_best_threshold(y_true, y_probs, beta=2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
    best_idx = np.nanargmax(f_beta)
    return thresholds[best_idx], f_beta[best_idx]

# ============================
# OPTUNA OBJECTIVES
# ============================

def objective_lgb(trial, X, y):
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 15.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 15.0, log=True),
        'random_state': 42,
        'early_stopping_rounds': 30,
        'verbose_eval': False
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(params, dtr, valid_sets=[dval])
        preds = model.predict(X_val)
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

def objective_xgb(trial, X, y):
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1e-0, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 20),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 50),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'lambda': trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 30
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)
