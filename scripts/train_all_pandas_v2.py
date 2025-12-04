#!/usr/bin/env python3
"""
Multi-Model Fraud Detection ML Pipeline (Pandas Version)
========================================================
Trains and evaluates Decision Tree, Random Forest, Logistic Regression, XGBoost, LightGBM, and Isolation Forest
using Pandas and scikit-learn (no Spark required).

Author: AI Team
Date: November 2025
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def load_config(config_path: Optional[str] = None) -> Dict:
    # ...same as Spark version, but for local CSVs...
    default_config = {
        "data_path": "data/fraud_data.csv",  # Path to CSV file
        "target_column": "fraud_flag",
        "model_dir": "models_pandas",
        "analysis_dir": "analysis_pandas",
        "model_version": "v2"
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    return default_config

def main():
    parser = argparse.ArgumentParser(description='Multi-Model Fraud Detection Pipeline (Pandas)')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--model_version', type=str, default='v2', help='Model version name to append to saved model files')
    args = parser.parse_args()

    config = load_config(args.config)
    config['model_version'] = args.model_version

    Path(config['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['analysis_dir']).mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(config['data_path'])
    target_col = config['target_column']
    features = [col for col in df.columns if col != target_col]

    # Split data
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}
    results = {}

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=100, random_state=42)
    dt.fit(X_train, y_train)
    models['decision_tree'] = dt

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=100, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    # Logistic Regression
    lr = LogisticRegression(max_iter=100, C=1.0, solver='lbfgs', random_state=42)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='auc', random_state=42)
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model

    # LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model

    # Isolation Forest (anomaly detection)
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso.fit(X_train)
    models['isolation_forest'] = iso

    # Evaluate all models
    for name, model in models.items():
        if name == 'isolation_forest':
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)
            scores = model.decision_function(X_test)
            auc = roc_auc_score(y_test, -scores)
        else:
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                scores = y_pred
            auc = roc_auc_score(y_test, scores)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            'auc': auc,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        print(f"Model: {name}")
        print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}\n")

    # Save results
    with open(os.path.join(config['analysis_dir'], f'model_comparison_summary_{config["model_version"]}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save models
    for name, model in models.items():
        import pickle
        with open(os.path.join(config['model_dir'], f'{name}_model_{config["model_version"]}.pkl'), 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    main()
