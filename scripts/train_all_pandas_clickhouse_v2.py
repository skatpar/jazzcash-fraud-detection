#!/usr/bin/env python3
"""
Multi-Model Fraud Detection ML Pipeline (ClickHouse + Spark + Pandas)
====================================================================
Loads data from ClickHouse using Spark, downsamples, converts to Pandas, and trains Decision Tree, Random Forest, Logistic Regression, XGBoost, LightGBM, and Isolation Forest using scikit-learn.

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

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def load_config(config_path: Optional[str] = None) -> Dict:
    default_config = {
        "clickhouse": {
            "host": "localhost",
            "port": 9000,
            "http_port": 8123,
            "database": "public",
            "user": "default",
            "password": "DfsTeChB1"
        },
        "table_name": "stixor_fraud_features_distributed",
        "selected_features": [
            'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 
            'start_balance', 'trx_amt', 'mbar_registered_channel',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 
            'is_business_hours', 'is_unusual_hour', 'night_weekend_combo',
            'txn_txns_3d', 'txn_total_amount_3d', 'txn_avg_amount_3d',
            'txn_max_amount_3d', 'txn_min_amount_3d', 'txn_unique_recipients_3d',
            'txn_unique_channels_3d', 'txn_unique_types_3d', 'txn_is_high_activity_3d',
            'txn_multi_channel_recent', 'txn_amount_deviation_from_avg',
            'txn_night_txns_3d', 'txn_weekend_txns_3d',
            'channel_new_jc_app', 'channel_ussd', 'channel_ussd_api',
            'channel_payment_gateway', 'channel_mobile_app',
            'type_transfer_c2c', 'type_transfer_c2b', 'type_bill_payment',
            'type_mobile_load', 'user_total_txns_3d', 'user_total_amount_3d',
            'user_avg_amount_3d', 'user_max_amount_3d', 'user_unique_recipients_3d',
            'user_unique_channels_3d', 'user_total_txns_7d', 'user_avg_amount_7d',
            'user_max_amount_7d', 'user_night_txns_7d', 'user_weekend_txns_7d'
        ],
        "target_column": "fraud_flag",
        "train_start": "2025-03-01",
        "train_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2025-07-31",
        "sample_rate": 0.01,
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
    parser = argparse.ArgumentParser(description='Multi-Model Fraud Detection Pipeline (ClickHouse+Spark+Pandas)')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--model_version', type=str, default='v2', help='Model version name to append to saved model files')
    parser.add_argument('--train_start', type=str, default='2025-03-01', help='Training period start date (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2025-06-30', help='Training period end date (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default='2025-07-01', help='Test period start date (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default='2025-07-31', help='Test period end date (YYYY-MM-DD)')
    parser.add_argument('--sample_rate', type=float, default=0.01, help='Downsample rate for non-fraud')
    args = parser.parse_args()

    config = load_config(args.config)
    config['model_version'] = args.model_version
    config['train_start'] = args.train_start
    config['train_end'] = args.train_end
    config['test_start'] = args.test_start
    config['test_end'] = args.test_end
    config['sample_rate'] = args.sample_rate

    Path(config['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['analysis_dir']).mkdir(parents=True, exist_ok=True)

    # Initialize Spark
    spark = (SparkSession.builder
        .appName("clickhouse-pandas-fraud-detection")
        .master("local[*]")
        .config("spark.jars.packages", "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1")
        .getOrCreate()
    )
    # spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    # Load training data from ClickHouse
    ch = config['clickhouse']
    features = config['selected_features']
    train_query = f"""
        SELECT {', '.join(features)}
        FROM clickhouse.{ch['database']}.{config['table_name']}
        WHERE cutoff_date BETWEEN '{config['train_start']}' AND '{config['train_end']}'
            AND mbar_account_type_name = 'Customer Account'
    """
    print("Loading training data from ClickHouse...")
    train_df_spark = spark.sql(train_query)
    print(f"Loaded {train_df_spark.count():,} rows for training.")

    # Downsample non-fraud in training data
    print("Downsampling non-fraud data in training set...")
    fraud_df = train_df_spark.filter(col('fraud_flag') == 1)
    non_fraud_df = train_df_spark.filter(col('fraud_flag') == 0)
    non_fraud_sampled = non_fraud_df.sample(withReplacement=False, fraction=config['sample_rate'], seed=42)
    balanced_train_df = fraud_df.union(non_fraud_sampled)
    print(f"Balanced training sample: {balanced_train_df.count():,} rows.")

    # Convert training data to Pandas
    print("Converting Spark training DataFrame to Pandas...")
    train_pdf = balanced_train_df.toPandas()
    print(f"Pandas training DataFrame shape: {train_pdf.shape}")

    # Load test data from ClickHouse
    test_query = f"""
        SELECT {', '.join(features)}
        FROM clickhouse.{ch['database']}.{config['table_name']}
        WHERE cutoff_date BETWEEN '{config['test_start']}' AND '{config['test_end']}'
            AND mbar_account_type_name = 'Customer Account'
    """
    print("Loading test data from ClickHouse...")
    test_df_spark = spark.sql(test_query)
    print(f"Loaded {test_df_spark.count():,} rows for test.")

    # Convert test data to Pandas
    print("Converting Spark test DataFrame to Pandas...")
    test_pdf = test_df_spark.toPandas()
    print(f"Pandas test DataFrame shape: {test_pdf.shape}")

    target_col = config['target_column']
    features = [col for col in train_pdf.columns if col != target_col]

    # Use time-based split
    X_train = train_pdf[features]
    y_train = train_pdf[target_col]
    X_test = test_pdf[features]
    y_test = test_pdf[target_col]

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
