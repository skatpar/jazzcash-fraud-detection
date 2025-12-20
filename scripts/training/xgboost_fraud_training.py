"""
XGBoost Fraud Detection Model Training Script
==============================================
- Trains XGBoost model with optimal hyperparameters
- Uses class weights based on original class ratio
- Training: 2025-05-01 to 2025-06-30 (2 months)
- Testing: 2025-07-01 to 2025-07-31 (July)
- Logs experiments to MLflow with all artifacts
- Feature analysis: Correlation, Variance, VIF
- Hyperparameter tuning with RandomizedSearchCV
- SHAP analysis and feature importance
- Stores predictions with trans_id and probability scores
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

from xgboost import XGBClassifier
import shap
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_START_DATE = '2025-06-01'
TRAIN_END_DATE = '2025-06-01'
TEST_START_DATE = '2025-07-01'
TEST_END_DATE = '2025-07-01'

CLICKHOUSE_CONFIG = {
    'host': 'localhost',
    'port': 9000,
    'database': 'public',
    'user': 'default',
    'password': 'DfsTeChB1'
}

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "fraud_detection_pipeline"

# Output directories
OUTPUT_DIR = "/root/research-dir/dev/jazzcash-fraud-detection/output/xgboost_training"
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
SELECTED_COLS = [
    'trans_id',
    'cutoff_date',
    'fraud_flag',
    'trx_amt',
    'mbar_registered_channel',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'is_night',
    'user_total_txns_3d',
    'user_total_amount_3d',
    'user_avg_amount_3d',
    'user_median_amount_3d',
    'user_max_amount_3d',
    'user_min_amount_3d',
    'user_unique_recipients_3d',
    'user_unique_channels_3d',
    'user_unique_types_3d',
    'user_total_txns_7d',
    'user_total_amount_7d',
    'user_avg_amount_7d',
    'user_median_amount_7d',
    'user_max_amount_7d',
    'user_min_amount_7d',
    'user_unique_recipients_7d',
    'user_unique_channels_7d',
    'user_unique_types_7d',
    'user_most_used_channel_7d',
    'user_last_used_channel',
    'user_channel_diversity_score_7d',
    'user_most_used_type_7d',
    'user_last_used_type',
    'user_type_diversity_score_7d',
    'user_night_txns_7d',
    'user_weekend_txns_7d',
    'user_peak_hour_txns_7d',
    'user_off_peak_hour_txns_7d',
    'user_avg_start_balance_7d',
    'user_avg_end_balance_7d',
    'user_min_balance_7d',
    'user_max_balance_7d',
    'user_balance_volatility_7d',
    'user_avg_amount_per_recipient_7d',
    'user_max_amount_to_single_recipient_7d',
    'user_recipient_concentration_ratio_7d',
    'user_avg_time_between_txns_7d',
    'user_txn_frequency_score_7d',
    'user_days_since_last_txn'
]

FEATURE_NAME_MAPPING = {
    'trx_amt': 'Transaction Amount (PKR)',
    'hour_of_day': 'Hour of Transaction (0-23)',
    'day_of_week': 'Day of Week (1=Mon, 7=Sun)',
    'is_weekend': 'Weekend Transaction Flag',
    'is_night': 'Night Hours Transaction Flag',
    'user_total_txns_3d': 'User Total Transactions (3 Days)',
    'user_total_amount_3d': 'User Total Amount (3 Days)',
    'user_avg_amount_3d': 'User Average Amount (3 Days)',
    'user_median_amount_3d': 'User Median Amount (3 Days)',
    'user_max_amount_3d': 'User Maximum Amount (3 Days)',
    'user_min_amount_3d': 'User Minimum Amount (3 Days)',
    'user_unique_recipients_3d': 'User Unique Recipients (3 Days)',
    'user_unique_channels_3d': 'User Unique Channels (3 Days)',
    'user_unique_types_3d': 'User Unique Types (3 Days)',
    'user_total_txns_7d': 'User Total Transactions (7 Days)',
    'user_total_amount_7d': 'User Total Amount (7 Days)',
    'user_avg_amount_7d': 'User Average Amount (7 Days)',
    'user_median_amount_7d': 'User Median Amount (7 Days)',
    'user_max_amount_7d': 'User Maximum Amount (7 Days)',
    'user_min_amount_7d': 'User Minimum Amount (7 Days)',
    'user_unique_recipients_7d': 'User Unique Recipients (7 Days)',
    'user_unique_channels_7d': 'User Unique Channels (7 Days)',
    'user_unique_types_7d': 'User Unique Types (7 Days)',
    'user_channel_diversity_score_7d': 'User Channel Diversity Score',
    'user_type_diversity_score_7d': 'User Transaction Type Diversity',
    'user_night_txns_7d': 'User Night Transactions (7 Days)',
    'user_weekend_txns_7d': 'User Weekend Transactions (7 Days)',
    'user_peak_hour_txns_7d': 'User Peak Hour Transactions (7 Days)',
    'user_off_peak_hour_txns_7d': 'User Off-Peak Transactions (7 Days)',
    'user_avg_start_balance_7d': 'User Average Starting Balance',
    'user_avg_end_balance_7d': 'User Average Ending Balance',
    'user_min_balance_7d': 'User Minimum Balance (7 Days)',
    'user_max_balance_7d': 'User Maximum Balance (7 Days)',
    'user_balance_volatility_7d': 'User Balance Volatility',
    'user_avg_amount_per_recipient_7d': 'User Avg Amount per Recipient',
    'user_max_amount_to_single_recipient_7d': 'User Max Amount to One Recipient',
    'user_recipient_concentration_ratio_7d': 'User Recipient Concentration',
    'user_avg_time_between_txns_7d': 'User Avg Time Between Transactions',
    'user_txn_frequency_score_7d': 'User Transaction Frequency Score',
    'user_days_since_last_txn': 'Days Since User Last Transaction',
}


def create_directories():
    """Create necessary output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print(f"✓ Output directories created: {OUTPUT_DIR}")


def initialize_spark():
    """Initialize Spark session with ClickHouse configuration."""
    print("\n" + "="*60)
    print("INITIALIZING SPARK SESSION")
    print("="*60)
    
    packages = [
        "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
        "com.clickhouse:clickhouse-client:0.9.4",
        "com.clickhouse:clickhouse-http-client:0.9.4",
        "org.apache.httpcomponents.client5:httpclient5:5.2.1"
    ]
    
    spark = (SparkSession.builder
        .appName("XGBoost-Fraud-Training")
        .master("local[*]")
        .config("spark.jars.packages", ",".join(packages))
        .config("spark.executor.memory", "16g")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    
    # Configure ClickHouse catalog
    spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
    spark.conf.set("spark.sql.catalog.clickhouse.host", "localhost")
    spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
    spark.conf.set("spark.sql.catalog.clickhouse.http_port", "8123")
    spark.conf.set("spark.sql.catalog.clickhouse.user", CLICKHOUSE_CONFIG['user'])
    spark.conf.set("spark.sql.catalog.clickhouse.password", CLICKHOUSE_CONFIG['password'])
    spark.conf.set("spark.sql.catalog.clickhouse.database", CLICKHOUSE_CONFIG['database'])
    
    print("✓ Spark session initialized successfully")
    return spark


def extract_data(spark):
    """Extract training and test data from ClickHouse."""
    print("\n" + "="*60)
    print("DATA EXTRACTION")
    print("="*60)
    
    cols_str = ', '.join(SELECTED_COLS)
    
    query_train = f"""
        SELECT {cols_str}
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE cutoff_date ='{TRAIN_START_DATE}' 
        AND mbar_account_type_name = 'Customer Account'
        -- AND trx_channel = 'NEW_JC_APP'
        -- AND trx_type = 'Transfer(C2C)'
        AND trx_type='Online Payment'
        AND ac_to IS NOT NULL 
        AND ac_to <> ''
        AND start_balance <> end_balance
    """
    
    query_test = f"""
        SELECT {cols_str}
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE cutoff_date = '{TEST_START_DATE}'
        AND mbar_account_type_name = 'Customer Account'
        -- AND trx_channel = 'NEW_JC_APP'
        -- AND trx_type = 'Transfer(C2C)'
        AND trx_type='Online Payment'
        AND ac_to IS NOT NULL 
        AND ac_to <> ''
        AND start_balance <> end_balance
    """
    
    print(f"Training period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"Testing period: {TEST_START_DATE} to {TEST_END_DATE}")
    
    df_train_spark = spark.sql(query_train)
    df_test_spark = spark.sql(query_test)
    
    train_count = df_train_spark.count()
    test_count = df_test_spark.count()
    
    print(f"✓ Training data: {train_count:,} records")
    print(f"✓ Test data: {test_count:,} records")
    
    # Convert to pandas
    df_train = df_train_spark.toPandas()
    df_test = df_test_spark.toPandas()
    
    return df_train, df_test


def analyze_class_distribution(df_train, df_test):
    """Analyze and return class distribution statistics."""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    train_fraud = df_train['fraud_flag'].sum()
    train_legit = len(df_train) - train_fraud
    train_ratio = train_legit / train_fraud if train_fraud > 0 else float('inf')
    
    test_fraud = df_test['fraud_flag'].sum()
    test_legit = len(df_test) - test_fraud
    test_ratio = test_legit / test_fraud if test_fraud > 0 else float('inf')
    
    print(f"\nTraining Set:")
    print(f"  Legitimate: {train_legit:,} ({train_legit/len(df_train)*100:.2f}%)")
    print(f"  Fraud: {train_fraud:,} ({train_fraud/len(df_train)*100:.2f}%)")
    print(f"  Imbalance Ratio: {train_ratio:.2f}:1")
    
    print(f"\nTest Set:")
    print(f"  Legitimate: {test_legit:,} ({test_legit/len(df_test)*100:.2f}%)")
    print(f"  Fraud: {test_fraud:,} ({test_fraud/len(df_test)*100:.2f}%)")
    print(f"  Imbalance Ratio: {test_ratio:.2f}:1")
    
    class_stats = {
        'train_total': len(df_train),
        'train_fraud': int(train_fraud),
        'train_legit': int(train_legit),
        'train_fraud_ratio': train_fraud/len(df_train),
        'train_imbalance_ratio': train_ratio,
        'test_total': len(df_test),
        'test_fraud': int(test_fraud),
        'test_legit': int(test_legit),
        'test_fraud_ratio': test_fraud/len(df_test),
        'test_imbalance_ratio': test_ratio,
        'scale_pos_weight': train_ratio  # For XGBoost
    }
    
    return class_stats


def compute_correlation_with_target(df, features, target='fraud_flag'):
    """Compute correlation of features with target variable."""
    print("\n" + "="*60)
    print("FEATURE CORRELATION WITH TARGET")
    print("="*60)
    
    correlations = []
    for feature in features:
        if feature in df.columns:
            corr = df[feature].corr(df[target])
            correlations.append({
                'feature': feature,
                'feature_name': FEATURE_NAME_MAPPING.get(feature, feature),
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    print("\nTop 15 Features by Correlation with Fraud Flag:")
    print(corr_df.head(15).to_string(index=False))
    
    # Save correlation results
    corr_path = os.path.join(ARTIFACTS_DIR, "feature_correlation_with_target.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\n✓ Correlation results saved to: {corr_path}")
    
    # Create correlation plot
    plt.figure(figsize=(12, 10))
    top_20 = corr_df.head(20)
    colors = ['red' if c < 0 else 'green' for c in top_20['correlation']]
    plt.barh(range(len(top_20)), top_20['correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_20)), [name[:35] for name in top_20['feature_name']])
    plt.xlabel('Correlation with Fraud Flag')
    plt.title('Top 20 Features: Correlation with Target', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    corr_plot_path = os.path.join(ARTIFACTS_DIR, "correlation_with_target.png")
    plt.savefig(corr_plot_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Correlation plot saved to: {corr_plot_path}")
    
    return corr_df


def compute_feature_variance(df, features):
    """Compute variance for each feature."""
    print("\n" + "="*60)
    print("FEATURE VARIANCE ANALYSIS")
    print("="*60)
    
    variance_data = []
    for feature in features:
        if feature in df.columns:
            var = df[feature].var()
            std = df[feature].std()
            variance_data.append({
                'feature': feature,
                'feature_name': FEATURE_NAME_MAPPING.get(feature, feature),
                'variance': var,
                'std_dev': std,
                'mean': df[feature].mean(),
                'min': df[feature].min(),
                'max': df[feature].max()
            })
    
    variance_df = pd.DataFrame(variance_data)
    variance_df = variance_df.sort_values('variance', ascending=False)
    
    print("\nTop 15 Features by Variance:")
    print(variance_df[['feature', 'variance', 'std_dev']].head(15).to_string(index=False))
    
    # Save variance results
    variance_path = os.path.join(ARTIFACTS_DIR, "feature_variance.csv")
    variance_df.to_csv(variance_path, index=False)
    print(f"\n✓ Variance results saved to: {variance_path}")
    
    return variance_df


def compute_vif(X, features):
    """Compute Variance Inflation Factor for multicollinearity detection."""
    print("\n" + "="*60)
    print("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    print("="*60)
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Handle infinite/missing values
    X_clean = X.copy()
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(X_clean.median())
    
    # Add constant for VIF calculation
    X_with_const = X_clean.copy()
    X_with_const['const'] = 1
    
    vif_data = []
    for i, feature in enumerate(features):
        if feature in X_with_const.columns:
            try:
                vif = variance_inflation_factor(X_with_const.values, X_with_const.columns.get_loc(feature))
                vif_data.append({
                    'feature': feature,
                    'feature_name': FEATURE_NAME_MAPPING.get(feature, feature),
                    'VIF': vif
                })
            except Exception as e:
                vif_data.append({
                    'feature': feature,
                    'feature_name': FEATURE_NAME_MAPPING.get(feature, feature),
                    'VIF': np.nan
                })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    # Categorize VIF values
    vif_df['multicollinearity'] = vif_df['VIF'].apply(
        lambda x: 'High (>10)' if x > 10 else ('Moderate (5-10)' if x > 5 else 'Low (<5)')
    )
    
    print("\nVIF Analysis Results:")
    print(f"  High multicollinearity (VIF > 10): {len(vif_df[vif_df['VIF'] > 10])}")
    print(f"  Moderate multicollinearity (5 < VIF <= 10): {len(vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)])}")
    print(f"  Low multicollinearity (VIF <= 5): {len(vif_df[vif_df['VIF'] <= 5])}")
    
    print("\nTop 15 Features by VIF:")
    print(vif_df[['feature', 'VIF', 'multicollinearity']].head(15).to_string(index=False))
    
    # Save VIF results
    vif_path = os.path.join(ARTIFACTS_DIR, "feature_vif.csv")
    vif_df.to_csv(vif_path, index=False)
    print(f"\n✓ VIF results saved to: {vif_path}")
    
    return vif_df


def prepare_features(df_train, df_test):
    """Prepare features for training."""
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)
    
    # Define features (exclude non-feature columns)
    exclude_cols = ['trans_id', 'cutoff_date', 'fraud_flag']
    features = [c for c in df_train.columns if c not in exclude_cols]
    
    # Keep only numeric features
    numeric_features = []
    for f in features:
        if df_train[f].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(f)
    
    features = numeric_features
    print(f"Number of features: {len(features)}")
    
    # Handle missing values
    df_train_clean = df_train.copy()
    df_test_clean = df_test.copy()
    
    for col in features:
        median_val = df_train_clean[col].median()
        df_train_clean[col] = df_train_clean[col].fillna(median_val)
        df_test_clean[col] = df_test_clean[col].fillna(median_val)
        
        # Handle infinite values
        df_train_clean[col] = df_train_clean[col].replace([np.inf, -np.inf], median_val)
        df_test_clean[col] = df_test_clean[col].replace([np.inf, -np.inf], median_val)
    
    # Extract features and target
    X_train = df_train_clean[features]
    y_train = df_train_clean['fraud_flag']
    X_test = df_test_clean[features]
    y_test = df_test_clean['fraud_flag']
    
    # Store trans_id for predictions
    test_trans_ids = df_test_clean['trans_id'].values if 'trans_id' in df_test_clean.columns else np.arange(len(df_test_clean))
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Training set shape: {X_train_scaled.shape}")
    print(f"✓ Test set shape: {X_test_scaled.shape}")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, features, scaler, test_trans_ids


def run_hyperparameter_search(X_train_scaled, y_train, scale_pos_weight):
    """Run RandomizedSearchCV for hyperparameter tuning."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("="*60)
    
    # Define parameter grid
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
        'min_child_weight': randint(1, 7),
        'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
        'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 1.5)
    }
    
    # Base model
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1
    )
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    print("Running RandomizedSearchCV with 20 iterations and 3-fold CV...")
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\n✓ Best ROC-AUC Score: {random_search.best_score_:.4f}")
    print(f"\n✓ Best Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"    {param}: {value}")
    
    # Save CV results
    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results_path = os.path.join(ARTIFACTS_DIR, "hyperparameter_cv_results.csv")
    cv_results.to_csv(cv_results_path, index=False)
    print(f"\n✓ CV results saved to: {cv_results_path}")
    
    return random_search.best_params_, random_search.best_score_


def train_final_model(X_train_scaled, y_train, best_params, scale_pos_weight):
    """Train the final XGBoost model with best hyperparameters."""
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    
    model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    print("✓ Model trained successfully")
    
    return model


def get_predictions(model, X_test_scaled, test_trans_ids, y_test):
    """Get predictions and save with trans_id."""
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'trans_id': test_trans_ids,
        'actual_label': y_test.values,
        'predicted_label': y_pred,
        'fraud_probability': y_pred_proba
    })
    
    # Sort by probability (highest first)
    predictions_df = predictions_df.sort_values('fraud_probability', ascending=False)
    
    # Save predictions
    predictions_path = os.path.join(ARTIFACTS_DIR, "predictions_with_scores.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to: {predictions_path}")
    print(f"  Total predictions: {len(predictions_df):,}")
    print(f"  Predicted frauds: {(y_pred == 1).sum():,}")
    print(f"  Actual frauds: {(y_test == 1).sum():,}")
    
    return y_pred, y_pred_proba, predictions_df


def compute_feature_importance(model, features):
    """Compute and visualize feature importance."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    importance_df = pd.DataFrame({
        'feature': features,
        'feature_name': [FEATURE_NAME_MAPPING.get(f, f) for f in features],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))
    
    # Save importance
    importance_path = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"\n✓ Feature importance saved to: {importance_path}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'].values, color='forestgreen', alpha=0.7)
    plt.yticks(range(len(top_20)), [name[:40] for name in top_20['feature_name'].values])
    plt.xlabel('Feature Importance (Gain)')
    plt.title('XGBoost Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    importance_plot_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Feature importance plot saved to: {importance_plot_path}")
    
    return importance_df


def compute_shap_values(model, X_train_scaled, X_test_scaled, features, y_test):
    """Compute SHAP values for model interpretability."""
    print("\n" + "="*60)
    print("SHAP VALUE ANALYSIS")
    print("="*60)
    
    # Use a sample for SHAP (faster computation)
    shap_sample_size = min(500, len(X_test_scaled))  # Reduced for KernelExplainer
    X_shap = X_test_scaled[:shap_sample_size]
    y_shap = y_test.iloc[:shap_sample_size].values
    
    print(f"Computing SHAP values for {shap_sample_size} samples...")
    
    # Create SHAP explainer - use KernelExplainer for XGBoost compatibility
    # TreeExplainer has compatibility issues with newer XGBoost versions
    try:
        # Try TreeExplainer first (faster)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        print("✓ Using TreeExplainer (fast)")
    except (ValueError, TypeError) as e:
        print(f"TreeExplainer failed ({str(e)[:50]}...), using KernelExplainer...")
        # Fallback to KernelExplainer which works with any model
        background_sample = shap.sample(X_train_scaled, 100)
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], 
            background_sample
        )
        shap_values = explainer.shap_values(X_shap, nsamples=100)
        print("✓ Using KernelExplainer (model-agnostic)")
    
    # Save SHAP summary plot
    print("Creating SHAP summary plots...")
    
    # Bar plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=features, show=False, plot_type="bar")
    plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    shap_bar_path = os.path.join(ARTIFACTS_DIR, "shap_importance_bar.png")
    plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ SHAP bar plot saved to: {shap_bar_path}")
    
    # Beeswarm plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=features, show=False)
    plt.title('SHAP Summary Plot (Feature Impact)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    shap_summary_path = os.path.join(ARTIFACTS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ SHAP beeswarm plot saved to: {shap_summary_path}")
    
    # Create SHAP waterfall for fraud and legitimate cases
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    fraud_indices = np.where(y_shap == 1)[0]
    legit_indices = np.where(y_shap == 0)[0]
    
    # Fraud case
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        shap_vals = shap_values[fraud_idx]
        sorted_indices = np.argsort(np.abs(shap_vals))[-15:]
        sorted_shap = shap_vals[sorted_indices]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_names = [FEATURE_NAME_MAPPING.get(f, f)[:30] for f in sorted_features]
        
        colors = ['red' if v > 0 else 'blue' for v in sorted_shap]
        axes[0].barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
        axes[0].set_yticks(range(len(sorted_shap)))
        axes[0].set_yticklabels(sorted_names)
        axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[0].set_xlabel('SHAP Value')
        axes[0].set_title('SHAP Values for Fraud Case', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
    
    # Legitimate case
    if len(legit_indices) > 0:
        legit_idx = legit_indices[0]
        shap_vals = shap_values[legit_idx]
        sorted_indices = np.argsort(np.abs(shap_vals))[-15:]
        sorted_shap = shap_vals[sorted_indices]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_names = [FEATURE_NAME_MAPPING.get(f, f)[:30] for f in sorted_features]
        
        colors = ['red' if v > 0 else 'blue' for v in sorted_shap]
        axes[1].barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
        axes[1].set_yticks(range(len(sorted_shap)))
        axes[1].set_yticklabels(sorted_names)
        axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1].set_xlabel('SHAP Value')
        axes[1].set_title('SHAP Values for Legitimate Case', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    shap_cases_path = os.path.join(ARTIFACTS_DIR, "shap_case_comparison.png")
    plt.savefig(shap_cases_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ SHAP case comparison saved to: {shap_cases_path}")
    
    # Compute mean absolute SHAP values per feature
    mean_shap = pd.DataFrame({
        'feature': features,
        'feature_name': [FEATURE_NAME_MAPPING.get(f, f) for f in features],
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0),
        'min_shap': shap_values.min(axis=0),
        'max_shap': shap_values.max(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    shap_values_path = os.path.join(ARTIFACTS_DIR, "shap_feature_importance.csv")
    mean_shap.to_csv(shap_values_path, index=False)
    print(f"✓ SHAP feature importance saved to: {shap_values_path}")
    
    # Create SHAP dependence plots for top 6 features
    print("Creating SHAP dependence plots for top features...")
    top_features_idx = mean_shap.head(6)['feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features_idx):
        feature_idx = features.index(feature)
        ax = axes[i]
        
        # Get feature values and SHAP values
        feature_values = X_shap[:, feature_idx]
        feature_shap = shap_values[:, feature_idx]
        
        # Color by fraud label
        scatter = ax.scatter(feature_values, feature_shap, 
                            c=y_shap, cmap='coolwarm', alpha=0.5, s=10)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(FEATURE_NAME_MAPPING.get(feature, feature)[:30], fontsize=10)
        ax.set_ylabel('SHAP Value', fontsize=10)
        ax.set_title(f'SHAP Dependence: {feature[:25]}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.suptitle('SHAP Dependence Plots (Top 6 Features)\nRed=Fraud, Blue=Legitimate', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    dependence_path = os.path.join(ARTIFACTS_DIR, "shap_dependence_plots.png")
    plt.savefig(dependence_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ SHAP dependence plots saved to: {dependence_path}")
    
    # Create waterfall plot for a fraud case
    if len(fraud_indices) > 0:
        print("Creating SHAP waterfall plot for fraud case...")
        fraud_idx = fraud_indices[0]
        
        # Create SHAP Explanation object for waterfall
        shap_explanation = shap.Explanation(
            values=shap_values[fraud_idx],
            base_values=explainer.expected_value,
            data=X_shap[fraud_idx],
            feature_names=features
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title('SHAP Waterfall Plot - Fraud Case Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        waterfall_path = os.path.join(ARTIFACTS_DIR, "shap_waterfall_fraud.png")
        plt.savefig(waterfall_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✓ SHAP waterfall plot saved to: {waterfall_path}")
    
    # Create force plot data (save as HTML for interactive viewing)
    print("Creating SHAP force plot...")
    try:
        # For a single fraud prediction
        if len(fraud_indices) > 0:
            fraud_idx = fraud_indices[0]
            force_plot = shap.force_plot(
                explainer.expected_value, 
                shap_values[fraud_idx], 
                X_shap[fraud_idx],
                feature_names=features,
                matplotlib=True,
                show=False
            )
            plt.title('SHAP Force Plot - Fraud Case', fontsize=12)
            force_path = os.path.join(ARTIFACTS_DIR, "shap_force_plot_fraud.png")
            plt.savefig(force_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"✓ SHAP force plot saved to: {force_path}")
    except Exception as e:
        print(f"  Note: Could not create force plot: {e}")
    
    # Save raw SHAP values for all test samples
    print("Saving raw SHAP values...")
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df['actual_label'] = y_shap
    shap_raw_path = os.path.join(ARTIFACTS_DIR, "shap_values_raw.csv")
    shap_df.to_csv(shap_raw_path, index=False)
    print(f"✓ Raw SHAP values saved to: {shap_raw_path}")
    
    # Print top SHAP features
    print("\n" + "="*50)
    print("TOP 15 FEATURES BY MEAN |SHAP| VALUE")
    print("="*50)
    print(mean_shap[['feature', 'feature_name', 'mean_abs_shap', 'mean_shap']].head(15).to_string(index=False))
    
    return shap_values, mean_shap, explainer


def evaluate_model(y_test, y_pred, y_pred_proba):
    """Comprehensive model evaluation."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    specificity = tn / (tn + fp)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'auc_roc': auc_score,
        'average_precision': avg_precision,
        'accuracy': accuracy,
        'precision_fraud': precision,
        'recall_fraud': recall,
        'f1_score_fraud': f1,
        'specificity': specificity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    print(f"\n{'='*50}")
    print("  PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"  AUC-ROC Score:       {auc_score:.4f}")
    print(f"  Average Precision:   {avg_precision:.4f}")
    print(f"  Accuracy:            {accuracy:.4f}")
    print(f"  Precision (Fraud):   {precision:.4f}")
    print(f"  Recall (Fraud):      {recall:.4f}")
    print(f"  F1-Score (Fraud):    {f1:.4f}")
    print(f"  Specificity:         {specificity:.4f}")
    print(f"\n  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
    
    print(f"\n{'='*50}")
    print("  CONFUSION MATRIX")
    print(f"{'='*50}")
    print(f"                    Predicted")
    print(f"                    Legit    Fraud")
    print(f"  Actual Legit     {tn:>7,}  {fp:>7,}")
    print(f"         Fraud     {fn:>7,}  {tp:>7,}")
    print(f"\n  True Positives (Frauds Caught): {tp:,}")
    print(f"  False Negatives (Frauds Missed): {fn:,}")
    print(f"  False Positives (Legitimate Flagged): {fp:,}")
    
    # Save metrics
    metrics_path = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    return metrics, cm, class_report


def create_confusion_matrix_plot(cm, metrics):
    """Create and save confusion matrix visualization."""
    print("\n" + "="*60)
    print("CREATING CONFUSION MATRIX VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute values
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[0],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('Actual Label', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('Actual Label', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.suptitle(f"XGBoost Model Performance\nAUC-ROC: {metrics['auc_roc']:.4f} | "
                 f"Precision: {metrics['precision_fraud']:.4f} | "
                 f"Recall: {metrics['recall_fraud']:.4f}", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    return cm_path


def create_roc_pr_curves(y_test, y_pred_proba):
    """Create ROC and Precision-Recall curves."""
    print("\nCreating ROC and PR curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    axes[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    axes[0].fill_between(fpr, tpr, alpha=0.2, color='blue')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    axes[1].plot(recall_curve, precision_curve, color='green', lw=2, 
                 label=f'PR curve (AP = {avg_precision:.4f})')
    axes[1].fill_between(recall_curve, precision_curve, alpha=0.2, color='green')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower left')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(ARTIFACTS_DIR, "roc_pr_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ ROC and PR curves saved to: {curves_path}")


def create_probability_distribution_plot(y_test, y_pred_proba):
    """Create probability distribution plot by class label."""
    print("\nCreating probability distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Distribution by class
    ax1 = axes[0]
    ax1.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Legitimate', color='green', density=True)
    ax1.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper center')
    ax1.grid(alpha=0.3)
    
    # Overall distribution with log scale
    ax2 = axes[1]
    ax2.hist(y_pred_proba, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Overall Probability Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(ARTIFACTS_DIR, "probability_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Probability distribution plot saved to: {dist_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("PROBABILITY DISTRIBUTION STATISTICS")
    print("="*60)
    print(f"\nLegitimate Transactions (y=0):")
    print(f"  Mean Probability:   {y_pred_proba[y_test == 0].mean():.6f}")
    print(f"  Median Probability: {np.median(y_pred_proba[y_test == 0]):.6f}")
    print(f"  Std Deviation:      {y_pred_proba[y_test == 0].std():.6f}")
    print(f"  Min:                {y_pred_proba[y_test == 0].min():.6f}")
    print(f"  Max:                {y_pred_proba[y_test == 0].max():.6f}")
    
    print(f"\nFraud Transactions (y=1):")
    print(f"  Mean Probability:   {y_pred_proba[y_test == 1].mean():.6f}")
    print(f"  Median Probability: {np.median(y_pred_proba[y_test == 1]):.6f}")
    print(f"  Std Deviation:      {y_pred_proba[y_test == 1].std():.6f}")
    print(f"  Min:                {y_pred_proba[y_test == 1].min():.6f}")
    print(f"  Max:                {y_pred_proba[y_test == 1].max():.6f}")


def log_to_mlflow(model, metrics, best_params, class_stats, features, scaler, run_name=None):
    """Log experiment to MLflow."""
    print("\n" + "="*60)
    print("LOGGING TO MLFLOW")
    print("="*60)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Use provided run_name or generate default
    if run_name is None:
        run_name = f"XGBoost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("train_start_date", TRAIN_START_DATE)
        mlflow.log_param("train_end_date", TRAIN_END_DATE)
        mlflow.log_param("test_start_date", TEST_START_DATE)
        mlflow.log_param("test_end_date", TEST_END_DATE)
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("scale_pos_weight", class_stats['scale_pos_weight'])
        
        # Log best hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)
        
        # Log class distribution
        mlflow.log_param("train_fraud_count", class_stats['train_fraud'])
        mlflow.log_param("train_legit_count", class_stats['train_legit'])
        mlflow.log_param("test_fraud_count", class_stats['test_fraud'])
        mlflow.log_param("test_legit_count", class_stats['test_legit'])
        
        # Log metrics
        mlflow.log_metric("auc_roc", metrics['auc_roc'])
        mlflow.log_metric("average_precision", metrics['average_precision'])
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision_fraud", metrics['precision_fraud'])
        mlflow.log_metric("recall_fraud", metrics['recall_fraud'])
        mlflow.log_metric("f1_score_fraud", metrics['f1_score_fraud'])
        mlflow.log_metric("specificity", metrics['specificity'])
        mlflow.log_metric("false_positive_rate", metrics['false_positive_rate'])
        mlflow.log_metric("false_negative_rate", metrics['false_negative_rate'])
        mlflow.log_metric("true_positives", metrics['true_positives'])
        mlflow.log_metric("false_positives", metrics['false_positives'])
        mlflow.log_metric("false_negatives", metrics['false_negatives'])
        
        # Log model
        mlflow.xgboost.log_model(model, "xgboost_model")
        
        # Log all artifacts
        mlflow.log_artifacts(ARTIFACTS_DIR, "artifacts")
        
        # Log feature list
        with open(os.path.join(ARTIFACTS_DIR, "features.json"), 'w') as f:
            json.dump(features, f, indent=2)
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "features.json"))
        
        # Log scaler
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path)
        
        run_id = mlflow.active_run().info.run_id
        print(f"✓ MLflow Run ID: {run_id}")
        print(f"✓ Experiment: {EXPERIMENT_NAME}")
        print(f"✓ Tracking URI: {MLFLOW_TRACKING_URI}")
    
    return run_id


def main(run_name=None):
    """Main execution function."""
    print("\n" + "="*70)
    print("  XGBOOST FRAUD DETECTION MODEL TRAINING")
    print("  " + "="*66)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if run_name:
        print(f"  Run Name: {run_name}")
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Extract data
        df_train, df_test = extract_data(spark)
        
        # Analyze class distribution
        class_stats = analyze_class_distribution(df_train, df_test)
        
        # Prepare features
        (X_train, X_test, X_train_scaled, X_test_scaled, 
         y_train, y_test, features, scaler, test_trans_ids) = prepare_features(df_train, df_test)
        
        # Feature analysis
        corr_df = compute_correlation_with_target(df_train, features)
        variance_df = compute_feature_variance(df_train, features)
        vif_df = compute_vif(X_train, features)
        
        # Hyperparameter tuning
        best_params, best_cv_score = run_hyperparameter_search(
            X_train_scaled, y_train, class_stats['scale_pos_weight']
        )
        
        # Train final model
        model = train_final_model(
            X_train_scaled, y_train, best_params, class_stats['scale_pos_weight']
        )
        
        # Get predictions
        y_pred, y_pred_proba, predictions_df = get_predictions(
            model, X_test_scaled, test_trans_ids, y_test
        )
        
        # Feature importance
        importance_df = compute_feature_importance(model, features)
        
        # SHAP analysis
        shap_values, shap_importance, shap_explainer = compute_shap_values(
            model, X_train_scaled, X_test_scaled, features, y_test
        )
        
        # Evaluate model
        metrics, cm, class_report = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        create_confusion_matrix_plot(cm, metrics)
        create_roc_pr_curves(y_test, y_pred_proba)
        create_probability_distribution_plot(y_test.values, y_pred_proba)
        
        # Log to MLflow
        run_id = log_to_mlflow(model, metrics, best_params, class_stats, features, scaler, run_name)
        
        print("\n" + "="*70)
        print("  TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"  Output Directory: {OUTPUT_DIR}")
        print(f"  MLflow Run ID: {run_id}")
        print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
    finally:
        # Stop Spark session
        spark.stop()
        print("\n✓ Spark session stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Fraud Detection Model Training")
    parser.add_argument(
        "--run-name", 
        type=str, 
        default=None,
        help="Name for the MLflow experiment run (default: XGBoost_YYYYMMDD_HHMMSS)"
    )
    args = parser.parse_args()
    
    main(run_name=args.run_name)
