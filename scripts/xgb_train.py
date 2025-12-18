#!/usr/bin/env python3
"""
XGBoost Fraud Detection Training Pipeline
==========================================
Pipeline using XGBoost with robust error handling to skip erroneous rows.

Key Features:
- XGBoost natively handles missing  values (NaN) in numeric features
- Infinite values are converted to NaN for XGBoost to handle
- Categorical features are encoded and missing values filled
- Only rows with invalid categorical or target values are removed
- Comprehensive error logging at each step

Author: AI Team
Date: November 2025
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, rand

import xgboost as xgb
from xgboost.spark import SparkXGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import logging


class XGBoostFraudPipeline:
    """XGBoost fraud detection pipeline with robust error handling."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spark = None
        self.logger = None
        self.model = None
        self.spark_model = None
        self.label_encoders = {}
        self.feature_cols = []
        self.use_spark_xgboost = config.get('use_spark_xgboost', True)
        
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config['model_dir'], self.config['log_dir'], self.config['analysis_dir']]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_filename = f"xgboost_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.config['log_dir'], log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('XGBoostFraudPipeline')
        self.logger.info("=" * 80)
        self.logger.info("XGBOOST FRAUD DETECTION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Log file: {log_path}")
    
    def initialize_spark(self):
        """Initialize Spark with ClickHouse catalog."""
        self.logger.info("Initializing Spark session...")
        
        try:
            existing = SparkSession.getActiveSession()
            if existing:
                existing.stop()
                self.logger.info("Stopped existing Spark session")
        except:
            pass
        
        packages = [
            "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
            "com.clickhouse:clickhouse-client:0.9.4",
            "com.clickhouse:clickhouse-http-client:0.9.4",
            "org.apache.httpcomponents.client5:httpclient5:5.2.1"
        ]
        
        # Get Spark configuration from config
        spark_cfg = self.config.get('spark', {})
        
        # Set environment variable for Arrow to use legacy memory allocation
        import os
        os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'
        
        # Java options to allow Unsafe operations (needed for XGBoost on Spark)
        java_opts = [
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
            "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED",
            "-Dio.netty.tryReflectionSetAccessible=true",
        ]
        java_opts_str = " ".join(java_opts)
        
        self.spark = (SparkSession.builder
            .appName("xgboost-fraud-detection")
            .master("spark://10.205.161.118:7077")
            .config("spark.jars.packages", ",".join(packages))
            .config("spark.executor.memory", spark_cfg.get('executor_memory', '150g'))
            .config("spark.executor.memoryOverhead", spark_cfg.get('executor_memory_overhead', '5g'))
            .config("spark.driver.memory", spark_cfg.get('driver_memory', '8g'))
            .config("spark.executor.cores", str(spark_cfg.get('executor_cores', 32)))
            .config("spark.executor.instances", str(spark_cfg.get('executor_instances', 2)))
            .config("spark.sql.shuffle.partitions", str(spark_cfg.get('shuffle_partitions', 200)))
            .config("spark.default.parallelism", str(spark_cfg.get('parallelism', 96)))
            # Java options to allow Unsafe operations (for Arrow/XGBoost compatibility)
            .config("spark.driver.extraJavaOptions", java_opts_str)
            .config("spark.executor.extraJavaOptions", java_opts_str)
            # Disable Arrow to avoid java.lang.UnsupportedOperationException with Unsafe
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
            # Additional XGBoost Spark configurations
            .config("spark.task.cpus", "1")
            .config("spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")
            .getOrCreate()
        )
        
        # Configure ClickHouse catalog
        ch = self.config['clickhouse']
        self.spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
        self.spark.conf.set("spark.sql.catalog.clickhouse.host", ch['host'])
        self.spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
        self.spark.conf.set("spark.sql.catalog.clickhouse.http_port", str(ch['http_port']))
        self.spark.conf.set("spark.sql.catalog.clickhouse.user", ch['user'])
        self.spark.conf.set("spark.sql.catalog.clickhouse.password", ch['password'])
        self.spark.conf.set("spark.sql.catalog.clickhouse.database", ch['database'])
        self.spark.conf.set("spark.clickhouse.write.format", "json")
        
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")
        self.logger.info(f"   Master: {self.spark.sparkContext.master}")
        self.logger.info(f"   Executor Memory: {spark_cfg.get('executor_memory', '150g')}")
        self.logger.info(f"   Executor Cores: {spark_cfg.get('executor_cores', 32)}")
        self.logger.info(f"   Executor Instances: {spark_cfg.get('executor_instances', 2)}")
        self.logger.info(f"   Shuffle Partitions: {spark_cfg.get('shuffle_partitions', 200)}")
        self.logger.info(f"   ClickHouse catalog: clickhouse.{ch['database']}")
    
    def load_data(self):
        """Load data from ClickHouse with balanced sampling."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING DATA")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {'Spark XGBoost (Distributed)' if self.use_spark_xgboost else 'Standard XGBoost'}")
        
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        
        # Load ALL fraud cases
        self.logger.info("Loading fraud cases (fraud_flag = 1)...")
        fraud_query = f"""
            SELECT {', '.join(data_cfg['selected_features'])}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE cutoff_date BETWEEN '{data_cfg['start_date']}' AND '{data_cfg['end_date']}'
                AND mbar_account_type_name = 'Customer Account'
                AND fraud_flag = 1
        """
        
        start = time.time()
        df_fraud_spark = self.spark.sql(fraud_query)
        fraud_count = df_fraud_spark.count()
        
        if not self.use_spark_xgboost:
            df_fraud = df_fraud_spark.toPandas()
        else:
            df_fraud = df_fraud_spark
        
        fraud_duration = time.time() - start
        self.logger.info(f"✅ Loaded {fraud_count:,} fraud cases in {fraud_duration:.2f}s")
        
        # Load and sample NON-FRAUD cases
        sample_rate = data_cfg.get('non_fraud_sample_rate', 0.01)
        self.logger.info(f"Loading non-fraud cases (fraud_flag = 0, {sample_rate*100:.0f}% sample)...")
        
        non_fraud_query = f"""
            SELECT {', '.join(data_cfg['selected_features'])}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE cutoff_date BETWEEN '{data_cfg['start_date']}' AND '{data_cfg['end_date']}'
                AND mbar_account_type_name = 'Customer Account'
                AND fraud_flag = 0
        """
        
        start = time.time()
        df_non_fraud_full = self.spark.sql(non_fraud_query)
        total_non_fraud = df_non_fraud_full.count()
        
        # Sample using Spark
        df_non_fraud_spark = df_non_fraud_full.sample(
            withReplacement=False,
            fraction=sample_rate,
            seed=self.config['training']['random_seed']
        )
        non_fraud_count = df_non_fraud_spark.count()
        
        if not self.use_spark_xgboost:
            df_non_fraud = df_non_fraud_spark.toPandas()
        else:
            df_non_fraud = df_non_fraud_spark
        
        non_fraud_duration = time.time() - start
        
        self.logger.info(f"✅ Sampled {non_fraud_count:,} non-fraud cases ({sample_rate*100:.0f}%) in {non_fraud_duration:.2f}s")
        self.logger.info(f"   Total non-fraud available: {total_non_fraud:,}")
        
        # Combine datasets
        self.logger.info("Combining datasets...")
        if self.use_spark_xgboost:
            df = df_fraud.union(df_non_fraud)
            df = df.orderBy(rand(self.config['training']['random_seed']))
            total_count = df.count()
        else:
            df = pd.concat([df_fraud, df_non_fraud], ignore_index=True)
            df = df.sample(frac=1, random_state=self.config['training']['random_seed']).reset_index(drop=True)
            total_count = len(df)
        
        self.logger.info(f"✅ Combined dataset: {total_count:,} rows")
        self.logger.info(f"   Fraud: {fraud_count:,} ({fraud_count/total_count*100:.2f}%)")
        self.logger.info(f"   Non-Fraud: {non_fraud_count:,} ({non_fraud_count/total_count*100:.2f}%)")
        self.logger.info(f"   Class Ratio: 1:{non_fraud_count/fraud_count:.2f}")
        
        return df
    
    def preprocess_spark_data(self, df: DataFrame):
        """Preprocess Spark DataFrame for Spark XGBoost."""
        self.logger.info("=" * 80)
        self.logger.info("PREPROCESSING DATA (SPARK)")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        
        # Identify categorical and numeric columns
        categorical_cols = ['trx_channel', 'trx_type', 'mbar_registered_channel']
        excluded_cols = [data_cfg['target_column'], 'cutoff_date', 'mbar_account_type_name']
        all_features = [f for f in data_cfg['selected_features'] if f not in excluded_cols]
        numeric_cols = [f for f in all_features if f not in categorical_cols]
        
        self.logger.info(f"Feature types:")
        self.logger.info(f"   Categorical: {len(categorical_cols)}")
        self.logger.info(f"   Numeric: {len(numeric_cols)}")
        
        # Build preprocessing pipeline
        stages = []
        indexed_cols = []
        
        # String Indexing for categorical features
        for col_name in categorical_cols:
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_idx",
                handleInvalid="keep"
            )
            stages.append(indexer)
            indexed_cols.append(f"{col_name}_idx")
        
        # Combine numeric and indexed columns
        self.feature_cols = numeric_cols + indexed_cols
        
        # Vector Assembler
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features",
            handleInvalid="skip"  # Skip rows with invalid values
        )
        stages.append(assembler)
        
        # Create and fit pipeline
        self.logger.info(f"Building preprocessing pipeline with {len(stages)} stages...")
        pipeline = Pipeline(stages=stages)
        
        start = time.time()
        pipeline_model = pipeline.fit(df)
        df_transformed = pipeline_model.transform(df)
        duration = time.time() - start
        
        self.logger.info(f"✅ Preprocessing completed in {duration:.2f}s")
        self.logger.info(f"   Features: {len(self.feature_cols)}")
        
        # Cache the transformed data for better performance
        df_transformed = df_transformed.select('features', data_cfg['target_column']).cache()
        count = df_transformed.count()
        self.logger.info(f"✅ Transformed dataset: {count:,} rows (cached)")
        
        return df_transformed
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess data with error handling."""
        self.logger.info("=" * 80)
        self.logger.info("PREPROCESSING DATA")
        self.logger.info("=" * 80)
        
        initial_count = len(df)
        data_cfg = self.config['data']
        
        # Identify categorical and numeric columns
        categorical_cols = ['trx_channel', 'trx_type', 'mbar_registered_channel']
        excluded_cols = [data_cfg['target_column'], 'cutoff_date', 'mbar_account_type_name']
        all_features = [f for f in data_cfg['selected_features'] if f not in excluded_cols]
        numeric_cols = [f for f in all_features if f not in categorical_cols]
        
        self.logger.info(f"Feature types:")
        self.logger.info(f"   Categorical: {len(categorical_cols)}")
        self.logger.info(f"   Numeric: {len(numeric_cols)}")
        
        # Handle missing values (XGBoost can handle NaN in numeric columns)
        self.logger.info("Handling missing values...")
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            self.logger.info(f"   Found {missing_before:,} missing values")
            
            # Fill categorical missing values (XGBoost cannot handle NaN in categorical)
            for col in categorical_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    missing_count = df[col].isnull().sum()
                    df[col].fillna('UNKNOWN', inplace=True)
                    self.logger.info(f"   {col}: filled {missing_count:,} missing values with 'UNKNOWN'")
            
            # For numeric columns, XGBoost can handle NaN natively
            # But we'll log which columns have missing values
            for col in numeric_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    missing_count = df[col].isnull().sum()
                    self.logger.info(f"   {col}: {missing_count:,} missing values (XGBoost will handle)")
            
            self.logger.info(f"✅ Missing values handled (categorical filled, numeric kept for XGBoost)")
        else:
            self.logger.info(f"✅ No missing values detected")
        
        # Encode categorical features
        self.logger.info("Encoding categorical features...")
        
        for col in categorical_cols:
            if col in df.columns:
                try:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    self.logger.info(f"   {col}: {len(le.classes_)} unique values")
                except Exception as e:
                    self.logger.warning(f"   Failed to encode {col}: {str(e)}")
                    df[f'{col}_encoded'] = 0
        
        # Handle infinite values in numeric columns
        self.logger.info("Checking for infinite values...")
        inf_count = 0
        for col in numeric_cols:
            if col in df.columns:
                # Check for positive and negative infinity
                inf_mask = np.isinf(df[col])
                if inf_mask.any():
                    count = inf_mask.sum()
                    inf_count += count
                    # Replace inf with NaN so XGBoost can handle it
                    df.loc[inf_mask, col] = np.nan
                    self.logger.warning(f"   {col}: {count:,} infinite values, converting to NaN for XGBoost")
        
        if inf_count > 0:
            self.logger.info(f"✅ Converted {inf_count:,} infinite values to NaN (XGBoost will handle)")
        else:
            self.logger.info(f"✅ No infinite values detected")
        
        # Remove rows with invalid encoded values only (XGBoost handles NaN in numeric)
        self.logger.info("Validating data quality...")
        encoded_cols = [f'{col}_encoded' for col in categorical_cols if col in df.columns]
        self.feature_cols = numeric_cols + encoded_cols
        
        # Only check encoded categorical columns and target for validity
        # XGBoost can handle NaN in numeric columns
        valid_mask = pd.Series([True] * len(df), index=df.index)
        
        # Check encoded categorical columns must not have NaN
        for col in encoded_cols:
            if col in df.columns:
                valid_mask &= df[col].notna()
        
        # Check target column must not have NaN
        if data_cfg['target_column'] in df.columns:
            valid_mask &= df[data_cfg['target_column']].notna()
        
        df_clean = df[valid_mask].copy()
        removed_count = initial_count - len(df_clean)
        
        if removed_count > 0:
            self.logger.warning(f"⚠️  Removed {removed_count:,} rows with invalid categorical or target values ({removed_count/initial_count*100:.2f}%)")
        else:
            self.logger.info(f"✅ No rows removed (XGBoost handles missing numeric values)")
        
        self.logger.info(f"✅ Final dataset: {len(df_clean):,} rows")
        
        # Prepare feature matrix and target
        # XGBoost will handle NaN values in numeric columns automatically
        X = df_clean[self.feature_cols].values
        y = df_clean[data_cfg['target_column']].values
        
        self.logger.info(f"✅ Feature matrix created:")
        self.logger.info(f"   Shape: {X.shape}")
        self.logger.info(f"   Features: {len(self.feature_cols)}")
        
        return X, y, df_clean
    
    def train_spark_model(self, df: DataFrame):
        """Train Spark XGBoost model on distributed data."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING SPARK XGBOOST MODEL (DISTRIBUTED)")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        
        # Calculate class distribution for scale_pos_weight
        fraud_count = df.filter(col(data_cfg['target_column']) == 1).count()
        non_fraud_count = df.filter(col(data_cfg['target_column']) == 0).count()
        scale_pos_weight = non_fraud_count / fraud_count
        
        self.logger.info(f"⚖️  Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # Spark XGBoost parameters
        # Note: SparkXGBClassifier doesn't allow custom 'objective' or 'eval_metric'
        # It automatically uses 'binary:logistic' for binary classification
        xgb_cfg = self.config['models']['xgboost']
        params = {
            'features_col': 'features',
            'label_col': data_cfg['target_column'],
            'max_depth': xgb_cfg['max_depth'],
            'learning_rate': xgb_cfg['learning_rate'],
            'n_estimators': xgb_cfg.get('n_estimators', 200),
            'subsample': xgb_cfg['subsample'],
            'colsample_bytree': xgb_cfg['colsample_bytree'],
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.config['training']['random_seed'],
            'tree_method': 'hist',
            'missing': 0.0,  # Value to treat as missing
            'num_workers': self.config['spark'].get('executor_instances', 2),
            'use_gpu': False,
            'verbosity': 1
        }
        
        self.logger.info(f"Spark XGBoost Parameters:")
        for key, value in params.items():
            self.logger.info(f"   {key}: {value}")
        
        # Train model
        self.logger.info("Training (distributed across Spark cluster)...")
        start = time.time()
        
        try:
            self.spark_model = SparkXGBClassifier(**params)
            self.spark_model = self.spark_model.fit(df)
            
            duration = time.time() - start
            self.logger.info(f"✅ Training completed in {duration:.2f}s ({duration/60:.2f} min)")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise
        
        return self.spark_model
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING XGBOOST MODEL")
        self.logger.info("=" * 80)
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        self.logger.info(f"⚖️  Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # XGBoost parameters with missing value handling
        xgb_cfg = self.config['models']['xgboost']
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': xgb_cfg['max_depth'],
            'learning_rate': xgb_cfg['learning_rate'],
            'n_estimators': xgb_cfg['n_estimators'],
            'subsample': xgb_cfg['subsample'],
            'colsample_bytree': xgb_cfg['colsample_bytree'],
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.config['training']['random_seed'],
            'tree_method': 'hist',
            'n_jobs': -1,
            'verbosity': 1,
            # Parameters for handling missing/invalid values
            'missing': np.nan,  # Treat NaN as missing value
            'enable_categorical': False,  # Disable categorical encoding (we handle it)
            'max_cat_to_onehot': 4,  # One-hot encode categoricals with <= 4 values
            'validate_parameters': True  # Validate parameters to catch errors early
        }
        
        self.logger.info(f"XGBoost Parameters:")
        for key, value in params.items():
            self.logger.info(f"   {key}: {value}")
        
        # Train model with error handling
        self.logger.info("Training...")
        start = time.time()
        
        try:
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(
                X, y,
                eval_set=[(X, y)],
                verbose=False
            )
            
            duration = time.time() - start
            self.logger.info(f"✅ Training completed in {duration:.2f}s ({duration/60:.2f} min)")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            self.logger.error(f"   Data shape: X={X.shape}, y={y.shape}")
            self.logger.error(f"   NaN in X: {np.isnan(X).sum()}, NaN in y: {np.isnan(y).sum()}")
            self.logger.error(f"   Inf in X: {np.isinf(X).sum()}, Inf in y: {np.isinf(y).sum()}")
            raise
        
        return self.model
    
    def evaluate_spark_model(self, df: DataFrame):
        """Evaluate Spark XGBoost model."""
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION (SPARK)")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        
        # Make predictions
        predictions = self.spark_model.transform(df)
        
        # Evaluate using Spark ML evaluators
        binary_eval = BinaryClassificationEvaluator(
            labelCol=data_cfg['target_column'],
            rawPredictionCol='rawPrediction',
            metricName='areaUnderROC'
        )
        
        multiclass_eval = MulticlassClassificationEvaluator(
            labelCol=data_cfg['target_column'],
            predictionCol='prediction'
        )
        
        auc = binary_eval.evaluate(predictions)
        accuracy = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: 'accuracy'})
        precision = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: 'weightedPrecision'})
        recall = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: 'weightedRecall'})
        f1 = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: 'f1'})
        
        self.logger.info(f"📊 Performance Metrics:")
        self.logger.info(f"   AUC-ROC:           {auc:.4f}")
        self.logger.info(f"   Accuracy:          {accuracy:.4f}")
        self.logger.info(f"   Precision:         {precision:.4f}")
        self.logger.info(f"   Recall:            {recall:.4f}")
        self.logger.info(f"   F1-Score:          {f1:.4f}")
        
        # Get confusion matrix (need to collect to driver for this)
        self.logger.info("Computing confusion matrix...")
        pred_and_label = predictions.select('prediction', data_cfg['target_column']).collect()
        y_true = [row[data_cfg['target_column']] for row in pred_and_label]
        y_pred = [row['prediction'] for row in pred_and_label]
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.logger.info(f"📊 Confusion Matrix:")
        self.logger.info(f"   True Negatives:  {tn:,}")
        self.logger.info(f"   False Positives: {fp:,}")
        self.logger.info(f"   False Negatives: {fn:,}")
        self.logger.info(f"   True Positives:  {tp:,}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray):
        """Evaluate the trained model."""
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION")
        self.logger.info("=" * 80)
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        ap_score = average_precision_score(y, y_pred_proba)
        
        self.logger.info(f"📊 Performance Metrics:")
        self.logger.info(f"   Accuracy:          {accuracy:.4f}")
        self.logger.info(f"   Precision:         {precision:.4f}")
        self.logger.info(f"   Recall:            {recall:.4f}")
        self.logger.info(f"   F1-Score:          {f1:.4f}")
        self.logger.info(f"   AUC-ROC:           {auc:.4f}")
        self.logger.info(f"   Average Precision: {ap_score:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.logger.info(f"📊 Confusion Matrix:")
        self.logger.info(f"   True Negatives:  {tn:,}")
        self.logger.info(f"   False Positives: {fp:,}")
        self.logger.info(f"   False Negatives: {fn:,}")
        self.logger.info(f"   True Positives:  {tp:,}")
        
        # Classification Report
        self.logger.info(f"📊 Classification Report:")
        report = classification_report(y, y_pred, target_names=['Non-Fraud', 'Fraud'])
        for line in report.split('\n'):
            if line.strip():
                self.logger.info(f"   {line}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'average_precision': ap_score,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
    
    def analyze_feature_importance(self):
        """Extract and save feature importance."""
        self.logger.info("=" * 80)
        self.logger.info("FEATURE IMPORTANCE")
        self.logger.info("=" * 80)
        
        feature_importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Top 20 features:")
        for idx, row in importance_df.head(20).iterrows():
            self.logger.info(f"   {row['feature']:<40} {row['importance']:.6f}")
        
        # Save to CSV
        csv_path = os.path.join(self.config['analysis_dir'], 'xgboost_feature_importance.csv')
        importance_df.to_csv(csv_path, index=False)
        self.logger.info(f"💾 Saved to: {csv_path}")
        
        # Plot
        plt.figure(figsize=(12, 10))
        top_n = min(25, len(importance_df))
        top_features = importance_df.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - XGBoost Model', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.config['analysis_dir'], 'xgboost_feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"💾 Plot saved to: {plot_path}")
        
        return importance_df
    
    def save_model(self):
        """Save the trained model and artifacts."""
        self.logger.info("=" * 80)
        self.logger.info("SAVING MODEL AND ARTIFACTS")
        self.logger.info("=" * 80)
        
        # Save XGBoost model
        model_path = os.path.join(self.config['model_dir'], 'xgboost_fraud_model.json')
        self.model.save_model(model_path)
        self.logger.info(f"✅ XGBoost model saved to: {model_path}")
        
        # Save label encoders
        import pickle
        encoders_path = os.path.join(self.config['model_dir'], 'xgboost_label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        self.logger.info(f"✅ Label encoders saved to: {encoders_path}")
        
        # Save feature names
        features_path = os.path.join(self.config['model_dir'], 'xgboost_features.json')
        with open(features_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_cols,
                'categorical_features': list(self.label_encoders.keys()),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        self.logger.info(f"✅ Feature names saved to: {features_path}")
    
    def run(self):
        """Execute the complete pipeline."""
        try:
            start_time = time.time()
            
            # Initialize
            self.initialize_spark()
            
            # Load data
            df = self.load_data()
            
            if self.use_spark_xgboost:
                # Spark XGBoost path
                self.logger.info("\n🚀 Using Spark XGBoost (Distributed Training)")
                
                # Preprocess Spark DataFrame
                df_transformed = self.preprocess_spark_data(df)
                
                # Train on Spark
                self.train_spark_model(df_transformed)
                
                # Evaluate on Spark
                metrics = self.evaluate_spark_model(df_transformed)
                
                # Save Spark model
                model_path = os.path.join(self.config['model_dir'], 'spark_xgboost_fraud_model')
                self.spark_model.save(model_path)
                self.logger.info(f"💾 Spark XGBoost model saved to: {model_path}")
                
            else:
                # Standard XGBoost path
                self.logger.info("\n🖥️  Using Standard XGBoost (Single Machine)")
                
                # Preprocess
                X, y, df_clean = self.preprocess_data(df)
                
                # Train
                self.train_model(X, y)
                
                # Evaluate
                metrics = self.evaluate_model(X, y)
                
                # Feature importance
                self.analyze_feature_importance()
                
                # Save
                self.save_model()
            
            # Summary
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
            self.logger.info(f"Model: {'Spark XGBoost (Distributed)' if self.use_spark_xgboost else 'XGBoost'}")
            self.logger.info(f"AUC-ROC: {metrics['auc']:.4f}")
            self.logger.info(f"F1-Score: {metrics['f1']:.4f}")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration."""
    default_config = {
        "use_spark_xgboost": False,  # Use standard XGBoost (set to True for distributed, but requires Java config on workers)
        "data": {
            "table_name": "stixor_fraud_features_distributed",
            "start_date": "2025-06-01",
            "end_date": "2025-06-30",
            "target_column": "fraud_flag",
            "non_fraud_sample_rate": 0.01,
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
            ]
        },
        "clickhouse": {
            "host": "localhost",
            "port": 9000,
            "http_port": 8123,
            "database": "public",
            "user": "default",
            "password": "DfsTeChB1"
        },
        "spark": {
            "executor_memory": "150g",
            "executor_memory_overhead": "5g",
            "driver_memory": "8g",
            "executor_cores": 32,
            "executor_instances": 2,
            "shuffle_partitions": 200,
            "parallelism": 96
        },
        "training": {
            "random_seed": 42
        },
        "models": {
            "xgboost": {
                "max_depth": 10,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }
        },
        "model_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models",
        "log_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models/logs",
        "analysis_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis"
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config


def main():
    parser = argparse.ArgumentParser(description='XGBoost Fraud Detection Pipeline')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--sample-rate', type=float, help='Non-fraud sample rate (0.0-1.0)')
    parser.add_argument('--use-spark', action='store_true', help='Use Spark XGBoost for distributed training')
    parser.add_argument('--use-standard', action='store_true', help='Use standard XGBoost (single machine)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    if args.sample_rate:
        config['data']['non_fraud_sample_rate'] = args.sample_rate
    if args.use_spark:
        config['use_spark_xgboost'] = True
    if args.use_standard:
        config['use_spark_xgboost'] = False
    
    # Run pipeline
    pipeline = XGBoostFraudPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
