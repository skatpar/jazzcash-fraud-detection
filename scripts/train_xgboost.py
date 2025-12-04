#!/usr/bin/env python3
"""
Fraud Detection XGBoost Pipeline
================================
Trains an XGBoost model on downsampled Spark DataFrame converted to pandas.

Author: AI Team
Date: November 2025
"""
import os
import sys
import logging
import time
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_config():
    return {
        "data": {
            "table_name": "stixor_fraud_features_distributed",
            "start_date": "2025-06-01",
            "end_date": "2025-06-30",
            "target_column": "fraud_flag",
            "selected_features": [
                'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 
                'start_balance', 'trx_amt',
                'hour_of_day','is_weekend',
                'txn_txns_3d', 'txn_total_amount_3d', 'txn_avg_amount_3d',
                'txn_max_amount_3d', 'txn_min_amount_3d', 'txn_unique_types_3d', 
                'txn_multi_channel_recent', 'txn_amount_deviation_from_avg'
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
        "output_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis/xgboost"
    }

class FraudXGBPipeline:
    def __init__(self, config):
        self.config = config
        self.spark = None
        self.logger = None
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self):
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        log_filename = f"xgboost_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.config['output_dir'], log_filename)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger('FraudXGBPipeline')
        self.logger.info("=" * 80)
        self.logger.info("XGBOOST FRAUD DETECTION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Log file: {log_path}")

    def initialize_spark(self):
        self.logger.info("Initializing Spark session...")
        packages = [
            "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
            "com.clickhouse:clickhouse-client:0.9.4",
            "com.clickhouse:clickhouse-http-client:0.9.4",
            "org.apache.httpcomponents.client5:httpclient5:5.2.1"
        ]
        spark_cfg = self.config.get('spark', {})
        self.spark = (SparkSession.builder
            .appName("spark-clickhouse-xgboost-fraud-detection")
            .master("local[*]")
            .config("spark.jars.packages", ",".join(packages))
            .config("spark.executor.memory", spark_cfg.get('executor_memory', '150g'))
            .config("spark.executor.memoryOverhead", spark_cfg.get('executor_memory_overhead', '5g'))
            .config("spark.driver.memory", spark_cfg.get('driver_memory', '8g'))
            .config("spark.executor.cores", str(spark_cfg.get('executor_cores', 32)))
            .config("spark.executor.instances", str(spark_cfg.get('executor_instances', 2)))
            .config("spark.sql.shuffle.partitions", str(spark_cfg.get('shuffle_partitions', 200)))
            .config("spark.default.parallelism", str(spark_cfg.get('parallelism', 96)))
            .getOrCreate()
        )
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

    def load_data(self, start_date=None, end_date=None, eval=False):
        self.logger.info("LOADING DATA FROM CLICKHOUSE")
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        if start_date is None:
            start_date = data_cfg['start_date']
        if end_date is None:
            end_date = data_cfg['end_date']
        if eval:
            query = f"""
                SELECT {', '.join(data_cfg['selected_features'])}
                FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
                WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                AND mbar_account_type_name = 'Customer Account'
                AND trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000
            """
        else:
            query = f"""
                SELECT {', '.join(data_cfg['selected_features'])}
                FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
                WHERE (cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                AND mbar_account_type_name = 'Customer Account'
                AND trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000)
                OR (fraud_flag=1 and cutoff_date<='{end_date}' and trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000)
            """
        start = time.time()
        df = self.spark.sql(query)
        count = df.count()
        duration = time.time() - start
        self.logger.info(f"✅ Loaded {count:,} rows in {duration:.2f}s")
        return df

    def downsample_data(self, df, sample_rate=100/554601):
        self.logger.info("DOWNSAMPLING NON-FRAUD DATA")
        fraud_df = df.filter(col('fraud_flag') == 1)
        non_fraud_df = df.filter(col('fraud_flag') == 0)
        non_fraud_sampled = non_fraud_df.sample(withReplacement=False, fraction=sample_rate, seed=self.config['training']['random_seed'])
        balanced_df = fraud_df.union(non_fraud_sampled)
        self.logger.info(f"After downsampling - Fraud: {fraud_df.count()}, Non-fraud: {non_fraud_sampled.count()}, Total: {balanced_df.count()}")
        return balanced_df

    def train_xgboost(self, df_spark):
        self.logger.info("TRAINING XGBOOST MODEL")
        df_pd = df_spark.select(self.config['data']['selected_features']).toPandas()
        target_col = self.config['data']['target_column']
        excluded = [target_col, 'cutoff_date', 'mbar_account_type_name']
        feature_cols = [f for f in self.config['data']['selected_features'] if f not in excluded]
        for col in ['trx_channel', 'trx_type']:
            if col in df_pd.columns:
                df_pd[col] = df_pd[col].astype('category').cat.codes
        X = df_pd[feature_cols].values
        y = df_pd[target_col].values
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config['training']['random_seed'],
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X, y)
        self.xgb_model = xgb_model
        self.logger.info("✅ XGBoost training complete")
        return xgb_model

    def evaluate_xgboost(self, df_spark):
        self.logger.info("EVALUATING XGBOOST MODEL")
        df_pd = df_spark.select(self.config['data']['selected_features']).toPandas()
        target_col = self.config['data']['target_column']
        excluded = [target_col, 'cutoff_date', 'mbar_account_type_name']
        feature_cols = [f for f in self.config['data']['selected_features'] if f not in excluded]
        for col in ['trx_channel', 'trx_type']:
            if col in df_pd.columns:
                df_pd[col] = df_pd[col].astype('category').cat.codes
        X = df_pd[feature_cols].values
        y = df_pd[target_col].values
        y_pred = self.xgb_model.predict(X)
        y_prob = self.xgb_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        self.logger.info(f"AUC-ROC: {auc:.4f}")
        self.logger.info(f"Accuracy: {acc:.4f}")
        self.logger.info(f"Precision: {prec:.4f}")
        self.logger.info(f"Recall: {rec:.4f}")
        self.logger.info(f"F1-Score: {f1:.4f}")
        cm_path = os.path.join(self.config['output_dir'], 'xgboost_confusion_matrix.csv')
        pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']).to_csv(cm_path)
        self.logger.info(f"Confusion matrix saved to: {cm_path}")
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('XGBoost - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        cm_plot_path = os.path.join(self.config['output_dir'], 'xgboost_confusion_matrix.png')
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion matrix plot saved to: {cm_plot_path}")
        return {'auc': auc, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    def run(self):
        self.initialize_spark()
        # Training data: June
        df_train = self.load_data(start_date="2025-06-01", end_date="2025-06-30")
        df_sampled = self.downsample_data(df_train)
        self.train_xgboost(df_sampled)
        # Test data: July
        df_test = self.load_data(start_date="2025-07-01", end_date="2025-07-31",eval=True)
        df_test_sampled = self.downsample_data(df_test)
        self.evaluate_xgboost(df_test_sampled)


def main():
    config = load_config()
    pipeline = FraudXGBPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()
