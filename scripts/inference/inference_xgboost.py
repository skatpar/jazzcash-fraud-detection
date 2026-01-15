#!/usr/bin/env python3
"""
Fraud Detection Inference Script (XGBoost)
=========================================
Loads trained XGBoost model and generates fraud scores for specific transaction IDs.
Pulls features from ClickHouse for the given transaction ID(s).

Usage:
    python inference_xgboost.py --trans_id <transaction_id>
    python inference_xgboost.py --trans_ids 123456 789012 345678

Author: AI Team
Date: November 2025
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tabulate import tabulate
import xgboost as xgb
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def load_config(config_path: Optional[str] = None) -> Dict:
    default_config = {
        "data": {
            "table_name": "stixor_fraud_features_distributed",
            "target_column": "fraud_flag",
            "selected_features": [
                'trans_id', 'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 
                'start_balance', 'trx_amt', 'hour_of_day', 'is_weekend',
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
            "driver_memory": "4g",
            "shuffle_partitions": 10
        },
        "model_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models",
        "analysis_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis/xgboost"
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    return default_config

class XGBoostFraudInference:
    def __init__(self, config: Dict):
        self.config = config
        self.spark = None
        self.logger = None
        self.model = None
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger('XGBoostFraudInference')

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
            .appName("fraud-xgboost-inference")
            .master("local[*]")
            .config("spark.jars.packages", ",".join(packages))
            .config("spark.driver.memory", spark_cfg.get('driver_memory', '4g'))
            .config("spark.sql.shuffle.partitions", str(spark_cfg.get('shuffle_partitions', 10)))
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
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")

    def load_model(self):
        model_path = os.path.join(self.config['model_dir'], 'xgboost_pandas_fraud_model.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.logger.info(f"Loading XGBoost model from: {model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.logger.info("✅ XGBoost model loaded")

    def fetch_transaction_features(self, trans_ids: List[str]) -> pd.DataFrame:
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        trans_id_list = ",".join([f"'{tid}'" for tid in trans_ids])
        query = f"""
            SELECT {', '.join(data_cfg['selected_features'])}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE trans_id IN ({trans_id_list})
                AND mbar_account_type_name = 'Customer Account'
        """
        self.logger.info(f"Fetching features for {len(trans_ids)} transaction(s)...")
        df_spark = self.spark.sql(query)
        count = df_spark.count()
        if count == 0:
            self.logger.warning(f"⚠️  No transactions found for the given IDs")
            return None
        self.logger.info(f"✅ Found {count} transaction(s)")
        df_pd = df_spark.select(data_cfg['selected_features']).toPandas()
        return df_pd

    def preprocess_features(self, df_pd: pd.DataFrame) -> pd.DataFrame:
        excluded = [self.config['data']['target_column'], 'cutoff_date', 'trans_id']
        feature_cols = [f for f in self.config['data']['selected_features'] if f not in excluded]
        for col in ['trx_channel', 'trx_type']:
            if col in df_pd.columns:
                df_pd[col] = df_pd[col].astype('category').cat.codes
        return df_pd, feature_cols

    def predict(self, trans_ids: List[str]) -> pd.DataFrame:
        df_pd = self.fetch_transaction_features(trans_ids)
        if df_pd is None or df_pd.empty:
            self.logger.error("No data to predict on")
            return None
        df_pd, feature_cols = self.preprocess_features(df_pd)
        X = df_pd[feature_cols].values
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)
        df_pd['fraud_probability'] = y_prob
        df_pd['risk_score'] = (df_pd['fraud_probability'] * 100).round(2)
        def risk_level(score):
            if score >= 80:
                return "CRITICAL"
            elif score >= 60:
                return "HIGH"
            elif score >= 40:
                return "MEDIUM"
            elif score >= 20:
                return "LOW"
            else:
                return "MINIMAL"
        df_pd['risk_level'] = df_pd['risk_score'].apply(risk_level)
        df_pd['prediction'] = y_pred
        return df_pd

    def display_results(self, results: pd.DataFrame):
        if results is None or results.empty:
            self.logger.warning("No results to display")
            return
        print("\n" + "=" * 100)
        print(f"FRAUD DETECTION RESULTS - XGBOOST MODEL")
        print("=" * 100)
        display_df = results[[
            'trans_id', 'cutoff_date', 'trx_channel', 'trx_type', 'trx_amt',
            'risk_score', 'risk_level', 'prediction', 'fraud_flag'
        ]].copy()
        display_df.columns = [
            'Transaction ID', 'Date', 'Channel', 'Type', 'Amount',
            'Risk Score', 'Risk Level', 'Predicted', 'Actual'
        ]
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"{x:,.2f}")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        print("=" * 100)
        print("\n📊 SUMMARY:")
        print(f"   Total Transactions: {len(results)}")
        print(f"   Predicted Fraud: {(results['prediction'] == 1).sum()}")
        print(f"   Predicted Legitimate: {(results['prediction'] == 0).sum()}")
        print(f"   Average Risk Score: {results['risk_score'].mean():.2f}")
        print(f"   Max Risk Score: {results['risk_score'].max():.2f}")
        print(f"   Min Risk Score: {results['risk_score'].min():.2f}")
        if 'fraud_flag' in results.columns and results['fraud_flag'].notna().any():
            correct = (results['prediction'] == results['fraud_flag']).sum()
            total = len(results)
            accuracy = (correct / total) * 100
            print(f"\n   Prediction Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print("=" * 100 + "\n")

    def save_results(self, results: pd.DataFrame, output_path: Optional[str] = None):
        if results is None or results.empty:
            self.logger.warning("No results to save")
            return
        if output_path is None:
            output_dir = self.config.get('analysis_dir', './output')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(
                output_dir, 'inference_results_xgboost.csv'
            )
        results.to_csv(output_path, index=False)
        self.logger.info(f"💾 Results saved to: {output_path}")

    def run(self, trans_ids: List[str], save_output: bool = False):
        try:
            self.initialize_spark()
            self.load_model()
            results = self.predict(trans_ids)
            if results is not None:
                self.display_results(results)
                if save_output:
                    self.save_results(results)
            return results
        except Exception as e:
            self.logger.error(f"❌ Inference failed: {str(e)}", exc_info=True)
            return None
        finally:
            if self.spark:
                self.spark.stop()

def main():
    parser = argparse.ArgumentParser(
        description='Fraud Detection Inference (XGBoost) - Generate fraud scores for transaction IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single transaction
  python inference_xgboost.py --trans_id 123456789
  # Multiple transactions
  python inference_xgboost.py --trans_ids 123456 789012 345678
  # With custom config and save output
  python inference_xgboost.py --trans_id 123456 --config config.json --save
        """
    )
    parser.add_argument('--trans_id', type=str, help='Single transaction ID to score')
    parser.add_argument('--trans_ids', type=str, nargs='+', help='Multiple transaction IDs to score')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--save', action='store_true', help='Save results to CSV file')
    args = parser.parse_args()
    if not args.trans_id and not args.trans_ids:
        parser.error("Must provide either --trans_id or --trans_ids")
    if args.trans_id:
        trans_ids = [args.trans_id]
    else:
        trans_ids = args.trans_ids
    config = load_config(args.config)
    print("\n" + "=" * 100)
    print("FRAUD DETECTION INFERENCE ENGINE (XGBOOST)")
    print("=" * 100)
    print(f"Transaction IDs: {len(trans_ids)}")
    print("=" * 100 + "\n")
    inference_engine = XGBoostFraudInference(config)
    results = inference_engine.run(trans_ids, save_output=args.save)
    sys.exit(0 if results is not None else 1)

if __name__ == '__main__':
    main()
