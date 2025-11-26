#!/usr/bin/env python3
"""
Fraud Detection Inference Script
=================================
Loads trained ML model and generates fraud scores for specific transaction IDs.
Pulls features from ClickHouse for the given transaction ID(s).

Usage:
    python inference.py --trans_id <transaction_id> --model <model_type>
    python inference.py --trans_id 123456 --model random_forest
    python inference.py --trans_ids 123456 789012 345678 --model gbt

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

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel

import logging


class FraudInference:
    """Fraud detection inference engine."""
    
    def __init__(self, config: Dict, model_type: str = 'random_forest'):
        self.config = config
        self.model_type = model_type
        self.spark = None
        self.model = None
        self.logger = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger('FraudInference')
        
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
        
        spark_cfg = self.config.get('spark', {})
        
        self.spark = (SparkSession.builder
            .appName("fraud-detection-inference")
            .master("local[*]")  # Use local mode for inference
            .config("spark.jars.packages", ",".join(packages))
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "10")
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
        
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")
        
    def load_model(self):
        """Load trained pipeline model."""
        model_path = os.path.join(
            self.config['model_dir'], 
            f'{self.model_type}_pipeline_model_v2'
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"Loading model from: {model_path}")
        self.model = PipelineModel.load(model_path)
        self.logger.info(f"✅ {self.model_type.replace('_', ' ').title()} model loaded")
        
    def fetch_transaction_features(self, trans_ids: List[str]) -> DataFrame:
        """
        Fetch features for specific transaction IDs from ClickHouse.
        
        Args:
            trans_ids: List of transaction IDs
            
        Returns:
            DataFrame with features for the transactions
        """
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        
        # Create IN clause for SQL
        trans_id_list = ",".join([f"'{tid}'" for tid in trans_ids])
        
        # Query to fetch transaction features
        query = f"""
            SELECT {', '.join(data_cfg['selected_features'])}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE trans_id IN ({trans_id_list})
                AND mbar_account_type_name = 'Customer Account'
        """
        
        self.logger.info(f"Fetching features for {len(trans_ids)} transaction(s)...")
        df = self.spark.sql(query)
        
        count = df.count()
        if count == 0:
            self.logger.warning(f"⚠️  No transactions found for the given IDs")
            return None
        
        self.logger.info(f"✅ Found {count} transaction(s)")
        return df
        
    def predict(self, trans_ids: List[str]) -> pd.DataFrame:
        """
        Generate fraud predictions for transaction IDs.
        
        Args:
            trans_ids: List of transaction IDs
            
        Returns:
            DataFrame with predictions and probabilities
        """
        # Fetch features
        features_df = self.fetch_transaction_features(trans_ids)
        
        if features_df is None or features_df.count() == 0:
            self.logger.error("No data to predict on")
            return None
        
        # Make predictions
        self.logger.info("Generating predictions...")
        predictions = self.model.transform(features_df)
        
        # Select relevant columns for output
        output_cols = [
            'trans_id', 
            'cutoff_date',
            'trx_channel', 
            'trx_type',
            'trx_amt',
            'fraud_flag',  # Actual label if available
            'prediction',
            'probability'
        ]
        
        # Convert to pandas for easier display
        result_df = predictions.select(output_cols).toPandas()
        
        # Extract fraud probability (probability of class 1)
        result_df['fraud_probability'] = result_df['probability'].apply(
            lambda x: float(x[1]) if len(x) > 1 else 0.0
        )
        
        # Create risk score (0-100)
        result_df['risk_score'] = (result_df['fraud_probability'] * 100).round(2)
        
        # Determine risk level
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
        
        result_df['risk_level'] = result_df['risk_score'].apply(risk_level)
        
        # Drop probability vector column for cleaner output
        result_df = result_df.drop(columns=['probability'])
        
        return result_df
        
    def display_results(self, results: pd.DataFrame):
        """Display prediction results in a formatted table."""
        if results is None or results.empty:
            self.logger.warning("No results to display")
            return
        
        print("\n" + "=" * 100)
        print(f"FRAUD DETECTION RESULTS - {self.model_type.upper().replace('_', ' ')} MODEL")
        print("=" * 100)
        
        # Prepare display dataframe
        display_df = results[[
            'trans_id',
            'cutoff_date',
            'trx_channel',
            'trx_type',
            'trx_amt',
            'risk_score',
            'risk_level',
            'prediction',
            'fraud_flag'
        ]].copy()
        
        # Rename columns for display
        display_df.columns = [
            'Transaction ID',
            'Date',
            'Channel',
            'Type',
            'Amount',
            'Risk Score',
            'Risk Level',
            'Predicted',
            'Actual'
        ]
        
        # Format amount
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"{x:,.2f}")
        
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        print("=" * 100)
        
        # Summary statistics
        print("\n📊 SUMMARY:")
        print(f"   Total Transactions: {len(results)}")
        print(f"   Predicted Fraud: {(results['prediction'] == 1).sum()}")
        print(f"   Predicted Legitimate: {(results['prediction'] == 0).sum()}")
        print(f"   Average Risk Score: {results['risk_score'].mean():.2f}")
        print(f"   Max Risk Score: {results['risk_score'].max():.2f}")
        print(f"   Min Risk Score: {results['risk_score'].min():.2f}")
        
        # If actual labels available, show accuracy
        if 'fraud_flag' in results.columns and results['fraud_flag'].notna().any():
            correct = (results['prediction'] == results['fraud_flag']).sum()
            total = len(results)
            accuracy = (correct / total) * 100
            print(f"\n   Prediction Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        print("=" * 100 + "\n")
        
    def save_results(self, results: pd.DataFrame, output_path: Optional[str] = None):
        """Save prediction results to CSV."""
        if results is None or results.empty:
            self.logger.warning("No results to save")
            return
        
        if output_path is None:
            output_dir = self.config.get('analysis_dir', './output')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(
                output_dir, 
                f'inference_results_{self.model_type}.csv'
            )
        
        results.to_csv(output_path, index=False)
        self.logger.info(f"💾 Results saved to: {output_path}")
        
    def run(self, trans_ids: List[str], save_output: bool = False):
        """Execute inference pipeline."""
        try:
            # Initialize
            self.initialize_spark()
            self.load_model()
            
            # Predict
            results = self.predict(trans_ids)
            
            if results is not None:
                # Display
                self.display_results(results)
                
                # Save if requested
                if save_output:
                    self.save_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Inference failed: {str(e)}", exc_info=True)
            return None
            
        finally:
            if self.spark:
                self.spark.stop()


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration."""
    default_config = {
        "data": {
            "table_name": "stixor_fraud_features_distributed",
            "target_column": "fraud_flag",
            "selected_features": [
                'trans_id', 'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 
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
            "driver_memory": "4g",
            "shuffle_partitions": 10
        },
        "model_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models",
        "analysis_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis"
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description='Fraud Detection Inference - Generate fraud scores for transaction IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single transaction
  python inference.py --trans_id 123456789 --model random_forest
  
  # Multiple transactions
  python inference.py --trans_ids 123456 789012 345678 --model gbt
  
  # With custom config and save output
  python inference.py --trans_id 123456 --model decision_tree --config config.json --save
        """
    )
    
    parser.add_argument('--trans_id', type=str, 
                       help='Single transaction ID to score')
    parser.add_argument('--trans_ids', type=str, nargs='+',
                       help='Multiple transaction IDs to score')
    parser.add_argument('--model', type=str, 
                       choices=['decision_tree', 'random_forest', 'logistic_regression', 'gbt'],
                       default='random_forest',
                       help='Model type to use for inference (default: random_forest)')
    parser.add_argument('--config', type=str, 
                       help='Path to config file (optional)')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.trans_id and not args.trans_ids:
        parser.error("Must provide either --trans_id or --trans_ids")
    
    # Collect transaction IDs
    if args.trans_id:
        trans_ids = [args.trans_id]
    else:
        trans_ids = args.trans_ids
    
    # Load config
    config = load_config(args.config)
    
    # Run inference
    print("\n" + "=" * 100)
    print("FRAUD DETECTION INFERENCE ENGINE")
    print("=" * 100)
    print(f"Model: {args.model.replace('_', ' ').title()}")
    print(f"Transaction IDs: {len(trans_ids)}")
    print("=" * 100 + "\n")
    
    inference_engine = FraudInference(config, model_type=args.model)
    results = inference_engine.run(trans_ids, save_output=args.save)
    
    sys.exit(0 if results is not None else 1)


if __name__ == '__main__':
    main()
