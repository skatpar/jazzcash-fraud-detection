#!/usr/bin/env python3
"""
Fraud Detection Inference Script
==================================
Load trained model and generate predictions on test data.

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

import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel

import logging


class FraudInference:
    """Fraud detection inference pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spark = None
        self.logger = None
        self.pipeline_model = None
        
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary directories."""
        output_dir = self.config.get('output_dir', './output')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_filename = f"fraud_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_dir = self.config.get('log_dir', './logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('FraudInference')
        self.logger.info("=" * 80)
        self.logger.info("FRAUD DETECTION INFERENCE")
        self.logger.info("=" * 80)
        self.logger.info(f"Log file: {log_path}")
    
    def initialize_spark(self):
        """Initialize Spark with ClickHouse catalog."""
        self.logger.info("Initializing Spark session...")
        
        # Set Python environment variables to ensure version consistency
        # between driver and workers (both must use Python 3.10)
        python_path = sys.executable  # Use the same Python that's running this script
        os.environ['PYSPARK_PYTHON'] = python_path
        os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
        
        self.logger.info(f"Setting PYSPARK_PYTHON={python_path}")
        self.logger.info(f"Setting PYSPARK_DRIVER_PYTHON={python_path}")
        
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
        
        self.spark = (SparkSession.builder
            .appName("fraud-detection-inference")
            .master("spark://10.205.161.118:7077")
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
        self.logger.info(f"   ClickHouse catalog: clickhouse.{ch['database']}")
    
    def load_model(self, model_path: str):
        """Load the trained pipeline model."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING MODEL")
        self.logger.info("=" * 80)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.logger.info(f"Loading model from: {model_path}")
        start = time.time()
        
        self.pipeline_model = PipelineModel.load(model_path)
        
        duration = time.time() - start
        self.logger.info(f"✅ Model loaded in {duration:.2f}s")
        self.logger.info(f"   Pipeline stages: {len(self.pipeline_model.stages)}")
        
        return self.pipeline_model
    
    def load_test_data(self, start_date: str, end_date: str) -> DataFrame:
        """Load test data from ClickHouse."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING TEST DATA")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        
        # Build query - include all features needed by the model plus trans_id
        # Ensure trans_id is included for output tracking
        features_to_select = data_cfg['selected_features'].copy() if isinstance(data_cfg['selected_features'], list) else list(data_cfg['selected_features'])
        if 'trans_id' not in features_to_select:
            features_to_select.insert(0, 'trans_id')
        
        query = f"""
            SELECT {', '.join(features_to_select)}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                AND mbar_account_type_name = 'Customer Account'
        """
        
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Table: {ch_cfg['database']}.{data_cfg['table_name']}")
        
        start = time.time()
        df = self.spark.sql(query)
        count = df.count()
        duration = time.time() - start
        
        self.logger.info(f"✅ Loaded {count:,} rows in {duration:.2f}s")
        
        # Log class distribution if fraud_flag exists
        if 'fraud_flag' in df.columns:
            fraud_dist = df.groupBy('fraud_flag').count().orderBy('fraud_flag').collect()
            self.logger.info("Class distribution:")
            for row in fraud_dist:
                pct = (row['count'] / count) * 100
                self.logger.info(f"   Class {row['fraud_flag']}: {row['count']:,} ({pct:.2f}%)")
        
        return df
    
    def generate_predictions(self, df: DataFrame) -> DataFrame:
        """Generate predictions using the loaded model."""
        self.logger.info("=" * 80)
        self.logger.info("GENERATING PREDICTIONS")
        self.logger.info("=" * 80)
        
        self.logger.info("Running inference...")
        start = time.time()
        
        predictions = self.pipeline_model.transform(df)
        
        duration = time.time() - start
        count = predictions.count()
        
        self.logger.info(f"✅ Generated {count:,} predictions in {duration:.2f}s")
        
        # Log prediction distribution
        pred_dist = predictions.groupBy('prediction').count().orderBy('prediction').collect()
        self.logger.info("Prediction distribution:")
        for row in pred_dist:
            pct = (row['count'] / count) * 100
            pred_label = "Non-Fraud" if row['prediction'] == 0.0 else "Fraud"
            self.logger.info(f"   {pred_label} ({row['prediction']}): {row['count']:,} ({pct:.2f}%)")
        
        # Calculate metrics if we have true labels
        if 'fraud_flag' in predictions.columns:
            from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
            
            binary_eval = BinaryClassificationEvaluator(labelCol='fraud_flag', metricName="areaUnderROC")
            multiclass_eval = MulticlassClassificationEvaluator(labelCol='fraud_flag', predictionCol="prediction")
            
            auc = binary_eval.evaluate(predictions)
            accuracy = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "accuracy"})
            precision = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "weightedPrecision"})
            recall = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "weightedRecall"})
            f1 = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "f1"})
            
            self.logger.info("=" * 80)
            self.logger.info("EVALUATION METRICS")
            self.logger.info("=" * 80)
            self.logger.info(f"   AUC-ROC: {auc:.4f}")
            self.logger.info(f"   Accuracy: {accuracy:.4f}")
            self.logger.info(f"   Precision: {precision:.4f}")
            self.logger.info(f"   Recall: {recall:.4f}")
            self.logger.info(f"   F1-Score: {f1:.4f}")
        
        return predictions
    
    def save_predictions(self, predictions: DataFrame, output_path: str):
        """Save predictions to Parquet file."""
        self.logger.info("=" * 80)
        self.logger.info("SAVING PREDICTIONS")
        self.logger.info("=" * 80)
        
        # Select relevant columns for output - prioritize key identifier and metadata
        output_cols = []
        
        # Add trans_id first (primary identifier)
        if 'trans_id' in predictions.columns:
            output_cols.append('trans_id')
        
        # Add date
        if 'cutoff_date' in predictions.columns:
            output_cols.append('cutoff_date')
        
        # Add trx_channel
        if 'trx_channel' in predictions.columns:
            output_cols.append('trx_channel')
        
        # Add prediction columns
        output_cols.append('prediction')
        
        # Add probability columns
        if 'probability' in predictions.columns:
            output_cols.append('probability')
        
        # Add true label if available
        if 'fraud_flag' in predictions.columns:
            output_cols.append('fraud_flag')
        
        # Add additional key features for analysis
        # feature_cols = ['trx_type', 'trx_amt', 'start_balance', 
        #                'hour_of_day', 'is_night', 'is_weekend']
        # for col_name in feature_cols:
        #     if col_name in predictions.columns and col_name not in output_cols:
        #         output_cols.append(col_name)
        
        self.logger.info(f"Saving {len(output_cols)} columns to Parquet...")
        self.logger.info(f"   Columns: {', '.join(output_cols[:10])}{'...' if len(output_cols) > 10 else ''}")
        start = time.time()
        
        # Select columns
        predictions_selected = predictions.select(output_cols)
        
        # Extract probability scores as separate columns using Spark SQL
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        from pyspark.ml.linalg import VectorUDT, DenseVector, SparseVector
        
        if 'probability' in predictions_selected.columns:
            # Create UDFs to extract probabilities
            @udf(returnType=DoubleType())
            def get_fraud_prob(probability):
                if probability is not None:
                    if isinstance(probability, (DenseVector, SparseVector)):
                        return float(probability[1]) if len(probability) > 1 else 0.0
                    elif hasattr(probability, '__getitem__'):
                        return float(probability[1]) if len(probability) > 1 else 0.0
                return 0.0
            
            @udf(returnType=DoubleType())
            def get_non_fraud_prob(probability):
                if probability is not None:
                    if isinstance(probability, (DenseVector, SparseVector)):
                        return float(probability[0]) if len(probability) > 0 else 0.0
                    elif hasattr(probability, '__getitem__'):
                        return float(probability[0]) if len(probability) > 0 else 0.0
                return 0.0
            
            predictions_selected = predictions_selected.withColumn('fraud_probability', get_fraud_prob(col('probability')))
            predictions_selected = predictions_selected.drop('probability')
        
        # Save as Parquet
        predictions_selected.write.mode('overwrite').parquet(output_path)
        
        duration = time.time() - start
        
        row_count = predictions_selected.count()
        
        self.logger.info(f"✅ Saved {row_count:,} predictions in {duration:.2f}s")
        self.logger.info(f"   File: {output_path}")
        
        return predictions_selected
    
    def run(self, model_path: str, start_date: str, end_date: str, output_path: Optional[str] = None):
        """Execute the complete inference pipeline."""
        try:
            pipeline_start = time.time()
            
            # Initialize Spark
            self.initialize_spark()
            
            # Load model
            self.load_model(model_path)
            
            # Load test data
            test_df = self.load_test_data(start_date, end_date)
            
            # Generate predictions
            predictions = self.generate_predictions(test_df)
            
            # Save predictions
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = self.config.get('output_dir', './output')
                output_path = os.path.join(output_dir, f'fraud_predictions_{start_date}_to_{end_date}_{timestamp}.parquet')
            
            predictions_df = self.save_predictions(predictions, output_path)
            
            # Summary
            total_time = time.time() - pipeline_start
            pred_count = predictions_df.count()
            self.logger.info("=" * 80)
            self.logger.info("INFERENCE COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Test period: {start_date} to {end_date}")
            self.logger.info(f"Predictions: {pred_count:,}")
            self.logger.info(f"Output: {output_path}")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Inference failed: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()
                self.logger.info("Spark session stopped")


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file or use defaults."""
    default_config = {
        "data": {
            "table_name": "stixor_fraud_features_distributed",
            "target_column": "fraud_flag",
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
        "model_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models",
        "output_dir": "/root/research-dir/dev/jazzcash-fraud-detection/output",
        "log_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models/logs"
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configs (user config takes precedence)
        for key in user_config:
            if isinstance(user_config[key], dict) and key in default_config:
                default_config[key].update(user_config[key])
            else:
                default_config[key] = user_config[key]
    
    return default_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate fraud predictions on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model and config
  python fraud_detection_inference.py --start-date 2025-07-01 --end-date 2025-07-30
  
  # Specify custom model path
  python fraud_detection_inference.py --model-path /path/to/model --start-date 2025-07-01 --end-date 2025-07-30
  
  # Use custom config file
  python fraud_detection_inference.py --config pipeline_config.json --start-date 2025-07-01 --end-date 2025-07-30
  
  # Specify output file
  python fraud_detection_inference.py --start-date 2025-07-01 --end-date 2025-07-30 --output predictions.parquet
        """
    )
    
    parser.add_argument('--config', type=str, 
                       help='Path to configuration JSON file')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained pipeline model directory')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date for test data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date for test data (YYYY-MM-DD)')
    parser.add_argument('--output', type=str,
                       help='Output Parquet file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Use default decision tree model
        model_path = os.path.join(config['model_dir'], 'decision_tree_pipeline_model')
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        sys.exit(1)
    
    # Run inference
    inference = FraudInference(config)
    success = inference.run(
        model_path=model_path,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
