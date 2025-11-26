#!/usr/bin/env python3
"""
Batch Fraud Detection Inference Script
========================================
Reads transaction features from CSV file and generates fraud scores.
Writes predictions back to CSV with risk scores and probabilities.

Usage:
    python batch_inference.py --input features.csv --output predictions.csv --model random_forest
    python batch_inference.py --input data.csv --model gbt --all-models

Author: AI Team
Date: November 2025
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List
import time

import pandas as pd
import numpy as np

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel

import logging


class BatchFraudInference:
    """Batch fraud detection inference engine."""
    
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
        self.logger = logging.getLogger('BatchFraudInference')
        
    def initialize_spark(self):
        """Initialize Spark session for local processing."""
        self.logger.info("Initializing Spark session...")
        
        try:
            existing = SparkSession.getActiveSession()
            if existing:
                existing.stop()
                self.logger.info("Stopped existing Spark session")
        except:
            pass
        
        spark_cfg = self.config.get('spark', {})
        
        self.spark = (SparkSession.builder
            .appName("fraud-batch-inference")
            .master("local[*]")  # Use all available cores
            .config("spark.driver.memory", spark_cfg.get('driver_memory', '4g'))
            .config("spark.sql.shuffle.partitions", str(spark_cfg.get('shuffle_partitions', 10)))
            .getOrCreate()
        )
        
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")
        
    def load_model(self):
        """Load trained pipeline model."""
        model_path = os.path.join(
            self.config['model_dir'], 
            f'{self.model_type}_pipeline_model_v2'
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"Loading {self.model_type.replace('_', ' ').title()} model...")
        self.model = PipelineModel.load(model_path)
        self.logger.info(f"✅ Model loaded from: {model_path}")
        
    def load_csv(self, input_path: str) -> DataFrame:
        """
        Load features from CSV file.
        
        Args:
            input_path: Path to input CSV file
            
        Returns:
            Spark DataFrame with features
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self.logger.info(f"Loading data from: {input_path}")
        
        # Read CSV with Spark
        df = self.spark.read.csv(
            input_path,
            header=True,
            inferSchema=True
        )
        cols_to_drop = [c for c in ['features', 'features_raw'] if c in df.columns]
        if cols_to_drop:
            df = df.drop(*cols_to_drop)

        count = df.count()
        self.logger.info(f"✅ Loaded {count:,} records from CSV")
        
        # Log columns
        self.logger.info(f"   Columns: {len(df.columns)}")
        
        return df
        
    def validate_features(self, df: DataFrame) -> bool:
        """
        Validate that required features are present in the dataframe.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if all required features present, False otherwise
        """
        data_cfg = self.config['data']
        required_features = data_cfg['selected_features']
        
        # Exclude metadata columns
        excluded = [data_cfg['target_column'], 'cutoff_date', 'mbar_account_type_name']
        feature_cols = [f for f in required_features if f not in excluded]
        
        # Also exclude trans_id if it's in the list
        if 'trans_id' in feature_cols:
            feature_cols.remove('trans_id')
        
        df_cols = set(df.columns)
        missing_cols = [col for col in feature_cols if col not in df_cols]
        
        if missing_cols:
            self.logger.error(f"❌ Missing required features: {missing_cols}")
            self.logger.info(f"Available columns: {list(df_cols)}")
            return False
        
        self.logger.info("✅ All required features present")
        return True
        
    def predict(self, df: DataFrame) -> pd.DataFrame:
        """
        Generate fraud predictions for input DataFrame.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Pandas DataFrame with predictions and probabilities
        """
        self.logger.info("Generating predictions...")
        start_time = time.time()
        
        # Make predictions
        predictions = self.model.transform(df)
                
        # Convert to pandas
        result_df = predictions.toPandas()
        
        duration = time.time() - start_time
        self.logger.info(f"✅ Predictions generated in {duration:.2f}s")
        
        # Extract fraud probability (probability of class 1)
        if 'probability' in result_df.columns:
            result_df['fraud_probability'] = result_df['probability'].apply(
                lambda x: float(x[1]) if hasattr(x, '__getitem__') and len(x) > 1 else 0.0
            )
            
            # Create risk score (0-100)
            result_df['risk_score'] = (result_df['fraud_probability'] * 100).round(2)
            
            
            # Drop probability vector for cleaner output
            result_df = result_df.drop(columns=['probability'])
        
       
        return result_df
        
    def save_results(self, results: pd.DataFrame, output_path: str, 
                     include_features: bool = True):
        """
        Save prediction results to CSV.
        
        Args:
            results: DataFrame with predictions
            output_path: Path to output CSV file
            include_features: Whether to include all input features in output
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Select columns for output
        if include_features:
            # Include all columns
            output_df = results
        else:
            # Only include key columns and predictions
            key_cols = ['trans_id', 'cutoff_date', 'trx_channel', 'trx_type', 'trx_amt']
            prediction_cols = ['predicted_fraud', 'fraud_probability', 'risk_score', 'risk_level']
            
            available_key_cols = [c for c in key_cols if c in results.columns]
            available_pred_cols = [c for c in prediction_cols if c in results.columns]
            
            # Add actual fraud flag if present
            if 'fraud_flag' in results.columns:
                available_key_cols.append('fraud_flag')
            
            output_df = results[available_key_cols + available_pred_cols]
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        self.logger.info(f"💾 Results saved to: {output_path}")
        self.logger.info(f"   Rows: {len(output_df):,}")
        self.logger.info(f"   Columns: {len(output_df.columns)}")
        
    def display_summary(self, results: pd.DataFrame):
        """Display summary statistics of predictions."""
        print("\n" + "=" * 100)
        print(f"PREDICTION SUMMARY - {self.model_type.upper().replace('_', ' ')} MODEL")
        print("=" * 100)
        
        total = len(results)
        
        if 'predicted_fraud' in results.columns:
            fraud_count = (results['predicted_fraud'] == 1).sum()
            legit_count = (results['predicted_fraud'] == 0).sum()
            
            print(f"\n📊 PREDICTIONS:")
            print(f"   Total Transactions: {total:,}")
            print(f"   Predicted Fraud: {fraud_count:,} ({fraud_count/total*100:.2f}%)")
            print(f"   Predicted Legitimate: {legit_count:,} ({legit_count/total*100:.2f}%)")
        
        if 'risk_score' in results.columns:
            print(f"\n📈 RISK SCORES:")
            print(f"   Average: {results['risk_score'].mean():.2f}")
            print(f"   Median: {results['risk_score'].median():.2f}")
            print(f"   Max: {results['risk_score'].max():.2f}")
            print(f"   Min: {results['risk_score'].min():.2f}")
            print(f"   Std Dev: {results['risk_score'].std():.2f}")
        
        if 'risk_level' in results.columns:
            print(f"\n🎯 RISK DISTRIBUTION:")
            risk_dist = results['risk_level'].value_counts().sort_index()
            for level, count in risk_dist.items():
                print(f"   {level}: {count:,} ({count/total*100:.2f}%)")
        
        # If actual labels available, show performance
        if 'fraud_flag' in results.columns and 'predicted_fraud' in results.columns:
            correct = (results['predicted_fraud'] == results['fraud_flag']).sum()
            accuracy = (correct / total) * 100
            
            tp = ((results['predicted_fraud'] == 1) & (results['fraud_flag'] == 1)).sum()
            fp = ((results['predicted_fraud'] == 1) & (results['fraud_flag'] == 0)).sum()
            tn = ((results['predicted_fraud'] == 0) & (results['fraud_flag'] == 0)).sum()
            fn = ((results['predicted_fraud'] == 0) & (results['fraud_flag'] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n✅ PERFORMANCE (Actual labels available):")
            print(f"   Accuracy: {accuracy:.2f}%")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"\n   Confusion Matrix:")
            print(f"   True Positives:  {tp:,}")
            print(f"   False Positives: {fp:,}")
            print(f"   True Negatives:  {tn:,}")
            print(f"   False Negatives: {fn:,}")
        
        print("=" * 100 + "\n")
        
    def run(self, input_path: str, output_path: str, include_features: bool = True):
        """
        Execute batch inference pipeline.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            include_features: Whether to include all features in output
        """
        try:
            # Initialize
            self.initialize_spark()
            self.load_model()
            
            # Load data
            df = self.load_csv(input_path)
            
            # Validate features
            if not self.validate_features(df):
                return False
            
            # Predict
            results = self.predict(df)
            
            # Display summary
            self.display_summary(results)
            
            # Save results
            self.save_results(results, output_path, include_features)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Batch inference failed: {str(e)}", exc_info=True)
            return False
            
        finally:
            if self.spark:
                self.spark.stop()


class MultiModelBatchInference:
    """Run inference with multiple models and compare results."""
    
    def __init__(self, config: Dict, model_types: List[str]):
        self.config = config
        self.model_types = model_types
        self.logger = logging.getLogger('MultiModelBatchInference')
        
    def run(self, input_path: str, output_dir: str, include_features: bool = False):
        """
        Run inference with all specified models.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save output files
            include_features: Whether to include all features in output
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        print("\n" + "=" * 100)
        print("MULTI-MODEL BATCH INFERENCE")
        print("=" * 100)
        print(f"Input: {input_path}")
        print(f"Models: {', '.join([m.replace('_', ' ').title() for m in self.model_types])}")
        print("=" * 100 + "\n")
        
        for model_type in self.model_types:
            print(f"\n{'=' * 100}")
            print(f"PROCESSING WITH {model_type.upper().replace('_', ' ')} MODEL")
            print("=" * 100 + "\n")
            
            output_path = os.path.join(output_dir, f'predictions_{model_type}.csv')
            
            inference = BatchFraudInference(self.config, model_type)
            success = inference.run(input_path, output_path, include_features)
            
            if success:
                results[model_type] = output_path
                self.logger.info(f"✅ {model_type} completed successfully")
            else:
                self.logger.error(f"❌ {model_type} failed")
        
        # Summary
        print("\n" + "=" * 100)
        print("MULTI-MODEL INFERENCE COMPLETE")
        print("=" * 100)
        print(f"\nProcessed {len(results)}/{len(self.model_types)} models successfully:")
        for model_type, path in results.items():
            print(f"   {model_type.replace('_', ' ').title()}: {path}")
        print("=" * 100 + "\n")
        
        return results


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration."""
    default_config = {
        "data": {
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
        "spark": {
            "driver_memory": "4g",
            "shuffle_partitions": 10
        },
        "model_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models"
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description='Batch Fraud Detection Inference - Score transactions from CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model inference
  python batch_inference.py --input features.csv --output predictions.csv --model random_forest
  
  # Include all features in output
  python batch_inference.py --input data.csv --output results.csv --model gbt --include-features
  
  # Run inference with all models
  python batch_inference.py --input features.csv --output-dir ./predictions --all-models
  
  # Multiple specific models
  python batch_inference.py --input data.csv --output-dir ./output --models random_forest gbt
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input CSV file with features')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to output CSV file (required for single model)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (required for multiple models)')
    parser.add_argument('--model', type=str, 
                       choices=['decision_tree', 'random_forest', 'logistic_regression', 'gbt'],
                       help='Single model type to use')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['decision_tree', 'random_forest', 'logistic_regression', 'gbt'],
                       help='Multiple model types to use')
    parser.add_argument('--all-models', action='store_true',
                       help='Run inference with all available models')
    parser.add_argument('--include-features', action='store_true',
                       help='Include all input features in output CSV')
    parser.add_argument('--config', type=str,
                       help='Path to config file (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.all_models:
        model_types = ['decision_tree', 'random_forest', 'logistic_regression', 'gbt']
        if not args.output_dir:
            parser.error("--output-dir is required when using --all-models")
    elif args.models:
        model_types = args.models
        if not args.output_dir:
            parser.error("--output-dir is required when using --models")
    elif args.model:
        model_types = [args.model]
        if not args.output:
            parser.error("--output is required when using --model")
    else:
        parser.error("Must specify either --model, --models, or --all-models")
    
    # Load config
    config = load_config(args.config)
    
    # Run inference
    if len(model_types) == 1:
        # Single model
        inference = BatchFraudInference(config, model_types[0])
        success = inference.run(args.input, args.output, args.include_features)
        sys.exit(0 if success else 1)
    else:
        # Multiple models
        multi_inference = MultiModelBatchInference(config, model_types)
        results = multi_inference.run(args.input, args.output_dir, args.include_features)
        sys.exit(0 if results else 1)


if __name__ == '__main__':
    main()
