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
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as SklearnLR

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel

import logging


def calibrate_probabilities_isotonic(
    probabilities: np.ndarray, 
    true_labels: np.ndarray,
    method: str = 'isotonic'
) -> Tuple[np.ndarray, object]:
    """
    Calibrate probabilities using Isotonic Regression or Platt Scaling.
    
    Args:
        probabilities: Raw model probabilities (fraud probability)
        true_labels: True binary labels (0 or 1)
        method: 'isotonic' for Isotonic Regression, 'platt' for Platt Scaling
        
    Returns:
        Tuple of (calibrated_probabilities, fitted_calibrator)
    """
    if method == 'isotonic':
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        calibrator.fit(probabilities, true_labels)
        calibrated_probs = calibrator.predict(probabilities)
    elif method == 'platt':
        # Platt scaling uses logistic regression on the log-odds
        calibrator = SklearnLR(solver='lbfgs', max_iter=1000)
        calibrator.fit(probabilities.reshape(-1, 1), true_labels)
        calibrated_probs = calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}. Use 'isotonic' or 'platt'")
    
    return calibrated_probs, calibrator


def apply_calibration(probabilities: np.ndarray, calibrator: object, method: str = 'isotonic') -> np.ndarray:
    """
    Apply a pre-fitted calibrator to new probabilities.
    
    Args:
        probabilities: Raw model probabilities
        calibrator: Fitted calibration model
        method: 'isotonic' or 'platt'
        
    Returns:
        Calibrated probabilities
    """
    if method == 'isotonic':
        return calibrator.predict(probabilities)
    elif method == 'platt':
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")


class FraudInference:
    """Fraud detection inference engine."""
    
    def __init__(self, config: Dict, model_type: str = 'random_forest'):
        self.config = config
        self.model_type = model_type
        self.spark = None
        self.model = None
        self.logger = None
        self.calibrator = None
        self.calibration_method = 'isotonic'
        
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
            f'{self.model_type}_pipeline_model_fraud_scenario_v7'
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
    
    def calibrate_predictions(self, results: pd.DataFrame, method: str = 'isotonic') -> pd.DataFrame:
        """
        Calibrate the prediction probabilities.
        
        Args:
            results: DataFrame with fraud_probability and fraud_flag columns
            method: 'isotonic' or 'platt'
            
        Returns:
            DataFrame with calibrated_probability column added
        """
        self.logger.info(f"Calibrating probabilities using {method} method...")
        
        if 'fraud_flag' not in results.columns or results['fraud_flag'].isna().all():
            self.logger.warning("No true labels available for calibration. Skipping calibration.")
            results['calibrated_probability'] = results['fraud_probability']
            return results
        
        # Get raw probabilities and true labels
        raw_probs = results['fraud_probability'].values
        true_labels = results['fraud_flag'].values
        
        # Calibrate
        calibrated_probs, self.calibrator = calibrate_probabilities_isotonic(
            raw_probs, true_labels, method=method
        )
        self.calibration_method = method
        
        results['calibrated_probability'] = calibrated_probs
        results['calibrated_risk_score'] = (results['calibrated_probability'] * 100).round(2)
        
        self.logger.info(f"✅ Probability calibration complete")
        self.logger.info(f"   Raw probability range: [{raw_probs.min():.4f}, {raw_probs.max():.4f}]")
        self.logger.info(f"   Calibrated probability range: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]")
        
        return results
    
    def extract_false_positives(self, results: pd.DataFrame, max_count: int = 10000) -> pd.DataFrame:
        """
        Extract false positives from predictions.
        
        False positives: Model predicted fraud (prediction=1) but actual was non-fraud (fraud_flag=0)
        
        Args:
            results: DataFrame with predictions
            max_count: Maximum number of false positives to return
            
        Returns:
            DataFrame containing false positive cases
        """
        self.logger.info("Extracting false positives...")
        
        # Filter for false positives
        fp_df = results[
            (results['prediction'] == 1) & (results['fraud_flag'] == 0)
        ].copy()
        
        total_fp = len(fp_df)
        self.logger.info(f"   Total false positives: {total_fp:,}")
        
        # Limit to max_count
        if total_fp > max_count:
            fp_df = fp_df.head(max_count)
            self.logger.info(f"   Limited to top {max_count:,} false positives")
        
        return fp_df
    
    def save_false_positives(self, fp_df: pd.DataFrame, output_dir: Optional[str] = None) -> str:
        """
        Save false positives to a CSV file with timestamp.
        
        Args:
            fp_df: DataFrame containing false positive cases
            output_dir: Optional output directory path
            
        Returns:
            Path to saved file
        """
        if fp_df is None or fp_df.empty:
            self.logger.warning("No false positives to save")
            return None
        
        if output_dir is None:
            output_dir = self.config.get('analysis_dir', './output')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'false_positives_{self.model_type}_{timestamp}.csv'
        output_path = os.path.join(output_dir, filename)
        
        fp_df.to_csv(output_path, index=False)
        self.logger.info(f"💾 False positives saved to: {output_path}")
        self.logger.info(f"   Records saved: {len(fp_df):,}")
        
        return output_path
    
    def plot_probability_distributions(
        self, 
        results: pd.DataFrame, 
        output_dir: Optional[str] = None
    ) -> str:
        """
        Plot and save probability distribution histograms.
        
        Creates plots for:
        1. Raw probability distribution
        2. Calibrated probability distribution (if available)
        3. Comparison by actual label
        
        Args:
            results: DataFrame with fraud_probability and optionally calibrated_probability
            output_dir: Optional output directory path
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating probability distribution plots...")
        
        if output_dir is None:
            output_dir = self.config.get('analysis_dir', './output')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        has_calibrated = 'calibrated_probability' in results.columns
        
        # Create figure
        if has_calibrated:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes = axes.reshape(1, 2)
        
        # Plot 1: Raw probability distribution
        ax1 = axes[0, 0] if has_calibrated else axes[0, 0]
        sns.histplot(data=results, x='fraud_probability', bins=50, kde=True, ax=ax1, color='steelblue')
        ax1.set_title(f'Raw Probability Distribution\n({self.model_type.replace("_", " ").title()})', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Fraud Probability')
        ax1.set_ylabel('Count')
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        ax1.legend()
        
        # Plot 2: Raw probability by actual label
        ax2 = axes[0, 1] if has_calibrated else axes[0, 1]
        if 'fraud_flag' in results.columns and results['fraud_flag'].notna().any():
            for label, color, name in [(0, 'green', 'Non-Fraud'), (1, 'red', 'Fraud')]:
                subset = results[results['fraud_flag'] == label]
                if len(subset) > 0:
                    sns.histplot(data=subset, x='fraud_probability', bins=50, kde=True, 
                               ax=ax2, color=color, alpha=0.5, label=name)
            ax2.set_title('Raw Probability by Actual Label', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Fraud Probability')
            ax2.set_ylabel('Count')
            ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No actual labels available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Raw Probability by Actual Label', fontsize=12, fontweight='bold')
        
        if has_calibrated:
            # Plot 3: Calibrated probability distribution
            ax3 = axes[1, 0]
            sns.histplot(data=results, x='calibrated_probability', bins=50, kde=True, ax=ax3, color='darkorange')
            ax3.set_title(f'Calibrated Probability Distribution\n(Method: {self.calibration_method.title()})', 
                         fontsize=12, fontweight='bold')
            ax3.set_xlabel('Calibrated Fraud Probability')
            ax3.set_ylabel('Count')
            ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
            ax3.legend()
            
            # Plot 4: Calibrated probability by actual label
            ax4 = axes[1, 1]
            if 'fraud_flag' in results.columns and results['fraud_flag'].notna().any():
                for label, color, name in [(0, 'green', 'Non-Fraud'), (1, 'red', 'Fraud')]:
                    subset = results[results['fraud_flag'] == label]
                    if len(subset) > 0:
                        sns.histplot(data=subset, x='calibrated_probability', bins=50, kde=True, 
                                   ax=ax4, color=color, alpha=0.5, label=name)
                ax4.set_title('Calibrated Probability by Actual Label', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Calibrated Fraud Probability')
                ax4.set_ylabel('Count')
                ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'No actual labels available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Calibrated Probability by Actual Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'probability_distribution_{self.model_type}_{timestamp}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"💾 Probability distribution plot saved to: {plot_path}")
        
        # Also save probability statistics
        stats = {
            'model': self.model_type,
            'timestamp': timestamp,
            'raw_probability': {
                'mean': float(results['fraud_probability'].mean()),
                'std': float(results['fraud_probability'].std()),
                'min': float(results['fraud_probability'].min()),
                'max': float(results['fraud_probability'].max()),
                'median': float(results['fraud_probability'].median()),
                'percentiles': {
                    '25%': float(results['fraud_probability'].quantile(0.25)),
                    '50%': float(results['fraud_probability'].quantile(0.50)),
                    '75%': float(results['fraud_probability'].quantile(0.75)),
                    '90%': float(results['fraud_probability'].quantile(0.90)),
                    '95%': float(results['fraud_probability'].quantile(0.95)),
                    '99%': float(results['fraud_probability'].quantile(0.99))
                }
            }
        }
        
        if has_calibrated:
            stats['calibrated_probability'] = {
                'method': self.calibration_method,
                'mean': float(results['calibrated_probability'].mean()),
                'std': float(results['calibrated_probability'].std()),
                'min': float(results['calibrated_probability'].min()),
                'max': float(results['calibrated_probability'].max()),
                'median': float(results['calibrated_probability'].median()),
                'percentiles': {
                    '25%': float(results['calibrated_probability'].quantile(0.25)),
                    '50%': float(results['calibrated_probability'].quantile(0.50)),
                    '75%': float(results['calibrated_probability'].quantile(0.75)),
                    '90%': float(results['calibrated_probability'].quantile(0.90)),
                    '95%': float(results['calibrated_probability'].quantile(0.95)),
                    '99%': float(results['calibrated_probability'].quantile(0.99))
                }
            }
        
        stats_path = os.path.join(output_dir, f'probability_stats_{self.model_type}_{timestamp}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"💾 Probability statistics saved to: {stats_path}")
        
        return plot_path
        
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
        
    def run(self, trans_ids: List[str], save_output: bool = False, 
            calibrate: bool = False, calibration_method: str = 'isotonic',
            save_false_positives: bool = False, max_fp: int = 10000,
            plot_distributions: bool = False):
        """
        Execute inference pipeline.
        
        Args:
            trans_ids: List of transaction IDs to score
            save_output: Whether to save all results to CSV
            calibrate: Whether to calibrate probabilities
            calibration_method: 'isotonic' or 'platt'
            save_false_positives: Whether to save false positives to file
            max_fp: Maximum number of false positives to save
            plot_distributions: Whether to plot probability distributions
        """
        try:
            # Initialize
            self.initialize_spark()
            self.load_model()
            
            # Predict
            results = self.predict(trans_ids)
            
            if results is not None:
                # Calibrate probabilities if requested
                if calibrate:
                    results = self.calibrate_predictions(results, method=calibration_method)
                
                # Display
                self.display_results(results)
                
                # Save all results if requested
                if save_output:
                    self.save_results(results)
                
                # Extract and save false positives if requested
                if save_false_positives:
                    fp_df = self.extract_false_positives(results, max_count=max_fp)
                    if len(fp_df) > 0:
                        self.save_false_positives(fp_df)
                
                # Plot probability distributions if requested
                if plot_distributions:
                    self.plot_probability_distributions(results)
            
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
  
  # With probability calibration and false positive extraction
  python inference.py --trans_ids 123456 789012 --model gbt --calibrate --save-fp --plot
  
  # Full analysis with all options
  python inference.py --trans_ids 123456 789012 --model gbt --save --calibrate --calibration-method isotonic --save-fp --max-fp 5000 --plot
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
    parser.add_argument('--calibrate', action='store_true',
                       help='Calibrate probabilities using isotonic regression or Platt scaling')
    parser.add_argument('--calibration-method', type=str, 
                       choices=['isotonic', 'platt'],
                       default='isotonic',
                       help='Calibration method (default: isotonic)')
    parser.add_argument('--save-fp', action='store_true',
                       help='Save false positives to CSV file with timestamp')
    parser.add_argument('--max-fp', type=int, default=10000,
                       help='Maximum number of false positives to save (default: 10000)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot probability distributions')
    
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
    print(f"Calibrate: {args.calibrate} ({args.calibration_method if args.calibrate else 'N/A'})")
    print(f"Save False Positives: {args.save_fp} (max: {args.max_fp})")
    print(f"Plot Distributions: {args.plot}")
    print("=" * 100 + "\n")
    
    inference_engine = FraudInference(config, model_type=args.model)
    results = inference_engine.run(
        trans_ids, 
        save_output=args.save,
        calibrate=args.calibrate,
        calibration_method=args.calibration_method,
        save_false_positives=args.save_fp,
        max_fp=args.max_fp,
        plot_distributions=args.plot
    )
    
    sys.exit(0 if results is not None else 1)


if __name__ == '__main__':
    main()
