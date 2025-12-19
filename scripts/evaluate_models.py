#!/usr/bin/env python3
"""
Model Evaluation and Confusion Matrix Generator
================================================
Loads trained models, generates predictions on evaluation data,
and creates confusion matrices for each model.

Evaluation Period: 2025-07-01 to 2025-07-31

Author: AI Team
Date: November 2025
"""

import os
import sys
import json
import argparse

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report

import logging


class ModelEvaluator:
    """Load trained models and generate predictions with confusion matrices."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spark = None
        self.logger = None
        
        self._setup_directories()
        self._setup_logging()
    
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config['analysis_dir'], self.config['log_dir'], 
                         self.config['predictions_dir']]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_filename = f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.config['log_dir'], log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('ModelEvaluator')
        self.logger.info("=" * 80)
        self.logger.info("MODEL EVALUATION AND CONFUSION MATRIX GENERATOR")
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
        
        spark_cfg = self.config.get('spark', {})
        
        self.spark = (SparkSession.builder
            .appName("fraud-detection-evaluation")
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
        
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")
        self.logger.info(f"   Master: {self.spark.sparkContext.master}")
    
    def load_evaluation_data(self, start_date: str, end_date: str) -> DataFrame:
        """Load evaluation data from ClickHouse."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING EVALUATION DATA")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        
        query = f"""
            SELECT {', '.join(data_cfg['selected_features'])}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                AND mbar_account_type_name = 'Customer Account'
                    AND trx_channel='NEW_JC_APP'
                    AND trx_type='Transfer(C2C)'
                    AND ac_to IS NOT NULL 
                    AND start_balance<>end_balance
                    AND ac_to<>''


        """
        
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        df = self.spark.sql(query)
        count = df.count()
        
        self.logger.info(f"✅ Loaded {count:,} rows")
        
        # Log class distribution
        fraud_dist = df.groupBy('fraud_flag').count().orderBy('fraud_flag').collect()
        for row in fraud_dist:
            pct = (row['count'] / count) * 100
            self.logger.info(f"   Class {row['fraud_flag']}: {row['count']:,} ({pct:.2f}%)")
        
        return df
    
    def load_model(self, model_name: str) -> Optional[PipelineModel]:
        """Load a trained model."""
        model_path = os.path.join(self.config['model_dir'], f'{model_name}_pipeline_model_v2')
        
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Model not found: {model_path}")
            return None
        
        try:
            model = PipelineModel.load(model_path)
            self.logger.info(f"✅ Loaded model: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load {model_name}: {str(e)}")
            return None
    
    def generate_predictions(self, model: PipelineModel, df: DataFrame, model_name: str) -> DataFrame:
        """Generate predictions using the model."""
        self.logger.info(f"Generating predictions for {model_name}...")
        predictions = model.transform(df)
        
        # # Save predictions to CSV (sample)
        # pred_sample_path = os.path.join(
        #     self.config['predictions_dir'], 
        #     f'{model_name}_predictions_sample.csv'
        # )
        
        # # Convert to pandas and save a sample
        # pred_cols = ['fraud_flag', 'prediction', 'probability']
        # pred_sample = predictions.select(pred_cols).limit(10000).toPandas()
        # pred_sample.to_csv(pred_sample_path, index=False)
        
        # self.logger.info(f"💾 Saved prediction sample to: {pred_sample_path}")
        
        return predictions
    
    def evaluate_model(self, predictions: DataFrame, model_name: str) -> Dict:
        """Evaluate model performance."""
        self.logger.info("=" * 80)
        self.logger.info(f"EVALUATING {model_name.upper().replace('_', ' ')}")
        self.logger.info("=" * 80)
        
        target_col = self.config['data']['target_column']
        
        # Binary classification metrics
        binary_eval = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
        auc = binary_eval.evaluate(predictions)
        
        # Multiclass metrics
        multiclass_eval = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction")
        accuracy = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "accuracy"})
        precision = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "weightedPrecision"})
        recall = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "weightedRecall"})
        f1 = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "f1"})
        
        self.logger.info(f"📊 Performance Metrics:")
        self.logger.info(f"   AUC-ROC: {auc:.4f}")
        self.logger.info(f"   Accuracy: {accuracy:.4f}")
        self.logger.info(f"   Precision: {precision:.4f}")
        self.logger.info(f"   Recall: {recall:.4f}")
        self.logger.info(f"   F1-Score: {f1:.4f}")
        
        return {
            'model': model_name,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def create_confusion_matrix(self, predictions: DataFrame, model_name: str):
        """Create and save confusion matrix."""
        self.logger.info(f"Creating confusion matrix for {model_name}...")
        
        # Convert to pandas for sklearn
        pred_pdf = predictions.select('fraud_flag', 'prediction').toPandas()
        
        y_true = pred_pdf['fraud_flag'].values
        y_pred = pred_pdf['prediction'].values
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'],
                   cbar_kws={'label': 'Count'})
        axes[0].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix (Counts)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Actual', fontsize=12)
        axes[0].set_xlabel('Predicted', fontsize=12)
        
        # Plot 2: Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'],
                   cbar_kws={'label': 'Percentage (%)'})
        axes[1].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix (Percentages)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Actual', fontsize=12)
        axes[1].set_xlabel('Predicted', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        cm_path = os.path.join(self.config['analysis_dir'], f'{model_name}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"💾 Confusion matrix saved to: {cm_path}")
        
        # Save confusion matrix data to CSV
        cm_df = pd.DataFrame(cm, 
                            columns=['Predicted_Non-Fraud', 'Predicted_Fraud'],
                            index=['Actual_Non-Fraud', 'Actual_Fraud'])
        cm_csv_path = os.path.join(self.config['analysis_dir'], f'{model_name}_confusion_matrix.csv')
        cm_df.to_csv(cm_csv_path)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, 
                                       target_names=['Non-Fraud', 'Fraud'],
                                       output_dict=True)
        
        # Save classification report
        report_path = os.path.join(self.config['analysis_dir'], f'{model_name}_classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"💾 Classification report saved to: {report_path}")
        
        # Log key metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        self.logger.info(f"\n📊 Confusion Matrix Details:")
        self.logger.info(f"   True Negatives (TN):  {tn:,}")
        self.logger.info(f"   False Positives (FP): {fp:,}")
        self.logger.info(f"   False Negatives (FN): {fn:,}")
        self.logger.info(f"   True Positives (TP):  {tp:,}")
        
        # Calculate additional metrics
        if tp + fn > 0:
            fraud_recall = tp / (tp + fn)
            self.logger.info(f"   Fraud Detection Rate (Recall): {fraud_recall:.4f}")
        if fp + tn > 0:
            false_positive_rate = fp / (fp + tn)
            self.logger.info(f"   False Positive Rate: {false_positive_rate:.4f}")
        
        return cm, report
    
    def evaluate_all_models(self, model_names: List[str], start_date: str, end_date: str):
        """Evaluate all specified models."""
        try:
            # Initialize Spark
            self.initialize_spark()
            
            # Load evaluation data
            eval_df = self.load_evaluation_data(start_date, end_date)
            
            # Store all results
            all_results = []
            
            # Evaluate each model
            for model_name in model_names:
                self.logger.info("\n" + "=" * 80)
                self.logger.info(f"PROCESSING MODEL: {model_name.upper().replace('_', ' ')}")
                self.logger.info("=" * 80)
                
                # Load model
                model = self.load_model(model_name)
                if model is None:
                    self.logger.warning(f"Skipping {model_name} - model not found")
                    continue
                
                # Generate predictions
                predictions = self.generate_predictions(model, eval_df, model_name)
                
                # Evaluate
                metrics = self.evaluate_model(predictions, model_name)
                all_results.append(metrics)
                
                # Create confusion matrix
                self.create_confusion_matrix(predictions, model_name)
                
                self.logger.info(f"✅ Completed evaluation for {model_name}\n")
            
            # Create comparison summary
            self.create_comparison_summary(all_results)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("ALL EVALUATIONS COMPLETE")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()
    
    def create_comparison_summary(self, results: List[Dict]):
        """Create a comparison summary of all models."""
        if not results:
            return
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL COMPARISON SUMMARY")
        self.logger.info("=" * 80)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by F1 score
        df = df.sort_values('f1', ascending=False)
        
        # Display table
        self.logger.info("\n" + df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(self.config['analysis_dir'], 'model_comparison_results.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"\n💾 Comparison results saved to: {csv_path}")
        
        # Create visualization
        self.create_comparison_plots(df)
    
    def create_comparison_plots(self, df: pd.DataFrame):
        """Create comparison plots for all models."""
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            df_sorted = df.sort_values(metric, ascending=True)
            
            bars = ax.barh(df_sorted['model'].str.replace('_', ' ').str.title(), 
                          df_sorted[metric])
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xlabel(metric.upper().replace('_', '-'), fontsize=11)
            ax.set_title(f'{metric.upper().replace("_", "-")} Comparison', 
                        fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.4f}', va='center', fontsize=9)
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = os.path.join(self.config['analysis_dir'], 'model_comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"💾 Comparison plots saved to: {plot_path}")


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration."""
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
        "log_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models/logs",
        "analysis_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis",
        "predictions_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis/predictions"
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation and Confusion Matrix Generator')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['decision_tree', 'random_forest', 'logistic_regression', 'gbt', 'all'],
                       default=['all'], help='Models to evaluate (default: all)')
    parser.add_argument('--start-date', type=str, default='2025-07-01',
                       help='Evaluation start date (default: 2025-07-01)')
    parser.add_argument('--end-date', type=str, default='2025-07-31',
                       help='Evaluation end date (default: 2025-07-31)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine which models to evaluate
    if 'all' in args.models:
        model_names = ['decision_tree', 'random_forest', 'logistic_regression', 'gbt']
    else:
        model_names = args.models
    
    # Run evaluation
    evaluator = ModelEvaluator(config)
    success = evaluator.evaluate_all_models(model_names, args.start_date, args.end_date)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
