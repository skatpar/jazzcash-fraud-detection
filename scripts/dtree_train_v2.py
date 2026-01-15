#!/usr/bin/env python3
"""
Simplified Fraud Detection ML Pipeline
=======================================
Streamlined pipeline using Spark ML Pipeline stages.

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import logging


class SimpleFraudPipeline:
    """Simplified fraud detection pipeline using Spark ML Pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spark = None
        self.logger = None
        self.pipeline_model = None
        self.feature_cols = []
        
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config['model_dir'], self.config['log_dir'], self.config['analysis_dir']]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_filename = f"fraud_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.config['log_dir'], log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('SimpleFraudPipeline')
        self.logger.info("=" * 80)
        self.logger.info("SIMPLIFIED FRAUD DETECTION PIPELINE")
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
        
        self.spark = (SparkSession.builder
            .appName("spark-clickhouse-fraud-detection")
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
        self.logger.info(f"   Shuffle Partitions: {spark_cfg.get('shuffle_partitions', 200)}")
        self.logger.info(f"   ClickHouse catalog: clickhouse.{ch['database']}")
    
    def load_data(self) -> DataFrame:
        """Load data from ClickHouse using Spark SQL."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING DATA")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        
        query = f"""
            SELECT {', '.join(data_cfg['selected_features'])}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE cutoff_date BETWEEN '{data_cfg['start_date']}' AND '{data_cfg['end_date']}'
                AND mbar_account_type_name = 'Customer Account'
        """
        
        self.logger.info(f"Date range: {data_cfg['start_date']} to {data_cfg['end_date']}")
        
        start = time.time()
        df = self.spark.sql(query)
        count = df.count()
        duration = time.time() - start
        
        self.logger.info(f"✅ Loaded {count:,} rows in {duration:.2f}s")
        
        # Log class distribution
        fraud_dist = df.groupBy('fraud_flag').count().orderBy('fraud_flag').collect()
        for row in fraud_dist:
            pct = (row['count'] / count) * 100
            self.logger.info(f"   Class {row['fraud_flag']}: {row['count']:,} ({pct:.2f}%)")
        
        return df
    
    def build_pipeline(self, model_type: str = 'decision_tree') -> Pipeline:
        """
        Build complete Spark ML Pipeline with all stages.
        
        Args:
            model_type: 'decision_tree' or 'random_forest'
        """
        self.logger.info("=" * 80)
        self.logger.info(f"BUILDING {model_type.upper().replace('_', ' ')} PIPELINE")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        
        # Identify column types
        excluded = [data_cfg['target_column'], 'cutoff_date', 'mbar_account_type_name']
        
        # Get all feature columns
        all_features = [f for f in data_cfg['selected_features'] if f not in excluded]
        
        # Separate string and numeric columns (we'll determine this from schema later)
        # For now, assume these are the string columns based on the feature list
        string_cols = ['trx_channel', 'trx_type', 'mbar_registered_channel']
        numeric_cols = [f for f in all_features if f not in string_cols]
        
        self.logger.info(f"Features: {len(all_features)} total")
        self.logger.info(f"   Categorical: {len(string_cols)}")
        self.logger.info(f"   Numeric: {len(numeric_cols)}")
        
        # Build pipeline stages
        stages = []
        
        # Stage 1: String Indexing for categorical features
        indexed_cols = []
        for col_name in string_cols:
            indexer = StringIndexer(
                inputCol=col_name, 
                outputCol=f"{col_name}_idx", 
                handleInvalid="keep"
            )
            stages.append(indexer)
            indexed_cols.append(f"{col_name}_idx")
        
        # Combine numeric and indexed columns
        self.feature_cols = numeric_cols + indexed_cols
        
        # Stage 2: Vector Assembly
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        stages.append(assembler)
        
        # Stage 3: Model
        if model_type == 'decision_tree':
            dt_cfg = self.config['models']['decision_tree']
            model = DecisionTreeClassifier(
                featuresCol="features",
                labelCol=data_cfg['target_column'],
                predictionCol="prediction",
                probabilityCol="probability",
                maxDepth=dt_cfg['max_depth'],
                maxBins=dt_cfg['max_bins'],
                minInstancesPerNode=dt_cfg['min_instances_per_node'],
                impurity=dt_cfg['impurity'],
                seed=self.config['training']['random_seed']
            )
            self.logger.info(f"Decision Tree config: depth={dt_cfg['max_depth']}, bins={dt_cfg['max_bins']}")
        else:  # random_forest
            rf_cfg = self.config['models']['random_forest']
            model = RandomForestClassifier(
                featuresCol="features",
                labelCol=data_cfg['target_column'],
                predictionCol="prediction",
                probabilityCol="probability",
                numTrees=rf_cfg['num_trees'],
                maxDepth=rf_cfg['max_depth'],
                maxBins=rf_cfg['max_bins'],
                minInstancesPerNode=rf_cfg['min_instances_per_node'],
                impurity=rf_cfg['impurity'],
                subsamplingRate=rf_cfg['subsampling_rate'],
                seed=self.config['training']['random_seed']
            )
            self.logger.info(f"Random Forest config: trees={rf_cfg['num_trees']}, depth={rf_cfg['max_depth']}")
        
        stages.append(model)
        
        # Create pipeline
        pipeline = Pipeline(stages=stages)
        
        self.logger.info(f"✅ Pipeline built with {len(stages)} stages:")
        self.logger.info(f"   1. String indexers ({len(string_cols)} features)")
        self.logger.info(f"   2. Vector assembler ({len(self.feature_cols)} features)")
        self.logger.info(f"   3. {model_type.replace('_', ' ').title()} model")
        
        return pipeline
    
    def train_pipeline(self, df: DataFrame, model_type: str = 'decision_tree') -> PipelineModel:
        """Train the complete pipeline on data."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING PIPELINE")
        self.logger.info("=" * 80)
        
        # Build pipeline
        pipeline = self.build_pipeline(model_type)
        
        # Train
        self.logger.info("Training...")
        start = time.time()
        self.pipeline_model = pipeline.fit(df)
        duration = time.time() - start
        
        self.logger.info(f"✅ Training completed in {duration:.2f}s")
        
        # Extract trained model from pipeline
        model = self.pipeline_model.stages[-1]
        
        if hasattr(model, 'depth'):  # Decision Tree
            self.logger.info(f"   Tree depth: {model.depth}")
            self.logger.info(f"   Nodes: {model.numNodes}")
        elif hasattr(model, 'trees'):  # Random Forest
            self.logger.info(f"   Trees trained: {len(model.trees)}")
        
        return self.pipeline_model
    
    def evaluate_pipeline(self, df: DataFrame, model_name: str):
        """Evaluate the trained pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION")
        self.logger.info("=" * 80)
        
        # Make predictions
        predictions = self.pipeline_model.transform(df)
        
        # Evaluate
        target_col = self.config['data']['target_column']
        
        binary_eval = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
        multiclass_eval = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction")
        
        auc = binary_eval.evaluate(predictions)
        accuracy = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "accuracy"})
        precision = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "weightedPrecision"})
        recall = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "weightedRecall"})
        f1 = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "f1"})
        
        self.logger.info(f"📊 {model_name} Performance:")
        self.logger.info(f"   AUC-ROC: {auc:.4f}")
        self.logger.info(f"   Accuracy: {accuracy:.4f}")
        self.logger.info(f"   Precision: {precision:.4f}")
        self.logger.info(f"   Recall: {recall:.4f}")
        self.logger.info(f"   F1-Score: {f1:.4f}")
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def analyze_feature_importance(self, model_name: str):
        """Extract and save feature importance."""
        self.logger.info("=" * 80)
        self.logger.info("FEATURE IMPORTANCE")
        self.logger.info("=" * 80)
        
        # Get model from pipeline
        model = self.pipeline_model.stages[-1]
        
        if not hasattr(model, 'featureImportances'):
            self.logger.warning("Model does not support feature importance")
            return
        
        importance = model.featureImportances.toArray()
        
        df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Top 20 features:")
        for idx, row in df.head(20).iterrows():
            self.logger.info(f"   {row['feature']}: {row['importance']:.6f}")
        
        # Save to CSV
        csv_path = os.path.join(self.config['analysis_dir'], f'{model_name}_feature_importance.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"💾 Saved to: {csv_path}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df.head(25), y='feature', x='importance', palette='viridis')
        plt.title(f'{model_name.replace("_", " ").title()} - Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        plot_path = os.path.join(self.config['analysis_dir'], f'{model_name}_feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"💾 Plot saved to: {plot_path}")
    
    def save_pipeline(self, model_name: str):
        """Save the trained pipeline."""
        model_path = os.path.join(self.config['model_dir'], f'{model_name}_pipeline_model_v2')
        self.pipeline_model.write().overwrite().save(model_path)
        self.logger.info(f"💾 Pipeline saved to: {model_path}")
    
    def run(self, model_type: str = 'decision_tree'):
        """Execute the complete pipeline."""
        try:
            start_time = time.time()
            
            # Initialize
            self.initialize_spark()
            
            # Load data
            df = self.load_data()
            
            # Train pipeline (all stages together)
            self.train_pipeline(df, model_type)
            
            # Evaluate
            # metrics = self.evaluate_pipeline(df, model_type)
            
            # Feature importance
            # self.analyze_feature_importance(model_type)
            
            # Save
            self.save_pipeline(model_type)
            
            # Summary
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Model: {model_type}")
            # self.logger.info(f"AUC-ROC: {metrics['auc']:.4f}")
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
        "data": {
            "table_name": "stixor_fraud_features_distributed",
            "start_date": "2025-06-01",
            "end_date": "2025-06-30",
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
        "training": {
            "random_seed": 42
        },
        "models": {
            "decision_tree": {
                "max_depth": 10,
                "max_bins": 128,
                "min_instances_per_node": 100,
                "impurity": "gini"
            },
            "random_forest": {
                "num_trees": 100,
                "max_depth": 10,
                "max_bins": 128,
                "min_instances_per_node": 100,
                "impurity": "gini",
                "subsampling_rate": 0.8
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
    parser = argparse.ArgumentParser(description='Simplified Fraud Detection Pipeline')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, choices=['decision_tree', 'random_forest'],
                       default='decision_tree', help='Model type')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    
    # Run pipeline
    pipeline = SimpleFraudPipeline(config)
    success = pipeline.run(args.model)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
