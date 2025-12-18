#!/usr/bin/env python3
"""
Multi-Model Fraud Detection ML Pipeline
========================================
Streamlined pipeline using Spark ML Pipeline stages.
Trains multiple models (Decision Tree, Random Forest, Logistic Regression, GBT)
on downsampled data and evaluates on separate test period.

Training Period: 2025-03-01 to 2025-06-30 (downsampled non-fraud to 10%)
Evaluation Period: 2025-07-01 to 2025-07-31

Author: AI Team
Date: November 2025
"""

import os
import sys
import json
import time
import argparse

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
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
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, GBTClassifier
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
            # .master("spark://10.205.161.118:7077")
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
            -- WHERE trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000 and cutoff_date between '2025-06-01' and '2025-06-30'      AND mbar_account_type_name = 'Customer Account'
        WHERE cutoff_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        AND mbar_account_type_name = 'Customer Account'
        AND trx_channel='Payment Gateway'
        AND trx_type='Online Payment'
        AND ac_to IS NOT NULL 
        AND ac_to<>''
        AND start_balance<>end_balance
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
    
    def downsample_data(self, df: DataFrame, sample_rate: float = 0.1) -> DataFrame:
        """Downsample non-fraud data while keeping all fraud data."""
        self.logger.info("=" * 80)
        self.logger.info("DOWNSAMPLING NON-FRAUD DATA")
        self.logger.info("=" * 80)
        
        # Separate fraud and non-fraud
        fraud_df = df.filter(col('fraud_flag') == 1)
        non_fraud_df = df.filter(col('fraud_flag') == 0)
        
        fraud_count = fraud_df.count()
        non_fraud_count = non_fraud_df.count()
        
        self.logger.info(f"Original - Fraud: {fraud_count:,}, Non-fraud: {non_fraud_count:,}")
        
        # Sample non-fraud data
        non_fraud_sampled = non_fraud_df.sample(withReplacement=False, fraction=sample_rate, seed=self.config['training']['random_seed'])
        sampled_count = non_fraud_sampled.count()
        
        # Combine
        balanced_df = fraud_df.union(non_fraud_sampled)
        total_count = balanced_df.count()
        
        self.logger.info(f"After downsampling - Fraud: {fraud_count:,}, Non-fraud: {sampled_count:,}")
        self.logger.info(f"Total samples: {total_count:,}")
        self.logger.info(f"Fraud ratio: {(fraud_count/total_count)*100:.2f}%")
        
        return balanced_df
    
    def build_pipeline(self, model_type: str = 'decision_tree') -> Pipeline:
        """
        Build complete Spark ML Pipeline with all stages.
        
        Args:
            model_type: 'decision_tree', 'random_forest', 'logistic_regression', or 'gbt'
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
        # string_cols = ['trx_channel', 'trx_type', 'mbar_registered_channel']
        string_cols = ['trx_channel', 'trx_type']
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
        elif model_type == 'random_forest':
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
        elif model_type == 'logistic_regression':
            lr_cfg = self.config['models']['logistic_regression']
            model = LogisticRegression(
                featuresCol="features",
                labelCol=data_cfg['target_column'],
                predictionCol="prediction",
                probabilityCol="probability",
                maxIter=lr_cfg['max_iter'],
                regParam=lr_cfg['reg_param'],
                elasticNetParam=lr_cfg['elastic_net_param'],
                family="binomial",
                standardization=True
            )
            self.logger.info(f"Logistic Regression config: maxIter={lr_cfg['max_iter']}, regParam={lr_cfg['reg_param']}")
        elif model_type == 'gbt':
            gbt_cfg = self.config['models']['gbt']
            model = GBTClassifier(
                featuresCol="features",
                labelCol=data_cfg['target_column'],
                predictionCol="prediction",
                maxIter=gbt_cfg['max_iter'],
                maxDepth=gbt_cfg['max_depth'],
                maxBins=gbt_cfg['max_bins'],
                stepSize=gbt_cfg['step_size'],
                seed=self.config['training']['random_seed']
            )
            self.logger.info(f"GBT config: maxIter={gbt_cfg['max_iter']}, depth={gbt_cfg['max_depth']}, stepSize={gbt_cfg['step_size']}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
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

        # PySpark confusion matrix calculation
        self.logger.info("Confusion Matrix:")
        cm_counts = predictions.groupBy(target_col, "prediction").count().collect()
        # Initialize counts
        TP = TN = FP = FN = 0
        for row in cm_counts:
            if row[target_col] == 1 and row['prediction'] == 1:
                TP = row['count']
            elif row[target_col] == 0 and row['prediction'] == 0:
                TN = row['count']
            elif row[target_col] == 0 and row['prediction'] == 1:
                FP = row['count']
            elif row[target_col] == 1 and row['prediction'] == 0:
                FN = row['count']
        self.logger.info(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        # Save confusion matrix as CSV and plot
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        matrix = [[TN, FP], [FN, TP]]
        df_cm = pd.DataFrame(matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        cm_path = os.path.join(self.config['analysis_dir'], f'{model_name}_confusion_matrix.csv')
        df_cm.to_csv(cm_path)
        self.logger.info(f"💾 Confusion matrix saved to: {cm_path}")
        plt.figure(figsize=(5,4))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name.replace("_", " ").title()} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        cm_plot_path = os.path.join(self.config['analysis_dir'], f'{model_name}_confusion_matrix.png')
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"💾 Confusion matrix plot saved to: {cm_plot_path}")
        
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
        model_path = os.path.join(self.config['model_dir'], f'{model_name}_pipeline_model_fraud_scenario_v6')
        self.pipeline_model.write().overwrite().save(model_path)
        self.logger.info(f"💾 Pipeline saved to: {model_path}")
    
    def load_data_by_period(self, start_date: str, end_date: str, eval=False) -> DataFrame:
        """Load data for a specific date period."""
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        if eval:
            query = f"""
                SELECT {', '.join(data_cfg['selected_features'])}
                FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
                WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                    AND mbar_account_type_name = 'Customer Account'
                    AND trx_channel='Payment Gateway'
                    AND trx_type='Online Payment'
                    AND ac_to IS NOT NULL 
                    AND ac_to<>''
                    AND start_balance<>end_balance


                -- WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                -- AND mbar_account_type_name = 'Customer Account'
                -- AND trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000
            """
        else:
            query = f"""
                SELECT {', '.join(data_cfg['selected_features'])}
                FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
                WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                    AND mbar_account_type_name = 'Customer Account'
                    AND trx_channel='Payment Gateway'
                    AND trx_type='Online Payment'
                    AND ac_to IS NOT NULL 
                    AND ac_to<>''
                    AND start_balance<>end_balance

               -- WHERE (cutoff_date BETWEEN '{start_date}' AND '{end_date}'
               -- AND mbar_account_type_name = 'Customer Account'
               -- AND trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000)
               -- OR (fraud_flag=1 and cutoff_date<='{end_date}' and trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000)
            """

        self.logger.info(f"Loading data from {start_date} to {end_date}...")
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
    
    def run(self, model_types: list = None):
        """Execute the complete pipeline for all models."""
        if model_types is None:
            model_types = ['random_forest', 'logistic_regression', 'gbt','decision_tree']
        
        try:
            start_time = time.time()
            
            # Initialize
            self.initialize_spark()
            
            # Load training data (2025-03-01 to 2025-06-30)
            self.logger.info("=" * 80)
            self.logger.info("LOADING TRAINING DATA")
            self.logger.info("=" * 80)
            train_df = self.load_data_by_period('2025-06-01', '2025-06-01')
            
            # Downsample non-fraud data to 10%
            # train_df_balanced = train_df
            train_df_balanced = self.downsample_data(train_df, sample_rate=0.1)

            # Store results for all models
            all_results = {}
            
            # Train and evaluate each model
            for model_type in model_types:
                self.logger.info("\n" + "=" * 80)
                self.logger.info(f"TRAINING {model_type.upper().replace('_', ' ')} MODEL")
                self.logger.info("=" * 80)
                
                model_start = time.time()
                
                # Train pipeline
                self.train_pipeline(train_df_balanced, model_type)
                model_duration = time.time() - model_start
                self.logger.info(f"✅ {model_type} completed in {model_duration:.2f}s")
                # Save model
                self.save_pipeline(model_type)

                # Feature importance (if supported)
                try:
                    self.analyze_feature_importance(model_type)
                except Exception as e:
                    self.logger.warning(f"Could not extract feature importance: {str(e)}")

            eval_df = self.load_data_by_period('2025-07-01', '2025-07-01', eval=True)

            for model_type in model_types:
                # Evaluate on test data
                metrics = self.evaluate_pipeline(eval_df, model_type)
                self.logger.info(f"Evaluation for {model_type} completed.")
                all_results[model_type] = metrics
                
                
                
            
            # Summary
            total_time = time.time() - start_time
            self.logger.info("\n" + "=" * 80)
            self.logger.info("ALL MODELS TRAINING COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info("\nModel Performance Summary (on 2025-07-01 to 2025-07-31):")
            for model_type, metrics in all_results.items():
                self.logger.info(f"\n{model_type.upper().replace('_', ' ')}:")
                self.logger.info(f"   AUC-ROC: {metrics['auc']:.4f}")
                self.logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
                self.logger.info(f"   Precision: {metrics['precision']:.4f}")
                self.logger.info(f"   Recall: {metrics['recall']:.4f}")
                self.logger.info(f"   F1-Score: {metrics['f1']:.4f}")
            
            # Save summary to JSON
            summary_path = os.path.join(self.config['analysis_dir'], 'model_comparison_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            self.logger.info(f"\n💾 Summary saved to: {summary_path}")
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
            # "selected_features": [
            #     'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 
            #     'start_balance', 'trx_amt', 'mbar_registered_channel',
            #     'hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 
            #     'is_business_hours', 'is_unusual_hour', 'night_weekend_combo',
            #     'txn_txns_3d', 'txn_total_amount_3d', 'txn_avg_amount_3d',
            #     'txn_max_amount_3d', 'txn_min_amount_3d', 'txn_unique_recipients_3d',
            #     'txn_unique_channels_3d', 'txn_unique_types_3d', 'txn_is_high_activity_3d',
            #     'txn_multi_channel_recent', 'txn_amount_deviation_from_avg',
            #     'txn_night_txns_3d', 'txn_weekend_txns_3d',
            #     'channel_new_jc_app', 'channel_ussd', 'channel_ussd_api',
            #     'channel_payment_gateway', 'channel_mobile_app',
            #     'type_transfer_c2c', 'type_transfer_c2b', 'type_bill_payment',
            #     'type_mobile_load', 'user_total_txns_3d', 'user_total_amount_3d',
            #     'user_avg_amount_3d', 'user_max_amount_3d', 'user_unique_recipients_3d',
            #     'user_unique_channels_3d', 'user_total_txns_7d', 'user_avg_amount_7d',
            #     'user_max_amount_7d', 'user_night_txns_7d', 'user_weekend_txns_7d'
            # ]
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
            },
            "logistic_regression": {
                "max_iter": 100,
                "reg_param": 0.01,
                "elastic_net_param": 0.5
            },
            # "gbt": {
            #     "max_iter": 50,
            #     "max_depth": 5,
            #     "max_bins": 128,
            #     "step_size": 0.1
            # }
            "gbt": {
                "max_iter": 200,
                "max_depth": 20,
                "max_bins": 256,
                "step_size": 0.001
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
    parser = argparse.ArgumentParser(description='Multi-Model Fraud Detection Pipeline')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['decision_tree', 'random_forest', 'logistic_regression', 'gbt', 'all'],
                       default=['all'], help='Model types to train (default: all)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine which models to train
    if 'all' in args.models:
        model_types = ['decision_tree', 'random_forest', 'logistic_regression', 'gbt']
    else:
        model_types = args.models
    
    # Run pipeline
    pipeline = SimpleFraudPipeline(config)
    success = pipeline.run(model_types)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
