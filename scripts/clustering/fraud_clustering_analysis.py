#!/usr/bin/env python3
"""
Fraud Clustering Analysis with KMeans
======================================
Trains a KMeans clustering model to identify fraud transaction patterns.
Uses 9 clusters and t-SNE visualization to analyze transaction patterns.

Strategy:
- Keep all fraud transactions
- Downsample non-fraud to 10%
- Apply KMeans clustering
- Visualize all clusters with t-SNE

Analysis Period: One month of transactions (default: 2025-07-01 to 2025-07-31)

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
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import ClusteringEvaluator

from sklearn.manifold import TSNE

import logging

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'


class FraudClusteringAnalysis:
    """KMeans clustering to identify fraud-like transaction patterns."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spark = None
        self.logger = None
        self.pipeline_model = None
        self.kmeans_model = None
        self.feature_cols = []
        
        self._setup_directories()
        self._setup_logging()
    
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config['model_dir'], self.config['log_dir'], 
                         self.config['analysis_dir'], self.config['cluster_dir']]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_filename = f"fraud_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.config['log_dir'], log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('FraudClusteringAnalysis')
        self.logger.info("=" * 80)
        self.logger.info("FRAUD CLUSTERING ANALYSIS WITH KMEANS")
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
            .appName("fraud-clustering-analysis")
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
    
    def load_data(self, start_date: str, end_date: str) -> DataFrame:
        """Load data from ClickHouse for clustering analysis."""
        self.logger.info("=" * 80)
        self.logger.info("LOADING DATA")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        ch_cfg = self.config['clickhouse']
        
        query = f"""
            SELECT {', '.join(data_cfg['selected_features'])}
            FROM clickhouse.{ch_cfg['database']}.{data_cfg['table_name']}
            WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                AND mbar_account_type_name = 'Customer Account'
        """
        
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        start_time = time.time()
        df = self.spark.sql(query)  
        count = df.count()
        duration = time.time() - start_time
        
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
        non_fraud_sampled = non_fraud_df.sample(
            withReplacement=False, 
            fraction=sample_rate, 
            seed=self.config['clustering']['random_seed']
        )
        sampled_count = non_fraud_sampled.count()
        
        # Combine
        balanced_df = fraud_df.union(non_fraud_sampled)
        total_count = balanced_df.count()
        
        self.logger.info(f"After downsampling - Fraud: {fraud_count:,}, Non-fraud: {sampled_count:,}")
        self.logger.info(f"Total samples: {total_count:,}")
        self.logger.info(f"Fraud ratio: {(fraud_count/total_count)*100:.2f}%")
        
        return balanced_df
    
    def build_clustering_pipeline(self) -> Pipeline:
        """Build KMeans clustering pipeline with preprocessing."""
        self.logger.info("=" * 80)
        self.logger.info("BUILDING CLUSTERING PIPELINE")
        self.logger.info("=" * 80)
        
        data_cfg = self.config['data']
        cluster_cfg = self.config['clustering']
        
        # Identify column types
        excluded = [data_cfg['target_column'], 'cutoff_date', 'mbar_account_type_name']
        all_features = [f for f in data_cfg['selected_features'] if f not in excluded]
        
        # Categorical and numeric columns
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
            outputCol="features_unscaled",
            handleInvalid="skip"
        )
        stages.append(assembler)
        
        # Stage 3: Feature Scaling (important for KMeans)
        scaler = StandardScaler(
            inputCol="features_unscaled",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)
        
        # Stage 4: KMeans Clustering
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=cluster_cfg['n_clusters'],
            seed=cluster_cfg['random_seed'],
            maxIter=cluster_cfg['max_iter'],
            initMode="k-means||"
        )
        stages.append(kmeans)
        
        # Create pipeline
        pipeline = Pipeline(stages=stages)
        
        self.logger.info(f"✅ Pipeline built with {len(stages)} stages:")
        self.logger.info(f"   1. String indexers ({len(string_cols)} features)")
        self.logger.info(f"   2. Vector assembler ({len(self.feature_cols)} features)")
        self.logger.info(f"   3. Standard scaler")
        self.logger.info(f"   4. KMeans (k={cluster_cfg['n_clusters']})")
        
        return pipeline
    
    def train_clustering(self, df: DataFrame) -> PipelineModel:
        """Train the KMeans clustering pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING KMEANS CLUSTERING")
        self.logger.info("=" * 80)
        
        # Build pipeline
        pipeline = self.build_clustering_pipeline()
        
        # Train
        self.logger.info("Training KMeans model...")
        start_time = time.time()
        self.pipeline_model = pipeline.fit(df)
        duration = time.time() - start_time
        
        self.logger.info(f"✅ Training completed in {duration:.2f}s")
        
        # Extract KMeans model from pipeline
        self.kmeans_model = self.pipeline_model.stages[-1]
        
        # Evaluate clustering
        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="cluster",
            metricName="silhouette"
        )
        
        predictions = self.pipeline_model.transform(df)
        silhouette = evaluator.evaluate(predictions)
        
        self.logger.info(f"   Silhouette Score: {silhouette:.4f}")
        self.logger.info(f"   Number of clusters: {self.config['clustering']['n_clusters']}")
        
        return self.pipeline_model
    
    def analyze_clusters(self, df: DataFrame) -> Tuple[DataFrame, pd.DataFrame]:
        """Analyze cluster composition and fraud distribution."""
        self.logger.info("=" * 80)
        self.logger.info("ANALYZING CLUSTER COMPOSITION")
        self.logger.info("=" * 80)
        
        # Generate predictions
        predictions = self.pipeline_model.transform(df)
        
        # Analyze cluster-fraud distribution
        cluster_analysis = predictions.groupBy('cluster', 'fraud_flag').count()
        cluster_analysis_pd = cluster_analysis.toPandas()
        
        # Calculate fraud rate per cluster
        cluster_stats = []
        for cluster_id in range(self.config['clustering']['n_clusters']):
            cluster_data = cluster_analysis_pd[cluster_analysis_pd['cluster'] == cluster_id]
            
            total = cluster_data['count'].sum()
            fraud_count = cluster_data[cluster_data['fraud_flag'] == 1]['count'].sum() if len(cluster_data[cluster_data['fraud_flag'] == 1]) > 0 else 0
            non_fraud_count = cluster_data[cluster_data['fraud_flag'] == 0]['count'].sum() if len(cluster_data[cluster_data['fraud_flag'] == 0]) > 0 else 0
            
            fraud_rate = (fraud_count / total * 100) if total > 0 else 0
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'total_transactions': int(total),
                'fraud_count': int(fraud_count),
                'non_fraud_count': int(non_fraud_count),
                'fraud_rate_percent': fraud_rate
            })
        
        cluster_stats_df = pd.DataFrame(cluster_stats).sort_values('fraud_rate_percent', ascending=False)
        
        self.logger.info("\n📊 Cluster Analysis:")
        self.logger.info(cluster_stats_df.to_string(index=False))
        
        # Save cluster statistics
        stats_path = os.path.join(self.config['cluster_dir'], 'cluster_statistics.csv')
        cluster_stats_df.to_csv(stats_path, index=False)
        self.logger.info(f"\n💾 Cluster statistics saved to: {stats_path}")
        
        return predictions, cluster_stats_df
    
    def create_tsne_visualization(self, predictions: DataFrame, cluster_stats_df: pd.DataFrame):
        """Create t-SNE visualization of clusters."""
        self.logger.info("=" * 80)
        self.logger.info("CREATING T-SNE VISUALIZATION")
        self.logger.info("=" * 80)
        
        # Sample data for t-SNE (t-SNE is computationally expensive)
        sample_size = min(10000, predictions.count())
        self.logger.info(f"Sampling {sample_size} transactions for t-SNE visualization...")
        
        # Get features, clusters, and fraud labels
        sample_df = predictions.select('features', 'cluster', 'fraud_flag').limit(sample_size).toPandas()
        
        # Convert sparse vectors to dense numpy arrays
        features_array = np.array([vec.toArray() for vec in sample_df['features']])
        clusters = sample_df['cluster'].values
        fraud_labels = sample_df['fraud_flag'].values
        
        self.logger.info(f"Running t-SNE on {features_array.shape[0]} samples with {features_array.shape[1]} features...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=self.config['clustering']['random_seed'], 
                   perplexity=30, max_iter=1000, verbose=1)
        tsne_results = tsne.fit_transform(features_array)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Color by cluster
        ax1 = axes[0]
        scatter1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                              c=clusters, cmap='tab10', alpha=0.6, s=20)
        ax1.set_title('t-SNE Visualization: Colored by Cluster', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1', fontsize=12)
        ax1.set_ylabel('t-SNE Component 2', fontsize=12)
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Cluster ID', fontsize=11)
        
        # Plot 2: Color by fraud label
        ax2 = axes[1]
        colors = ['green' if label == 0 else 'red' for label in fraud_labels]
        scatter2 = ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                              c=colors, alpha=0.6, s=20)
        ax2.set_title('t-SNE Visualization: Colored by Fraud Label', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1', fontsize=12)
        ax2.set_ylabel('t-SNE Component 2', fontsize=12)
        
        # Add legend for fraud labels
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Non-Fraud'),
                         Patch(facecolor='red', label='Fraud')]
        ax2.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        # Save t-SNE plot
        tsne_path = os.path.join(self.config['cluster_dir'], 'tsne_cluster_visualization.png')
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"💾 t-SNE visualization saved to: {tsne_path}")
        
        # Save t-SNE coordinates
        tsne_df = pd.DataFrame({
            'tsne_1': tsne_results[:, 0],
            'tsne_2': tsne_results[:, 1],
            'cluster': clusters,
            'fraud_flag': fraud_labels
        })
        
        tsne_csv_path = os.path.join(self.config['cluster_dir'], 'tsne_coordinates.csv')
        tsne_df.to_csv(tsne_csv_path, index=False)
        self.logger.info(f"💾 t-SNE coordinates saved to: {tsne_csv_path}")
    
    def visualize_clusters(self, cluster_stats_df: pd.DataFrame):
        """Create visualizations of cluster analysis."""
        self.logger.info("=" * 80)
        self.logger.info("CREATING VISUALIZATIONS")
        self.logger.info("=" * 80)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cluster sizes
        ax1 = axes[0, 0]
        bars = ax1.bar(cluster_stats_df['cluster_id'], cluster_stats_df['total_transactions'])
        ax1.set_xlabel('Cluster ID', fontsize=12)
        ax1.set_ylabel('Number of Transactions', fontsize=12)
        ax1.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Color bars by fraud rate
        colors = plt.cm.RdYlGn_r(cluster_stats_df['fraud_rate_percent'] / 100)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot 2: Fraud rate by cluster
        ax2 = axes[0, 1]
        bars2 = ax2.bar(cluster_stats_df['cluster_id'], cluster_stats_df['fraud_rate_percent'])
        ax2.set_xlabel('Cluster ID', fontsize=12)
        ax2.set_ylabel('Fraud Rate (%)', fontsize=12)
        ax2.set_title('Fraud Rate by Cluster', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Color bars by fraud rate (gradient)
        colors = plt.cm.RdYlGn_r(cluster_stats_df['fraud_rate_percent'] / cluster_stats_df['fraud_rate_percent'].max())
        for bar, color in zip(bars2, colors):
            bar.set_color(color)
        
        # Plot 3: Fraud vs Non-Fraud distribution
        ax3 = axes[1, 0]
        cluster_ids = cluster_stats_df['cluster_id']
        fraud_counts = cluster_stats_df['fraud_count']
        non_fraud_counts = cluster_stats_df['non_fraud_count']
        
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        ax3.bar(x - width/2, fraud_counts, width, label='Fraud', color='red', alpha=0.7)
        ax3.bar(x + width/2, non_fraud_counts, width, label='Non-Fraud', color='green', alpha=0.7)
        ax3.set_xlabel('Cluster ID', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Fraud vs Non-Fraud by Cluster', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(cluster_ids)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text with top clusters by fraud rate
        top_clusters = cluster_stats_df.head(5)
        
        summary_text = f"""
Clustering Summary
{'='*40}

Total Clusters: {len(cluster_stats_df)}
Total Transactions: {cluster_stats_df['total_transactions'].sum():,}
Total Frauds: {cluster_stats_df['fraud_count'].sum():,}

Top 5 Clusters by Fraud Rate:
{'-'*40}
"""
        for _, row in top_clusters.iterrows():
            summary_text += f"\nCluster {int(row['cluster_id'])}:\n"
            summary_text += f"  Fraud Rate: {row['fraud_rate_percent']:.2f}%\n"
            summary_text += f"  Total: {int(row['total_transactions']):,}\n"
            summary_text += f"  Frauds: {int(row['fraud_count']):,}\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.config['cluster_dir'], 'cluster_analysis_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"💾 Visualizations saved to: {plot_path}")
    
    def save_model(self):
        """Save the trained clustering pipeline."""
        model_path = os.path.join(self.config['model_dir'], 'fraud_clustering_kmeans_model')
        self.pipeline_model.write().overwrite().save(model_path)
        self.logger.info(f"💾 Clustering model saved to: {model_path}")
    
    def run(self, start_date: str, end_date: str):
        """Execute the complete clustering analysis."""
        try:
            overall_start = time.time()
            
            # Initialize Spark
            self.initialize_spark()
            
            # Load data
            df = self.load_data(start_date, end_date)
            
            # Downsample non-fraud data
            balanced_df = self.downsample_data(df, sample_rate=0.01)
            # Train clustering
            train_start = time.time()
            self.train_clustering(balanced_df)
            train_duration = time.time() - train_start
            self.logger.info(f"⏱️  Total training time: {train_duration:.2f}s")
            self.save_model()

            
            # Analyze clusters
            predictions, cluster_stats_df = self.analyze_clusters(balanced_df)
            
            # Create t-SNE visualization
            self.create_tsne_visualization(predictions, cluster_stats_df)
            
            # Visualize results
            self.visualize_clusters(cluster_stats_df)
            
            
            # Summary
            total_time = time.time() - overall_start
            self.logger.info("\n" + "=" * 80)
            self.logger.info("CLUSTERING ANALYSIS COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Number of clusters: {self.config['clustering']['n_clusters']}")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Clustering analysis failed: {str(e)}", exc_info=True)
            return False
        
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
        "clustering": {
            "n_clusters": 9,
            "random_seed": 42,
            "max_iter": 100
        },
        "model_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models",
        "log_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models/logs",
        "analysis_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis",
        "cluster_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis/clustering"
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config


def main():
    parser = argparse.ArgumentParser(description='Fraud Clustering Analysis with KMeans')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--start-date', type=str, default='2025-07-01',
                       help='Analysis start date (default: 2025-07-01)')
    parser.add_argument('--end-date', type=str, default='2025-07-31',
                       help='Analysis end date (default: 2025-07-31)')
    parser.add_argument('--n-clusters', type=int, default=9,
                       help='Number of clusters (default: 9)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    config['clustering']['n_clusters'] = args.n_clusters
    
    # Run clustering analysis
    analyzer = FraudClusteringAnalysis(config)
    success = analyzer.run(args.start_date, args.end_date)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
