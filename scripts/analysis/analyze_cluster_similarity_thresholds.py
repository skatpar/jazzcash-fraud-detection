#!/usr/bin/env python3
"""
Analyze Cluster Similarity Score Thresholds
============================================
Counts transactions and total amounts in clusters 6 and 7
at different similarity score thresholds (60%, 70%, 80%, 90%).

Usage:
    python analyze_cluster_similarity_thresholds.py --start-date 2025-07-01 --end-date 2025-07-31
"""

import os
import sys
import argparse
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, sum as spark_sum, count as spark_count, when, min as spark_min, max as spark_max
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeansModel
import math

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'


class ClusterSimilarityAnalyzer:
    """Analyze clusters by similarity score thresholds."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.spark = None
        self.pipeline_model = None
        self.kmeans_model = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger('ClusterSimilarityAnalyzer')
    
    def initialize_spark(self):
        """Initialize Spark session."""
        self.logger.info("Initializing Spark session...")
        
        try:
            existing = SparkSession.getActiveSession()
            if existing:
                existing.stop()
        except:
            pass
        
        packages = [
            "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
            "com.clickhouse:clickhouse-client:0.9.4",
            "com.clickhouse:clickhouse-http-client:0.9.4",
            "org.apache.httpcomponents.client5:httpclient5:5.2.1"
        ]
        
        self.spark = (SparkSession.builder
            .appName("cluster-similarity-threshold-analyzer")
            .master("spark://10.205.161.118:7077")
            .config("spark.jars.packages", ",".join(packages))
            .config("spark.executor.memory", "150g")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.cores", "32")
            .config("spark.executor.instances", "2")
            .getOrCreate()
        )
        
        # Configure ClickHouse
        self.spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
        self.spark.conf.set("spark.sql.catalog.clickhouse.host", "localhost")
        self.spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
        self.spark.conf.set("spark.sql.catalog.clickhouse.http_port", "8123")
        self.spark.conf.set("spark.sql.catalog.clickhouse.user", "default")
        self.spark.conf.set("spark.sql.catalog.clickhouse.password", "DfsTeChB1")
        self.spark.conf.set("spark.sql.catalog.clickhouse.database", "public")
        
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")
    
    def load_model(self):
        """Load trained pipeline model."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        self.pipeline_model = PipelineModel.load(self.model_path)
        
        # Extract KMeans model from pipeline
        for stage in self.pipeline_model.stages:
            if isinstance(stage, KMeansModel):
                self.kmeans_model = stage
                break
        
        if self.kmeans_model is None:
            raise ValueError("No KMeans model found in pipeline")
        
        self.logger.info(f"✅ Model loaded successfully")
        self.logger.info(f"   Number of clusters: {self.kmeans_model.getK()}")
    
    def load_data(self, start_date: str, end_date: str) -> DataFrame:
        """Load transaction data from ClickHouse."""
        self.logger.info(f"Loading data from {start_date} to {end_date}...")
        
        features = [
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
        
        query = f"""
        SELECT {', '.join(features)}
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
            AND mbar_account_type_name = 'Customer Account'
        """
        
        df = self.spark.sql(query)
        count = df.count()
        self.logger.info(f"✅ Loaded {count:,} transactions")
        
        return df
    
    def calculate_similarity_scores(self, df: DataFrame) -> DataFrame:
        """Calculate similarity scores for all transactions."""
        self.logger.info("Calculating similarity scores...")
        
        # Generate predictions (cluster assignments)
        predictions = self.pipeline_model.transform(df)
        
        # Get cluster centers
        cluster_centers = self.kmeans_model.clusterCenters()
        
        # Create UDF to calculate Euclidean distance
        def euclidean_distance(features, cluster_id):
            """Calculate Euclidean distance between feature vector and cluster center."""
            if features is None or cluster_id is None:
                return None
            
            center = cluster_centers[int(cluster_id)]
            feature_array = features.toArray()
            
            # Calculate Euclidean distance
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(feature_array, center)))
            return float(distance)
        
        # Register UDF
        distance_udf = udf(euclidean_distance, DoubleType())
        
        # Add distance and similarity columns
        predictions_with_scores = predictions.withColumn(
            'distance_to_center',
            distance_udf(col('features'), col('cluster'))
        ).withColumn(
            'similarity_score',
            1.0 / (1.0 + col('distance_to_center'))
        )
        
        self.logger.info("✅ Similarity scores calculated")
        
        return predictions_with_scores
    
    def analyze_top_n_by_similarity(self, df_with_scores: DataFrame, 
                                    target_clusters: list = [6, 7],
                                    top_n_values: list = [100, 500, 1000, 5000, 10000]) -> pd.DataFrame:
        """Analyze transaction counts and amounts for top N most similar non-fraud transactions."""
        self.logger.info("=" * 80)
        self.logger.info("ANALYZING TOP N MOST SIMILAR NON-FRAUD TRANSACTIONS")
        self.logger.info("=" * 80)
        
        # Filter to non-fraud transactions only
        non_fraud_data = df_with_scores.filter(col('fraud_flag') == 0)
        
        results = []
        
        for cluster_id in target_clusters:
            self.logger.info(f"\n📊 Cluster {cluster_id} Analysis:")
            self.logger.info("-" * 40)
            
            # Filter by cluster
            cluster_df = non_fraud_data.filter(col('cluster') == cluster_id)
            
            # Order by similarity score (descending) to get most similar first
            ordered_df = cluster_df.orderBy(col('similarity_score').desc())
            
            for top_n in top_n_values:
                # Get top N most similar transactions
                top_n_df = ordered_df.limit(top_n)
                
                # Calculate statistics
                stats = top_n_df.agg(
                    spark_count('*').alias('transaction_count'),
                    spark_sum('trx_amt').alias('total_amount'),
                    spark_sum(col('similarity_score')).alias('total_similarity'),
                    spark_min(col('similarity_score')).alias('min_similarity'),
                    spark_max(col('similarity_score')).alias('max_similarity')
                ).collect()[0]
                
                transaction_count = stats['transaction_count'] if stats['transaction_count'] is not None else 0
                total_amount = stats['total_amount'] if stats['total_amount'] is not None else 0
                total_similarity = stats['total_similarity'] if stats['total_similarity'] is not None else 0
                min_similarity = stats['min_similarity'] if stats['min_similarity'] is not None else 0
                max_similarity = stats['max_similarity'] if stats['max_similarity'] is not None else 0
                
                avg_similarity = (total_similarity / transaction_count) if transaction_count > 0 else 0
                avg_amount = (total_amount / transaction_count) if transaction_count > 0 else 0
                
                results.append({
                    'cluster_id': cluster_id,
                    'top_n': top_n,
                    'transaction_count': int(transaction_count),
                    'total_amount': float(total_amount),
                    'avg_similarity_score': float(avg_similarity),
                    'min_similarity_score': float(min_similarity),
                    'max_similarity_score': float(max_similarity),
                    'avg_amount': float(avg_amount)
                })
                
                self.logger.info(f"Top {top_n} transactions:")
                self.logger.info(f"  Count: {transaction_count:,}")
                self.logger.info(f"  Total Amount: {total_amount:,.2f}")
                self.logger.info(f"  Avg Similarity: {avg_similarity:.4f}")
                self.logger.info(f"  Min Similarity: {min_similarity:.4f}")
                self.logger.info(f"  Max Similarity: {max_similarity:.4f}")
                self.logger.info(f"  Avg Amount: {avg_amount:,.2f}")
        
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def create_visualizations(self, results_df: pd.DataFrame, output_dir: str):
        """Create visualizations of top N similarity analysis."""
        self.logger.info("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top N Most Similar Non-Fraud Transactions Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Transaction Count by Top N
        ax1 = axes[0, 0]
        for cluster_id in results_df['cluster_id'].unique():
            cluster_data = results_df[results_df['cluster_id'] == cluster_id]
            ax1.plot(cluster_data['top_n'], 
                    cluster_data['transaction_count'],
                    marker='o', linewidth=2, markersize=8, label=f'Cluster {cluster_id}')
        
        ax1.set_xlabel('Top N Transactions', fontsize=12)
        ax1.set_ylabel('Transaction Count', fontsize=12)
        ax1.set_title('Transaction Count by Top N', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Total Amount by Top N
        ax2 = axes[0, 1]
        for cluster_id in results_df['cluster_id'].unique():
            cluster_data = results_df[results_df['cluster_id'] == cluster_id]
            ax2.plot(cluster_data['top_n'], 
                    cluster_data['total_amount'],
                    marker='s', linewidth=2, markersize=8, label=f'Cluster {cluster_id}')
        
        ax2.set_xlabel('Top N Transactions', fontsize=12)
        ax2.set_ylabel('Total Amount', fontsize=12)
        ax2.set_title('Total Amount by Top N', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.ticklabel_format(style='plain', axis='y')
        ax2.set_xscale('log')
        
        # Plot 3: Average Similarity Score by Top N
        ax3 = axes[1, 0]
        for cluster_id in results_df['cluster_id'].unique():
            cluster_data = results_df[results_df['cluster_id'] == cluster_id]
            ax3.plot(cluster_data['top_n'], 
                    cluster_data['avg_similarity_score'],
                    marker='^', linewidth=2, markersize=8, label=f'Cluster {cluster_id}')
        
        ax3.set_xlabel('Top N Transactions', fontsize=12)
        ax3.set_ylabel('Average Similarity Score', fontsize=12)
        ax3.set_title('Average Similarity Score by Top N', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot 4: Comparison Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for _, row in results_df.iterrows():
            table_data.append([
                f"Cluster {int(row['cluster_id'])}",
                f"Top {int(row['top_n'])}",
                f"{int(row['transaction_count']):,}",
                f"{row['total_amount']:,.0f}",
                f"{row['avg_similarity_score']:.3f}",
                f"{row['min_similarity_score']:.3f}",
                f"{row['max_similarity_score']:.3f}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Cluster', 'Top N', 'Count', 'Amount', 'Avg', 'Min', 'Max'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 2)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(7):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax4.set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'cluster_top_n_similarity_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"💾 Visualization saved to: {plot_path}")
    
    def run(self, start_date: str, end_date: str, 
            target_clusters: list = [6, 7],
            top_n_values: list = [100, 500, 1000, 5000, 10000],
            output_dir: str = '/root/research-dir/dev/jazzcash-fraud-detection/analysis/clustering'):
        """Execute top N similarity analysis."""
        try:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize
            self.initialize_spark()
            self.load_model()
            
            # Load data
            df = self.load_data(start_date, end_date)
            
            # Calculate similarity scores
            df_with_scores = self.calculate_similarity_scores(df)
            
            # Analyze top N by similarity
            results_df = self.analyze_top_n_by_similarity(df_with_scores, target_clusters, top_n_values)
            
            # Save results to CSV
            csv_path = os.path.join(output_dir, 'cluster_top_n_similarity_results.csv')
            results_df.to_csv(csv_path, index=False)
            self.logger.info(f"\n💾 Results saved to: {csv_path}")
            
            # Create visualizations
            self.create_visualizations(results_df, output_dir)
            
            # Print summary table
            self.logger.info("\n" + "=" * 80)
            self.logger.info("FINAL SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info("\n" + results_df.to_string(index=False))
            self.logger.info("\n" + "=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze top N most similar non-fraud transactions by cluster'
    )
    parser.add_argument('--model-path', type=str, 
                       default='/root/research-dir/dev/jazzcash-fraud-detection/models/fraud_clustering_kmeans_model',
                       help='Path to trained model')
    parser.add_argument('--start-date', type=str, default='2025-07-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-07-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--clusters', type=int, nargs='+', default=[6, 7],
                       help='Target cluster IDs (default: 6 7)')
    parser.add_argument('--top-n-values', type=int, nargs='+', default=[100, 500, 1000, 5000, 10000],
                       help='Top N values to analyze (default: 100 500 1000 5000 10000)')
    
    args = parser.parse_args()
    
    analyzer = ClusterSimilarityAnalyzer(args.model_path)
    success = analyzer.run(args.start_date, args.end_date, args.clusters, args.top_n_values)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
