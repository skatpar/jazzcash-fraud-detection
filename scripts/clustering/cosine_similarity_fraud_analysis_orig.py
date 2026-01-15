#!/usr/bin/env python3
"""
Cosine Similarity Fraud Analysis
=================================
Calculates cosine similarity between fraud and non-fraud transactions
to identify the top non-fraud transactions most similar to fraud patterns.

Usage:
    python cosine_similarity_fraud_analysis.py --start-date 2025-07-01 --end-date 2025-07-31 --top-n 10
"""

import os
import sys
import argparse
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg as spark_avg,
    when, lit, sqrt, pow as spark_pow, expr
)
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.sql.functions as F

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'


class CosineSimilarityFraudAnalyzer:
    """Analyze fraud patterns using cosine similarity."""
    
    def __init__(self):
        self.spark = None
        self.logger = self._setup_logger()
        self.feature_cols = []
        
    def _setup_logger(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def initialize_spark(self):
        """Initialize Spark session."""
        self.logger.info("🚀 Initializing Spark session...")
        
        packages = [
            "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
            "com.clickhouse:clickhouse-client:0.9.4",
            "com.clickhouse:clickhouse-http-client:0.9.4",
            "org.apache.httpcomponents.client5:httpclient5:5.2.1"
        ]
        
        self.spark = (SparkSession.builder
            .appName("CosineSimilarityFraudAnalysis")
            .master("spark://10.205.161.118:7077")
            # .master("local[*]")
            .config("spark.jars.packages", ",".join(packages))
            .config("spark.executor.memory", "150g")
            .config("spark.executor.memoryOverhead", "5g")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.cores", "32")
            .config("spark.executor.instances", "2")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", "96")
            .getOrCreate()
        )
        
        # Configure ClickHouse catalog
        self.spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
        self.spark.conf.set("spark.sql.catalog.clickhouse.host", "localhost")
        self.spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
        self.spark.conf.set("spark.sql.catalog.clickhouse.http_port", "8123")
        self.spark.conf.set("spark.sql.catalog.clickhouse.user", "default")
        self.spark.conf.set("spark.sql.catalog.clickhouse.password", "DfsTeChB1")
        self.spark.conf.set("spark.sql.catalog.clickhouse.database", "public")
        self.spark.conf.set("spark.clickhouse.write.format", "json")
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")
        self.logger.info(f"   Master: {self.spark.sparkContext.master}")
        self.logger.info(f"   Executor Memory: 150g")
        self.logger.info(f"   Executor Cores: 32")
        self.logger.info(f"   Executor Instances: 2")
        self.logger.info(f"   ClickHouse catalog: clickhouse.public")
    
    def load_data(self, start_date: str, end_date: str) -> DataFrame:
        """Load transaction data with features from ClickHouse."""
        self.logger.info(f"\n📥 Loading data from {start_date} to {end_date}...")
        
        query = f"""
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE (cutoff_date >= '{start_date}' AND cutoff_date <= '{end_date}'
        AND mbar_account_type_name = 'Customer Account') OR fraud_flag = 1
        """
        
        df = self.spark.sql(query)
        
        return df
    
    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Features are already engineered in stixor_fraud_features table."""
        self.logger.info("\n✅ Using pre-engineered features from stixor_fraud_features table")
        return df
    
    def prepare_feature_vectors(self, df: DataFrame) -> DataFrame:
        """Prepare feature vectors for similarity calculation."""
        self.logger.info("\n📐 Preparing feature vectors...")
        
        # Use specific selected features (excluding non-numeric identifiers)
        self.feature_cols = [
            'start_balance', 'trx_amt',
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
        
        self.logger.info(f"📊 Using {len(self.feature_cols)} selected features")
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        df = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        
        self.logger.info("✅ Feature vectors prepared and scaled")
        
        return df
    
    def calculate_cosine_similarity(self, df: DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Calculate cosine similarity to fraud centroid (most efficient approach)."""
        self.logger.info(f"\n🔍 Calculating cosine similarity to fraud centroid (top {top_n})...")
        
        from pyspark.ml.stat import Summarizer
        
        # Separate fraud and non-fraud - use RAW features before scaling
        fraud_raw = df.filter(col('fraud_flag') == 1).select('features_raw')
        non_fraud_df = df.filter(col('fraud_flag') == 0).limit(100)
        
        # Calculate fraud centroid from RAW features (before scaling)
        self.logger.info("⏳ Computing fraud centroid from RAW features...")
        fraud_centroid = fraud_raw.select(
            Summarizer.mean(col('features_raw')).alias('centroid')
        ).collect()[0]['centroid']
        
        
        # Convert centroid to numpy array
        centroid_array = fraud_centroid.toArray()
        centroid_norm = float(np.linalg.norm(centroid_array))
        
        self.logger.info(f"   Centroid norm: {centroid_norm:.6f}")
        self.logger.info(f"   Centroid dimensions: {len(centroid_array)}")
        
        # Define efficient UDF for cosine similarity with centroid
        def cosine_with_centroid(features):
            """Calculate cosine similarity with fraud centroid."""
            if features is None:
                return 0.0
            
            arr = features.toArray()
            norm = np.linalg.norm(arr)
            
            if norm == 0 or centroid_norm == 0:
                return 0.0
            
            dot_product = np.dot(arr, centroid_array)
            return float(dot_product / (norm * centroid_norm))
        
        cosine_udf = F.udf(cosine_with_centroid, DoubleType())
        
        # Calculate similarity for all non-fraud transactions using RAW features
        self.logger.info("⏳ Computing cosine similarities...")
        non_fraud_with_similarity = non_fraud_df.withColumn(
            'cosine_similarity',
            cosine_udf(col('features_raw'))
        )
        
        # Get top N most similar to fraud centroid
        top_similar = non_fraud_with_similarity.filter(
            col('cosine_similarity') >= 0.8
        ).orderBy(
            col('cosine_similarity').desc()
        ).limit(top_n)
        
        # Select relevant columns
        result_df = top_similar.toPandas()
        
        # Unpersist
        fraud_raw.unpersist()
        non_fraud_df.unpersist()
        
        
        return result_df
            
    def display_results(self, results_df: pd.DataFrame):
        """Display analysis results."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("TOP FRAUD-SIMILAR NON-FRAUD TRANSACTIONS (Cosine Similarity to Fraud Centroid)")
        self.logger.info("=" * 100)
        
        for idx, row in results_df.iterrows():
            self.logger.info(f"\n🔹 Rank {idx + 1}:")
            self.logger.info(f"   Transaction ID: {row['trx_id']}")
            self.logger.info(f"   Sender: {row['sender_mobile_no']}")
            self.logger.info(f"   Receiver: {row['receiver_mobile_no']}")
            self.logger.info(f"   Amount: {row['trx_amt']:,.2f}")
            self.logger.info(f"   Channel: {row['trx_channel']}")
            self.logger.info(f"   Type: {row['trx_type']}")
            self.logger.info(f"   Date: {row['cutoff_date']}")
            self.logger.info(f"   Cosine Similarity: {row['cosine_similarity']:.6f}")
        
        self.logger.info("\n" + "=" * 100)
        
        # Summary statistics
        self.logger.info("\n📊 Summary Statistics:")
        self.logger.info(f"   Mean Similarity: {results_df['cosine_similarity'].mean():.6f}")
        self.logger.info(f"   Median Similarity: {results_df['cosine_similarity'].median():.6f}")
        self.logger.info(f"   Min Similarity: {results_df['cosine_similarity'].min():.6f}")
        self.logger.info(f"   Max Similarity: {results_df['cosine_similarity'].max():.6f}")
        self.logger.info(f"   Std Dev: {results_df['cosine_similarity'].std():.6f}")
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str):
        """Save results to CSV."""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f'cosine_similarity_top_transactions_{timestamp}.csv')
        
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"\n💾 Results saved to: {csv_path}")
    
    def run(self, start_date: str, end_date: str, top_n: int = 10,
            output_dir: str = '/root/research-dir/dev/jazzcash-fraud-detection/analysis/cosine_similarity'):
        """Execute cosine similarity analysis."""
        try:
            # Initialize
            self.initialize_spark()
            
            # Load data
            df = self.load_data(start_date, end_date)            
            
            # Prepare feature vectors
            df = self.prepare_feature_vectors(df)
            
            # Calculate cosine similarity
            results_df = self.calculate_cosine_similarity(df, top_n)
            results_df.to_csv('/root/research-dir/dev/jazzcash-fraud-detection/scripts/cosine_similarity_results.csv', index=False)   
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate cosine similarity between fraud and non-fraud transactions'
    )
    parser.add_argument('--start-date', type=str, default='2025-06-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-06-30',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=100000,
                       help='Number of top similar transactions to return (default: 10)')
    parser.add_argument('--output-dir', type=str,
                       default='/root/research-dir/dev/jazzcash-fraud-detection/analysis/cosine_similarity',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    analyzer = CosineSimilarityFraudAnalyzer()
    success = analyzer.run(args.start_date, args.end_date, args.top_n, args.output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
