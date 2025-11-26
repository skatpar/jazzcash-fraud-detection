#!/usr/bin/env python3
"""
Cosine Similarity Fraud Analysis
=================================
Calculate cosine similarity between a specific transaction and fraud transactions
to identify the most similar fraud cases to the given transaction.

Usage:
    python cosine_similarity_fraud_analysis_fraud_case.py --trans-id 89942951756 --top-n 10
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
    
    def load_data(self, trans_id: int) -> DataFrame:
        """Load transaction data with features from ClickHouse."""
        self.logger.info(f"\n📥 Loading data for transaction {trans_id} and all fraud cases...")
        
        query = f"""
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE trans_id = '{trans_id}' OR fraud_flag = 1 
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
    
    def calculate_cosine_similarity(self, df: DataFrame, trans_id: int, top_n: int = 10) -> pd.DataFrame:
        """Calculate cosine similarity between specific transaction and all fraud cases."""
        self.logger.info(f"\n🔍 Calculating cosine similarity between transaction {trans_id} and fraud cases (top {top_n})...")
        
        # Get the specific transaction
        target_transaction = df.filter(col('trans_id') == trans_id).select('features', 'trans_id').first()
        
        if target_transaction is None:
            self.logger.error(f"❌ Transaction {trans_id} not found!")
            return pd.DataFrame()
        
        target_vector = target_transaction['features']
        target_array = target_vector.toArray()
        target_norm = float(np.linalg.norm(target_array))
        
        self.logger.info(f"   Target transaction vector norm: {target_norm:.6f}")
        self.logger.info(f"   Target vector dimensions: {len(target_array)}")
        
        # Get all fraud transactions
        fraud_df = df.filter(col('fraud_flag') == 1)
        
        # self.logger.info(f"   Comparing against {fraud_df.count()} fraud transactions...")
        
        # Define efficient UDF for cosine similarity with target transaction
        def cosine_with_target(features):
            """Calculate cosine similarity with target transaction."""
            if features is None:
                return 0.0
            
            arr = features.toArray()
            norm = np.linalg.norm(arr)
            
            if norm == 0 or target_norm == 0:
                return 0.0
            
            dot_product = np.dot(arr, target_array)
            return float(dot_product / (norm * target_norm))
        
        cosine_udf = F.udf(cosine_with_target, DoubleType())
        
        # Calculate similarity for all fraud transactions
        self.logger.info("⏳ Computing cosine similarities...")
        fraud_with_similarity = fraud_df.withColumn(
            'cosine_similarity',
            cosine_udf(col('features'))
        )
        
        # Get top N most similar fraud transactions
        top_similar = fraud_with_similarity.orderBy(
            col('cosine_similarity').desc()
        ).limit(top_n)
        
        # Select relevant columns
        result_df = top_similar.toPandas()
        
        # Unpersist
        fraud_df.unpersist()
        
        return result_df
            
    def display_results(self, results_df: pd.DataFrame):
        """Display analysis results."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("TOP FRAUD-SIMILAR NON-FRAUD TRANSACTIONS (Cosine Similarity to Fraud Centroid)")
        self.logger.info("=" * 100)
        
        for idx, row in results_df.iterrows():
            self.logger.info(f"\n🔹 Rank {idx + 1}:")
            self.logger.info(f"   Transaction ID: {row['trans_id']}")
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
    
    def run(self, trans_id: int, top_n: int = 10,
            output_dir: str = '/root/research-dir/dev/jazzcash-fraud-detection/analysis/cosine_similarity'):
        """Execute cosine similarity analysis."""
        try:
            # Initialize
            self.initialize_spark()
            
            # Load data
            df = self.load_data(trans_id)            
            
            # Prepare feature vectors
            df = self.prepare_feature_vectors(df)
            
            # Calculate cosine similarity
            results_df = self.calculate_cosine_similarity(df, trans_id, top_n)
            results_df.to_csv('/root/research-dir/dev/jazzcash-fraud-detection/scripts/cosine_similarity_results_fraud_case.csv', index=False)   
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate cosine similarity between a specific transaction and fraud transactions'
    )
    parser.add_argument('--trans-id', type=int, default=89942951756,
                       help='Transaction ID to compare against fraud cases (default: 89942951756)')
    parser.add_argument('--top-n', type=int, default=10000,
                       help='Number of top similar transactions to return (default: 10)')
    parser.add_argument('--output-dir', type=str,
                       default='/root/research-dir/dev/jazzcash-fraud-detection/analysis/cosine_similarity',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    analyzer = CosineSimilarityFraudAnalyzer()
    success = analyzer.run(args.trans_id, args.top_n, args.output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
