#!/usr/bin/env python3
"""
Cosine Similarity Fraud Analysis
=================================
Calculates cosine similarity between the average fraud vector (centroid)
and a specific transaction.

Usage:
    python cosine_similarity_fraud_analysis_fraud_case_2.py --trans-id 89942951756
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
    
    def load_fraud_data(self) -> DataFrame:
        """Load all fraud transactions from ClickHouse."""
        self.logger.info("\n📥 Loading fraud transactions...")
        
        query = """
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE fraud_flag = 1 and data_date between '2025-01-01' and '2025-06-30'
        """
        
        df = self.spark.sql(query)
        # count = df.count()
        # self.logger.info(f"✅ Loaded {count:,} fraud transactions")
        
        return df
    
    def load_transaction_by_id(self, trans_id: str) -> DataFrame:
        """Load a specific transaction by ID from ClickHouse."""
        self.logger.info(f"\n📥 Loading transaction ID: {trans_id}...")
        
        query = f"""
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE trans_id = '{trans_id}'
        """
        
        df = self.spark.sql(query)
        # count = df.count()
        
        # if count == 0:
        #     self.logger.error(f"❌ Transaction ID '{trans_id}' not found in the dataset")
        #     return None
        
        self.logger.info(f"✅ Found transaction ID: {trans_id}")
        return df
    
    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Features are already engineered in stixor_fraud_features table."""
        self.logger.info("\n✅ Using pre-engineered features from stixor_fraud_features table")
        return df
    
    def prepare_feature_vectors(self, df: DataFrame, scaler_model=None) -> tuple:
        """Prepare feature vectors for similarity calculation.
        
        Args:
            df: Input DataFrame
            scaler_model: Optional pre-fitted scaler model. If None, fits a new scaler.
            
        Returns:
            tuple: (transformed_df, scaler_model)
        """
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
        if scaler_model is None:
            self.logger.info("⏳ Fitting new scaler...")
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=True
            )
            scaler_model = scaler.fit(df)
        else:
            self.logger.info("⏳ Applying existing scaler...")
        
        df = scaler_model.transform(df)
        
        self.logger.info("✅ Feature vectors prepared and scaled")
        
        return df, scaler_model
    
    def calculate_cosine_similarity(self, df: DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Calculate cosine similarity to fraud centroid (most efficient approach)."""
        self.logger.info(f"\n🔍 Calculating cosine similarity to fraud centroid (top {top_n})...")
        
        from pyspark.ml.stat import Summarizer
        
        # Separate fraud and non-fraud
        fraud_features = df.filter(col('fraud_flag') == 1).select('features')
        non_fraud_df = df.filter(col('fraud_flag') == 0).limit(10000)
        
        # Calculate fraud centroid (mean vector of all fraud transactions)
        self.logger.info("⏳ Computing fraud centroid...")
        fraud_centroid = fraud_features.select(
            Summarizer.mean(col('features')).alias('centroid')
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
        
        # Calculate similarity for all non-fraud transactions
        self.logger.info("⏳ Computing cosine similarities...")
        non_fraud_with_similarity = non_fraud_df.withColumn(
            'cosine_similarity',
            cosine_udf(col('features'))
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
        fraud_features.unpersist()
        non_fraud_df.unpersist()
        
        
        return result_df
    
    def calculate_similarity_for_transaction(self, trans_id: str) -> pd.DataFrame:
        """Calculate cosine similarity between fraud centroid and a specific transaction."""
        self.logger.info(f"\n🔍 Calculating cosine similarity for transaction ID: {trans_id}")
        
        from pyspark.ml.stat import Summarizer
        
        # Load fraud transactions
        fraud_df = self.load_fraud_data()
        if fraud_df is None:
            return pd.DataFrame()
        
        # Load the specific transaction
        target_df = self.load_transaction_by_id(trans_id)
        if target_df is None:
            return pd.DataFrame()
        
        # Assemble raw features first (before scaling)
        self.logger.info("\n📐 Assembling raw features...")
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
        
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        fraud_df = assembler.transform(fraud_df)
        target_df = assembler.transform(target_df)
        
        # Calculate fraud centroid BEFORE scaling (on raw features)
        self.logger.info("⏳ Computing fraud centroid from RAW features...")
        fraud_centroid = fraud_df.select(
            Summarizer.mean(col('features_raw')).alias('centroid')
        ).collect()[0]['centroid']
        
        # Convert centroid to numpy array
        centroid_array = fraud_centroid.toArray()
        centroid_norm = float(np.linalg.norm(centroid_array))
        
        self.logger.info(f"   Centroid norm: {centroid_norm:.6f}")
        self.logger.info(f"   Centroid dimensions: {len(centroid_array)}")
        
        # Define UDF for cosine similarity with centroid (using RAW features)
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
        
        # Calculate similarity for the specific transaction using RAW features
        self.logger.info("⏳ Computing cosine similarity for target transaction...")
        target_with_similarity = target_df.withColumn(
            'cosine_similarity',
            cosine_udf(col('features_raw'))
        )
        
        # Convert to pandas
        result_df = target_with_similarity.toPandas()
        
        # Display the result
        self.logger.info("\n" + "=" * 100)
        self.logger.info(f"COSINE SIMILARITY RESULT FOR TRANSACTION: {trans_id}")
        self.logger.info("=" * 100)
        
        if not result_df.empty:
            row = result_df.iloc[0]
            self.logger.info(f"\n📊 Transaction Details:")
            self.logger.info(f"   Transaction ID: {row['trans_id']}")
            self.logger.info(f"   Sender: {row.get('sender_mobile_no', 'N/A')}")
            self.logger.info(f"   Receiver: {row.get('receiver_mobile_no', 'N/A')}")
            self.logger.info(f"   Amount: {row['trx_amt']:,.2f}")
            self.logger.info(f"   Channel: {row.get('trx_channel', 'N/A')}")
            self.logger.info(f"   Type: {row.get('trx_type', 'N/A')}")
            self.logger.info(f"   Date: {row.get('cutoff_date', 'N/A')}")
            self.logger.info(f"   Fraud Flag: {row.get('fraud_flag', 'N/A')}")
            self.logger.info(f"\n🎯 COSINE SIMILARITY TO FRAUD CENTROID: {row['cosine_similarity']:.6f}")
            
            # Interpret the similarity score
            similarity = row['cosine_similarity']
            if similarity >= 0.9:
                self.logger.info("   ⚠️  Very High similarity to fraud pattern!")
            elif similarity >= 0.8:
                self.logger.info("   ⚠️  High similarity to fraud pattern")
            elif similarity >= 0.6:
                self.logger.info("   ⚡ Moderate similarity to fraud pattern")
            elif similarity >= 0.4:
                self.logger.info("   ℹ️  Low similarity to fraud pattern")
            else:
                self.logger.info("   ✅ Very low similarity to fraud pattern")
        
        self.logger.info("\n" + "=" * 100)
        
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
    
    def run(self, trans_id: str):
        """Execute cosine similarity analysis for a specific transaction."""
        try:
            # Initialize
            self.initialize_spark()
            
            # Calculate similarity for a specific transaction
            results_df = self.calculate_similarity_for_transaction(trans_id)
            
            if not results_df.empty:
                output_path = f'/root/research-dir/dev/jazzcash-fraud-detection/scripts/cosine_similarity_trans_{trans_id}.csv'
                results_df.to_csv(output_path, index=False)
                self.logger.info(f"\n💾 Results saved to: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate cosine similarity between fraud centroid and a specific transaction'
    )
    parser.add_argument('--trans-id', type=str, required=True, default="89942951756",
                       help='Transaction ID to check similarity against fraud centroid (e.g., 89942951756)')
    
    args = parser.parse_args()
    
    analyzer = CosineSimilarityFraudAnalyzer()
    success = analyzer.run(args.trans_id)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
