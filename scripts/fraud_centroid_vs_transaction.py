#!/usr/bin/env python3
"""
Fraud Centroid vs Transaction - Cosine Similarity Analysis
===========================================================
Calculate cosine similarity between a fraud centroid (computed from transactions
within a date range) and a specific transaction ID.

Usage:
    python fraud_centroid_vs_transaction.py --trans-id 89942951756 --start-date 2025-07-01 --end-date 2025-09-30
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Summarizer

# Fix PySpark Python version
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'


class FraudCentroidVsTransaction:
    """Calculate cosine similarity between fraud centroid and a specific transaction."""
    
    FEATURE_COLS = [
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
    
    def __init__(self):
        self.spark = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def initialize_spark(self):
        """Initialize Spark session with ClickHouse."""
        self.logger.info("🚀 Initializing Spark session...")
        
        packages = [
            "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
            "com.clickhouse:clickhouse-client:0.9.4",
            "com.clickhouse:clickhouse-http-client:0.9.4",
            "org.apache.httpcomponents.client5:httpclient5:5.2.1"
        ]
        
        self.spark = (SparkSession.builder
            .appName("FraudCentroidVsTransaction")
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
        
        # Configure ClickHouse
        self.spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
        self.spark.conf.set("spark.sql.catalog.clickhouse.host", "localhost")
        self.spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
        self.spark.conf.set("spark.sql.catalog.clickhouse.http_port", "8123")
        self.spark.conf.set("spark.sql.catalog.clickhouse.user", "default")
        self.spark.conf.set("spark.sql.catalog.clickhouse.password", "DfsTeChB1")
        self.spark.conf.set("spark.sql.catalog.clickhouse.database", "public")
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.logger.info(f"✅ Spark initialized (Version: {self.spark.version})")
    
    def load_fraud_transactions(self, start_date, end_date):
        """Load fraud transactions within date range."""
        self.logger.info(f"\n📥 Loading fraud transactions ({start_date} to {end_date})...")
        
        query = f"""
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE fraud_flag = 1
        AND cutoff_date >= '{start_date}' 
        AND cutoff_date <= '{end_date}'
        """
        
        return self.spark.sql(query)
    
    def load_transaction(self, trans_id):
        """Load specific transaction by ID."""
        self.logger.info(f"📥 Loading transaction {trans_id}...")
        
        query = f"""
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE trans_id = '{trans_id}'
        """
        
        df = self.spark.sql(query)
        count = df.count()
        
        if count == 0:
            self.logger.error(f"❌ Transaction {trans_id} not found!")
            return None
            
        return df
    
    def vectorize_and_scale(self, df):
        """Convert features to vectors and scale."""
        # Assemble features
        assembler = VectorAssembler(
            inputCols=self.FEATURE_COLS,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        df = assembler.transform(df)
        
        # Scale (without centering to preserve centroid meaning)
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=False
        )
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        
        return df, scaler_model
    
    def calculate_fraud_centroid(self, fraud_df):
        """Calculate the mean vector (centroid) of fraud transactions."""
        self.logger.info("⏳ Computing fraud centroid...")
        
        fraud_features = fraud_df.select('features')
        centroid = fraud_features.select(
            Summarizer.mean(col('features')).alias('centroid')
        ).collect()[0]['centroid']
        
        centroid_array = centroid.toArray()
        centroid_norm = float(np.linalg.norm(centroid_array))
        
        self.logger.info(f"   Centroid norm: {centroid_norm:.6f}")
        self.logger.info(f"   Centroid dimensions: {len(centroid_array)}")
        
        return centroid_array, centroid_norm
    
    def compute_cosine_similarity(self, target_df, centroid_array, centroid_norm):
        """Compute cosine similarity between target transaction and fraud centroid."""
        self.logger.info("⏳ Computing cosine similarity...")
        
        def cosine_sim(features):
            if features is None:
                return 0.0
            arr = features.toArray()
            norm = np.linalg.norm(arr)
            if norm == 0 or centroid_norm == 0:
                return 0.0
            return float(np.dot(arr, centroid_array) / (norm * centroid_norm))
        
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        
        cosine_udf = udf(cosine_sim, DoubleType())
        result_df = target_df.withColumn('cosine_similarity', cosine_udf(col('features')))
        
        return result_df.toPandas()
    
    def display_results(self, result_df, trans_id):
        """Display analysis results."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info(f"COSINE SIMILARITY ANALYSIS FOR TRANSACTION: {trans_id}")
        self.logger.info("=" * 100)
        
        if result_df.empty:
            self.logger.warning("❌ No results to display")
            return
        
        row = result_df.iloc[0]
        similarity = row['cosine_similarity']
        
        self.logger.info(f"\n📊 Transaction Details:")
        self.logger.info(f"   Transaction ID: {row['trans_id']}")
        self.logger.info(f"   Amount: {row['trx_amt']:,.2f}")
        self.logger.info(f"   Channel: {row.get('trx_channel', 'N/A')}")
        self.logger.info(f"   Type: {row.get('trx_type', 'N/A')}")
        self.logger.info(f"   Date: {row.get('cutoff_date', 'N/A')}")
        self.logger.info(f"   Fraud Flag: {row.get('fraud_flag', 'N/A')}")
        
        self.logger.info(f"\n🎯 COSINE SIMILARITY TO FRAUD CENTROID: {similarity:.6f}")
        
        # Interpretation
        if similarity >= 0.9:
            self.logger.info("   ⚠️  VERY HIGH similarity to fraud pattern!")
        elif similarity >= 0.7:
            self.logger.info("   ⚠️  HIGH similarity to fraud pattern")
        elif similarity >= 0.5:
            self.logger.info("   ⚡ MODERATE similarity to fraud pattern")
        elif similarity >= 0.3:
            self.logger.info("   ℹ️  LOW similarity to fraud pattern")
        else:
            self.logger.info("   ✅ VERY LOW similarity to fraud pattern")
        
        self.logger.info("\n" + "=" * 100)
    
    def run(self, trans_id, start_date, end_date, output_path=None):
        """Execute the analysis."""
        try:
            self.initialize_spark()
            
            # Load fraud transactions for centroid
            fraud_df = self.load_fraud_transactions(start_date, end_date)
            fraud_count = fraud_df.count()
            self.logger.info(f"✅ Loaded {fraud_count:,} fraud transactions")
            
            if fraud_count == 0:
                self.logger.error("❌ No fraud transactions found in date range!")
                return False
            
            # Load target transaction
            target_df = self.load_transaction(trans_id)
            if target_df is None:
                return False
            
            # Vectorize and scale
            self.logger.info(f"\n📐 Vectorizing {len(self.FEATURE_COLS)} features...")
            fraud_df, scaler_model = self.vectorize_and_scale(fraud_df)
            target_df = scaler_model.transform(
                VectorAssembler(inputCols=self.FEATURE_COLS, outputCol="features_raw", 
                               handleInvalid="skip").transform(target_df)
            )
            
            # Calculate fraud centroid
            centroid_array, centroid_norm = self.calculate_fraud_centroid(fraud_df)
            
            # Compute similarity
            result_df = self.compute_cosine_similarity(target_df, centroid_array, centroid_norm)
            
            # Display results
            self.display_results(result_df, trans_id)
            
            # Save if requested
            if output_path:
                result_df.to_csv(output_path, index=False)
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
    parser.add_argument('--trans-id', type=str, required=True,
                       help='Transaction ID to analyze')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date for fraud centroid (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date for fraud centroid (YYYY-MM-DD)')
    parser.add_argument('--output', type=str,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    analyzer = FraudCentroidVsTransaction()
    success = analyzer.run(args.trans_id, args.start_date, args.end_date, args.output)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
