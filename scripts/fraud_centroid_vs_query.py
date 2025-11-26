#!/usr/bin/env python3
"""
Fraud Centroid vs Query - Cosine Similarity Analysis
=====================================================
Calculate cosine similarity between a fraud centroid (computed from transactions
within a date range) and transactions matching a custom SQL query.

Usage Examples:
    # Analyze all non-fraud transactions in September
    python fraud_centroid_vs_query.py \
        --fraud-start 2025-07-01 --fraud-end 2025-08-31 \
        --query "fraud_flag = 0 AND cutoff_date >= '2025-09-01' AND cutoff_date <= '2025-09-30'" \
        --top-n 100

    # Analyze specific channel/type combinations
    python fraud_centroid_vs_query.py \
        --fraud-start 2025-07-01 --fraud-end 2025-08-31 \
        --query "trx_channel = 'NEW_JC_APP' AND trx_type = 'Transfer(C2B)' AND trx_amt > 50000" \
        --top-n 50

    # Analyze high-value transactions
    python fraud_centroid_vs_query.py \
        --fraud-start 2025-07-01 --fraud-end 2025-08-31 \
        --query "trx_amt > 100000 AND fraud_flag = 0" \
        --top-n 200
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Summarizer
from pyspark.sql.types import DoubleType

# Fix PySpark Python version
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'


class FraudCentroidVsQuery:
    """Calculate cosine similarity between fraud centroid and query-matched transactions."""
    
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
            .appName("FraudCentroidVsQuery")
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
        """Load fraud transactions within date range for centroid calculation."""
        self.logger.info(f"\n📥 Loading fraud transactions ({start_date} to {end_date})...")
        
        query = f"""
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE fraud_flag = 1
        AND cutoff_date >= '{start_date}' 
        AND cutoff_date <= '{end_date}'
        """
        
        return self.spark.sql(query)
    
    def load_query_transactions(self, where_clause):
        """Load transactions matching the provided WHERE clause."""
        self.logger.info(f"\n📥 Loading transactions matching query...")
        self.logger.info(f"   Query: {where_clause}")
        
        query = f"""
        SELECT *
        FROM clickhouse.public.stixor_fraud_features_distributed
        WHERE {where_clause}
        """
        
        return self.spark.sql(query)
    
    def vectorize_and_scale(self, df):
        """Convert features to vectors and scale."""
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
    
    def compute_cosine_similarities(self, target_df, centroid_array, centroid_norm, top_n=100):
        """Compute cosine similarities and return top N results."""
        self.logger.info(f"⏳ Computing cosine similarities (top {top_n})...")
        
        def cosine_sim(features):
            if features is None:
                return 0.0
            arr = features.toArray()
            norm = np.linalg.norm(arr)
            if norm == 0 or centroid_norm == 0:
                return 0.0
            return float(np.dot(arr, centroid_array) / (norm * centroid_norm))
        
        from pyspark.sql.functions import udf
        
        cosine_udf = udf(cosine_sim, DoubleType())
        result_df = target_df.withColumn('cosine_similarity', cosine_udf(col('features')))
        
        # Get top N by similarity
        top_results = result_df.orderBy(col('cosine_similarity').desc()).limit(top_n)
        
        return top_results.toPandas()
    
    def display_results(self, result_df, top_n):
        """Display analysis results."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info(f"TOP {top_n} TRANSACTIONS BY COSINE SIMILARITY TO FRAUD CENTROID")
        self.logger.info("=" * 100)
        
        if result_df.empty:
            self.logger.warning("❌ No results to display")
            return
        
        # Summary statistics
        self.logger.info(f"\n📊 Summary Statistics:")
        self.logger.info(f"   Total Transactions Analyzed: {len(result_df):,}")
        self.logger.info(f"   Mean Similarity: {result_df['cosine_similarity'].mean():.6f}")
        self.logger.info(f"   Median Similarity: {result_df['cosine_similarity'].median():.6f}")
        self.logger.info(f"   Max Similarity: {result_df['cosine_similarity'].max():.6f}")
        self.logger.info(f"   Min Similarity: {result_df['cosine_similarity'].min():.6f}")
        self.logger.info(f"   Std Dev: {result_df['cosine_similarity'].std():.6f}")
        
        # Show top 10 transactions
        display_count = min(10, len(result_df))
        self.logger.info(f"\n🔍 Top {display_count} Most Similar Transactions:")
        self.logger.info("-" * 100)
        
        for idx, row in result_df.head(display_count).iterrows():
            similarity = row['cosine_similarity']
            
            # Similarity indicator
            if similarity >= 0.9:
                indicator = "⚠️  VERY HIGH"
            elif similarity >= 0.7:
                indicator = "⚠️  HIGH"
            elif similarity >= 0.5:
                indicator = "⚡ MODERATE"
            elif similarity >= 0.3:
                indicator = "ℹ️  LOW"
            else:
                indicator = "✅ VERY LOW"
            
            self.logger.info(f"\n#{idx + 1} | Similarity: {similarity:.6f} {indicator}")
            self.logger.info(f"   Trans ID: {row['trans_id']}")
            self.logger.info(f"   Amount: {row['trx_amt']:,.2f} | Channel: {row.get('trx_channel', 'N/A')} | Type: {row.get('trx_type', 'N/A')}")
            self.logger.info(f"   Date: {row.get('cutoff_date', 'N/A')} | Fraud Flag: {row.get('fraud_flag', 'N/A')}")
        
        self.logger.info("\n" + "=" * 100)
    
    def save_results(self, result_df, output_path):
        """Save results to CSV."""
        result_df.to_csv(output_path, index=False)
        self.logger.info(f"\n💾 Full results saved to: {output_path}")
    
    def run(self, fraud_start_date, fraud_end_date, where_clause, top_n=100, output_dir=None):
        """Execute the analysis."""
        try:
            self.initialize_spark()
            
            # Load fraud transactions for centroid
            fraud_df = self.load_fraud_transactions(fraud_start_date, fraud_end_date)
            fraud_count = fraud_df.count()
            self.logger.info(f"✅ Loaded {fraud_count:,} fraud transactions for centroid")
            
            if fraud_count == 0:
                self.logger.error("❌ No fraud transactions found in date range!")
                return False
            
            # Load target transactions
            target_df = self.load_query_transactions(where_clause)
            target_count = target_df.count()
            self.logger.info(f"✅ Loaded {target_count:,} transactions matching query")
            
            if target_count == 0:
                self.logger.error("❌ No transactions found matching query!")
                return False
            
            # Vectorize and scale
            self.logger.info(f"\n📐 Vectorizing {len(self.FEATURE_COLS)} features...")
            fraud_df, scaler_model = self.vectorize_and_scale(fraud_df)
            
            # Apply same scaler to target transactions
            target_df = scaler_model.transform(
                VectorAssembler(inputCols=self.FEATURE_COLS, outputCol="features_raw", 
                               handleInvalid="skip").transform(target_df)
            )
            
            # Calculate fraud centroid
            centroid_array, centroid_norm = self.calculate_fraud_centroid(fraud_df)
            
            # Compute similarities
            result_df = self.compute_cosine_similarities(target_df, centroid_array, centroid_norm, top_n)
            
            # Display results
            self.display_results(result_df, top_n)
            
            # Save if requested
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f'fraud_centroid_similarity_{timestamp}.csv')
                self.save_results(result_df, output_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error: {str(e)}", exc_info=True)
            return False
        
        finally:
            if self.spark:
                self.spark.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate cosine similarity between fraud centroid and query-matched transactions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze non-fraud transactions in September
  %(prog)s --fraud-start 2025-07-01 --fraud-end 2025-08-31 \\
           --query "fraud_flag = 0 AND cutoff_date >= '2025-09-01' AND cutoff_date <= '2025-09-30'" \\
           --top-n 100

  # Analyze high-value NEW_JC_APP transactions
  %(prog)s --fraud-start 2025-07-01 --fraud-end 2025-08-31 \\
           --query "trx_channel = 'NEW_JC_APP' AND trx_amt > 50000" \\
           --top-n 50
        """
    )
    parser.add_argument('--fraud-start', type=str, required=True,
                       help='Start date for fraud centroid calculation (YYYY-MM-DD)')
    parser.add_argument('--fraud-end', type=str, required=True,
                       help='End date for fraud centroid calculation (YYYY-MM-DD)')
    parser.add_argument('--query', type=str, required=True,
                       help='SQL WHERE clause to filter transactions to analyze')
    parser.add_argument('--top-n', type=int, default=100,
                       help='Number of top similar transactions to return (default: 100)')
    parser.add_argument('--output-dir', type=str,
                       default='/root/research-dir/dev/jazzcash-fraud-detection/analysis/cosine_similarity',
                       help='Output directory for results CSV')
    
    args = parser.parse_args()
    
    analyzer = FraudCentroidVsQuery()
    success = analyzer.run(
        args.fraud_start, 
        args.fraud_end, 
        args.query, 
        args.top_n,
        args.output_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
