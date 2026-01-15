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
    
    def vectorize(self, df):
        """Convert features to vectors."""
        # Count rows before vectorization
        initial_count = df.count()
        
        assembler = VectorAssembler(
            inputCols=self.FEATURE_COLS,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        df = assembler.transform(df)
        
        # Count rows after vectorization and log if any were dropped
        final_count = df.count()
        if final_count < initial_count:
            dropped = initial_count - final_count
            self.logger.warning(f"⚠️  {dropped:,} rows dropped due to null/invalid feature values")
        
        return df
    
    def fit_scaler(self, combined_df):
        """Fit scaler on combined fraud + target data to avoid data leakage."""
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=False
        )
        scaler_model = scaler.fit(combined_df)
        return scaler_model
    
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
    
    def compute_cosine_similarities(self, target_df, centroid_array, centroid_norm):
        """Compute cosine similarities for all transactions."""
        self.logger.info("⏳ Computing cosine similarities...")
        
        # Broadcast centroid for better performance across executors
        centroid_broadcast = self.spark.sparkContext.broadcast(centroid_array)
        centroid_norm_broadcast = self.spark.sparkContext.broadcast(centroid_norm)
        
        def cosine_sim(features):
            if features is None:
                return 0.0
            arr = features.toArray()
            norm = np.linalg.norm(arr)
            c_norm = centroid_norm_broadcast.value
            if norm == 0 or c_norm == 0:
                return 0.0
            return float(np.dot(arr, centroid_broadcast.value) / (norm * c_norm))
        
        from pyspark.sql.functions import udf
        
        cosine_udf = udf(cosine_sim, DoubleType())
        result_df = target_df.withColumn('cosine_similarity', cosine_udf(col('features')))
        
        return result_df
    
    def display_results(self, result_df):
        """Display analysis results."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("COSINE SIMILARITY TO FRAUD CENTROID - ANALYSIS RESULTS")
        self.logger.info("=" * 100)
        
        # Calculate summary statistics using Spark
        from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max, count
        
        stats = result_df.select(
            count('cosine_similarity').alias('total_count'),
            mean('cosine_similarity').alias('mean_sim'),
            stddev('cosine_similarity').alias('std_sim'),
            spark_min('cosine_similarity').alias('min_sim'),
            spark_max('cosine_similarity').alias('max_sim')
        ).collect()[0]
        
        # Calculate median using approxQuantile
        median_sim = result_df.approxQuantile('cosine_similarity', [0.5], 0.01)[0]
        
        self.logger.info(f"\n📊 Summary Statistics:")
        self.logger.info(f"   Total Transactions Analyzed: {stats['total_count']:,}")
        self.logger.info(f"   Mean Similarity: {stats['mean_sim']:.6f}")
        self.logger.info(f"   Median Similarity: {median_sim:.6f}")
        self.logger.info(f"   Max Similarity: {stats['max_sim']:.6f}")
        self.logger.info(f"   Min Similarity: {stats['min_sim']:.6f}")
        self.logger.info(f"   Std Dev: {stats['std_sim']:.6f}")
        
        # Calculate distribution in 20% buckets
        self.logger.info(f"\n📈 Similarity Distribution (by 20% ranges):")
        self.logger.info("-" * 100)
        
        from pyspark.sql.functions import when
        
        # Count transactions in each similarity range
        # range_negative = result_df.filter((col('cosine_similarity') >= -1) & (col('cosine_similarity') < 0)).count()
        # range_0_20 = result_df.filter((col('cosine_similarity') >= 0.0) & (col('cosine_similarity') < 0.2)).count()
        # range_20_40 = result_df.filter((col('cosine_similarity') >= 0.2) & (col('cosine_similarity') < 0.4)).count()
        # range_40_60 = result_df.filter((col('cosine_similarity') >= 0.4) & (col('cosine_similarity') < 0.6)).count()
        # range_60_80 = result_df.filter((col('cosine_similarity') >= 0.6) & (col('cosine_similarity') < 0.8)).count()
        # range_80_100 = result_df.filter((col('cosine_similarity') >= 0.8) & (col('cosine_similarity') <= 1.0)).count()
        # range_0_50 = result_df.filter((col('cosine_similarity') >= 0.0) & (col('cosine_similarity') < 0.5)).count()
        range_80_100 = result_df.filter((col('cosine_similarity') >= 0.8) & (col('cosine_similarity') <= 1.0)).count()
        
        total = stats['total_count']
        
        def format_range(count, total):
            pct = (count / total * 100) if total > 0 else 0
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            return f"{count:>10,} ({pct:>6.2f}%) {bar}"
        
        # self.logger.info(f"\n   < 0%      (NEGATIVE):    {format_range(range_negative, total)}")
        # self.logger.info(f"\n   0% - 20%  (VERY LOW):    {format_range(range_0_20, total)}")
        # self.logger.info(f"  20% - 40%  (LOW):         {format_range(range_20_40, total)}")
        # self.logger.info(f"  40% - 60%  (MODERATE):    {format_range(range_40_60, total)}")
        # self.logger.info(f"  60% - 80%  (HIGH):        {format_range(range_60_80, total)}")
        # self.logger.info(f"  80% - 100% (VERY HIGH):   {format_range(range_80_100, total)}")
        # self.logger.info(f"\n   0% - 50%  (LOW-MEDIUM):  {format_range(range_0_50, total)}")
        self.logger.info(f"  80% - 100% (HIGH):        {format_range(range_80_100, total)}")

        # Verify counts add up
        # range_sum = range_negative + range_0_20 + range_20_40 + range_40_60 + range_60_80 + range_80_100
        # range_sum = range_negative + range_0_50 + range_50_100
        # self.logger.info(f"\n   Total (verification):     {range_sum:>10,} ({(range_sum/total*100):>6.2f}%)")
        
        self.logger.info("\n" + "=" * 100)
    
    def save_results(self, result_df, output_path):
        """Save results to CSV using Spark native writer to avoid OOM."""
        # Use Spark's native CSV writer instead of toPandas() to handle large datasets
        output_dir = output_path.replace('.csv', '_temp')
        
        result_df.select(
            'trans_id', 'cutoff_date', 'trx_amt', 'trx_channel', 'trx_type', 
            'fraud_flag', 'cosine_similarity'
        ).coalesce(1).write.mode('overwrite').option('header', 'true').csv(output_dir)
        
        # Rename the part file to the desired output filename
        import glob
        part_files = glob.glob(os.path.join(output_dir, 'part-*.csv'))
        if part_files:
            import shutil
            shutil.move(part_files[0], output_path)
            shutil.rmtree(output_dir)
            self.logger.info(f"\n💾 Full results saved to: {output_path}")
        else:
            self.logger.warning(f"\n⚠️  Results saved to directory: {output_dir}")
    
    def run(self, fraud_start_date, fraud_end_date, where_clause, output_dir=None):
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
            
            # Vectorize both dataframes
            self.logger.info(f"\n📐 Vectorizing {len(self.FEATURE_COLS)} features...")
            fraud_df = self.vectorize(fraud_df)
            target_df = self.vectorize(target_df)
            
            # Fit scaler on combined data to avoid data leakage
            self.logger.info("📏 Fitting scaler on combined fraud + target data...")
            combined_df = fraud_df.select('features_raw').union(target_df.select('features_raw'))
            scaler_model = self.fit_scaler(combined_df)
            
            # Apply scaler to both dataframes
            fraud_df = scaler_model.transform(fraud_df)
            target_df = scaler_model.transform(target_df)
            
            # Calculate fraud centroid
            centroid_array, centroid_norm = self.calculate_fraud_centroid(fraud_df)
            
            # Compute similarities
            result_df = self.compute_cosine_similarities(target_df, centroid_array, centroid_norm)
            
            # Display results
            self.display_results(result_df)
            
            # Save if requested
            # if output_dir:
            #     os.makedirs(output_dir, exist_ok=True)
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     output_path = os.path.join(output_dir, f'fraud_centroid_similarity_{timestamp}.csv')
            #     self.save_results(result_df, output_path)
            
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
    parser.add_argument('--fraud-start', type=str, default='2025-01-01',
                       help='Start date for fraud centroid calculation (YYYY-MM-DD)')
    parser.add_argument('--fraud-end', type=str, default='2025-06-30',
                       help='End date for fraud centroid calculation (YYYY-MM-DD)')
    parser.add_argument('--query', type=str, default="fraud_flag = 0 and data_date between '2025-7-1' and '2025-7-1' and hour(trans_initiate_time) between 6 and 12",
    # parser.add_argument('--query', type=str, default="fraud_flag = 0 AND data_date between '2025-07-01' and '2025-07-31' and trx_channel='NEW_JC_APP' and trx_type='Transfer(C2B)' and start_balance>=25000 and trx_amt>=50000",
                       help='SQL WHERE clause to filter transactions to analyze')
    parser.add_argument('--output-dir', type=str,
                       default='/root/research-dir/dev/jazzcash-fraud-detection/analysis/cosine_similarity',
                       help='Output directory for results CSV')
    
    args = parser.parse_args()
    
    analyzer = FraudCentroidVsQuery()
    success = analyzer.run(
        args.fraud_start, 
        args.fraud_end, 
        args.query,
        args.output_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

# python fraud_centroid_vs_query.py --query "fraud_flag = 1 AND cutoff_date >= '2025-07-01' AND cutoff_date <= '2025-07-31'"