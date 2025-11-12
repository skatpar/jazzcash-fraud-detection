#!/usr/bin/env python3
"""
Channel-wise Fraud Analysis Script
Analyzes fraud patterns across different transaction channels and saves results to CSV
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, round as spark_round
from datetime import datetime
import logging
import os

# Configure logging
log_filename = f"channel_fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(os.path.dirname(__file__), log_filename)

logger = logging.getLogger('channel_fraud_analysis')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configuration
jar_files = [
    "/root/research-dir/dev/jazzcash-fraud-detection/utils/clickhouse-jdbc-0.9.2-all-dependencies.jar"
]

CLICKHOUSE_CONFIG = {
    'host': 'localhost',
    'port': 9000,
    'database': 'public',
    'user': 'default',
    'password': 'DfsTeChB1'
}

# Date range for analysis
START_DATE = '2025-01-01'
END_DATE = '2025-06-30'

def initialize_spark():
    """Initialize Spark session with ClickHouse JDBC connector"""
    logger.info("🚀 Initializing Spark session...")
    
    # Stop existing session if any
    try:
        existing_spark = SparkSession.getActiveSession()
        if existing_spark:
            existing_spark.stop()
            logger.info("🔄 Stopped existing Spark session")
    except Exception as e:
        pass
    
    spark = SparkSession.builder \
        .appName("channel_fraud_analysis") \
        .master("spark://dfs-ai-app2:7077") \
        .config("spark.jars", ",".join(jar_files)) \
        .config("spark.executor.memory", "100g") \
        .config("spark.executor.memoryOverhead", "5g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.cores", "32") \
        .config("spark.executor.instances", "2") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "96") \
        .getOrCreate()
    
    logger.info("✅ Spark session initialized")
    return spark

def load_data(spark):
    """Load fraud data from ClickHouse"""
    logger.info(f"📥 Loading fraud data from {START_DATE} to {END_DATE}...")
    
    url = f"jdbc:ch://{CLICKHOUSE_CONFIG['host']}:8123/{CLICKHOUSE_CONFIG['database']}"
    user = CLICKHOUSE_CONFIG['user']
    password = CLICKHOUSE_CONFIG['password']
    driver = "com.clickhouse.jdbc.ClickHouseDriver"
    
    # Select only necessary columns for analysis
    query = f"""
        SELECT 
            trx_channel,
            fraud_flag,
            trx_amt
        FROM stixor_fraud_features_distributed
        WHERE cutoff_date BETWEEN '{START_DATE}' AND '{END_DATE}'
            AND mbar_account_type_name = 'Customer Account'
    """
    
    subquery = f"({query}) AS fraud_data"
    
    df = (spark.read
        .format('jdbc')
        .option('driver', driver)
        .option('url', url)
        .option('user', user)
        .option('password', password)
        .option('dbtable', subquery)
        .option('fetchsize', '100000')
        .option("partitionColumn", "cutoff_date")
        .option('lowerBound', START_DATE)
        .option('upperBound', END_DATE)
        .option('numPartitions', '30')
        .load())
    
    # Cache for better performance
    df.cache()
    total_records = df.count()
    
    logger.info(f"✅ Data loaded: {total_records:,} records")
    return df

def analyze_channel_fraud(df):
    """Analyze fraud patterns by channel"""
    logger.info("🔍 Analyzing channel-wise fraud patterns...")
    
    # Group by channel and calculate fraud statistics
    channel_stats = df.groupBy('trx_channel').agg(
        count('*').alias('total_transactions'),
        spark_sum('fraud_flag').alias('fraud_count'),
        spark_sum(col('fraud_flag') * col('trx_amt')).alias('fraud_amount'),
        spark_sum('trx_amt').alias('total_amount')
    )
    
    # Calculate fraud percentage and sort by fraud count
    channel_stats = channel_stats.withColumn(
        'fraud_percentage',
        spark_round((col('fraud_count') / col('total_transactions')) * 100, 2)
    ).withColumn(
        'fraud_amount_percentage',
        spark_round((col('fraud_amount') / col('total_amount')) * 100, 2)
    ).orderBy(col('fraud_count').desc())
    
    logger.info("✅ Channel analysis completed")
    return channel_stats

def save_to_csv(df, output_path):
    """Save analysis results to CSV"""
    logger.info(f"💾 Saving results to {output_path}...")
    
    # Convert to Pandas and save as single CSV file
    pandas_df = df.toPandas()
    pandas_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Results saved to {output_path}")
    logger.info(f"   • Total channels analyzed: {len(pandas_df)}")
    logger.info(f"   • Total fraud transactions: {pandas_df['fraud_count'].sum():,.0f}")
    logger.info(f"   • Total transactions: {pandas_df['total_transactions'].sum():,.0f}")
    
    return pandas_df

def display_summary(df):
    """Display summary statistics"""
    logger.info("\n" + "="*80)
    logger.info("📊 CHANNEL-WISE FRAUD ANALYSIS SUMMARY")
    logger.info("="*80)
    
    # Top 10 channels by fraud count
    logger.info("\n🔝 Top 10 Channels by Fraud Count:")
    top_channels = df.head(10)
    for idx, row in top_channels.iterrows():
        logger.info(f"   {idx+1}. {row['trx_channel']}")
        logger.info(f"      • Total Transactions: {row['total_transactions']:,}")
        logger.info(f"      • Fraud Count: {row['fraud_count']:,.0f}")
        logger.info(f"      • Fraud %: {row['fraud_percentage']:.2f}%")
        logger.info(f"      • Fraud Amount: {row['fraud_amount']:,.2f}")
        logger.info(f"      • Fraud Amount %: {row['fraud_amount_percentage']:.2f}%")
    
    # Overall statistics
    total_txns = df['total_transactions'].sum()
    total_fraud = df['fraud_count'].sum()
    overall_fraud_pct = (total_fraud / total_txns) * 100
    
    logger.info(f"\n📈 Overall Statistics:")
    logger.info(f"   • Total Transactions: {total_txns:,}")
    logger.info(f"   • Total Fraud Cases: {total_fraud:,.0f}")
    logger.info(f"   • Overall Fraud Rate: {overall_fraud_pct:.2f}%")
    logger.info(f"   • Channels Analyzed: {len(df)}")
    logger.info("="*80 + "\n")

def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("🎯 Starting Channel-wise Fraud Analysis")
    logger.info("="*80)
    
    try:
        # Initialize Spark
        spark = initialize_spark()
        
        # Load data
        df = load_data(spark)
        
        # Analyze fraud by channel
        channel_stats = analyze_channel_fraud(df)
        
        # Show preview
        logger.info("\n📋 Preview of results:")
        channel_stats.show(10, truncate=False)
        
        # Save to CSV
        output_filename = f"channel_fraud_analysis_{START_DATE}_to_{END_DATE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", output_filename)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        pandas_df = save_to_csv(channel_stats, output_path)
        
        # Display summary
        display_summary(pandas_df)
        
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"📄 CSV file: {output_path}")
        logger.info(f"📝 Log file: {log_path}")
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Stop Spark session
        if spark:
            spark.stop()
            logger.info("🛑 Spark session stopped")

if __name__ == "__main__":
    main()
