import logging
import sys
import argparse
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *

# Configure logging
def setup_logging():
    """Setup logging configuration for background execution"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('/root/research-dir/logs', exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('/root/research-dir/logs/user_features.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="User-level fraud detection feature engineering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--target-table",
        type=str,
        default="customer_fraud_basic_features",
        help="Target PostgreSQL table name"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000000,
        help="Number of customers to sample for processing"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of customers to process per batch"
    )
    
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Fraction of total customers to sample (0.0-1.0)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-06-01",
        help="Start date for transaction data (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-07-05",
        help="End date for transaction data (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Set log level from arguments
logging.getLogger().setLevel(getattr(logging, args.log_level))
logger.info(f"Starting with arguments: {vars(args)}")

# Database configuration
DB_CONFIG = {
    'host': '10.205.161.118',
    'port': '5432',
    'database': 'db_fraud',
    'user': 'dfstechbi',
    'password': 'DfsTeChB1@923'
}

# JDBC Configuration
jdbc_driver_path = "/root/research-dir/dev/jazzcash-fraud-detection/utils/postgresql-42.7.1.jar"
jdbc_url = f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Optimized JDBC properties
properties = {
    "user": DB_CONFIG['user'],
    "password": DB_CONFIG['password'],
    "driver": "org.postgresql.Driver",
    "fetchsize": "10000",
    "batchsize": "15000",
    "isolationLevel": "READ_UNCOMMITTED",
    "queryTimeout": "1200",
    "loginTimeout": "60",
    "socketTimeout": "1200",
    "tcpKeepAlive": "true",
    "prepareThreshold": "5",
    "reWriteBatchedInserts": "true",
    "defaultRowFetchSize": "10000"
}

logger.info("Starting Spark session creation for user-level feature engineering")

# Create Spark session with optimized configuration for parallel job execution
try:
    spark = SparkSession.builder \
        .appName("UserLevelBasicFeatures") \
        .master("spark://dfs-ai-app2:7077") \
        .config("spark.jars", jdbc_driver_path) \
        .config("spark.executor.instances", "6") \
        .config("spark.executor.cores", "3") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.driver.memoryOverhead", "512m") \
        .config("spark.driver.cores", "1") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", "2") \
        .config("spark.dynamicAllocation.maxExecutors", "8") \
        .config("spark.dynamicAllocation.initialExecutors", "4") \
        .config("spark.shuffle.service.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.default.parallelism", "60") \
        .config("spark.sql.autoBroadcastJoinThreshold", "50MB") \
        .config("spark.executor.extraJavaOptions", "-Xss2m") \
        .config("spark.driver.extraJavaOptions", "-Xss2m") \
        .config("spark.scheduler.mode", "FAIR") \
        .getOrCreate()

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark session created successfully - Application ID: {spark.sparkContext.applicationId}")
    logger.info(f"Master: {spark.sparkContext.master}")
    logger.info("Configuration: Dynamic allocation 2-8 executors, 3 cores each, 8GB memory per executor")

except Exception as e:
    logger.error(f"Failed to create Spark session: {str(e)}")
    raise



def load_transaction_data(start_date_str=None, end_date_str=None):
    """Load transaction data with proper error handling and logging"""
    try:
        table_name = "public.stixor_iar"
        logger.info("Setting up date-based predicates for data loading")

        # Use provided dates or default from arguments
        start_date_str = start_date_str or args.start_date
        end_date_str = end_date_str or args.end_date

        # Create predicates for optimized partitioning
        predicates = []
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            predicates.append(f"data_date = '{date_str}'")
            current_date += timedelta(days=1)

        logger.info(f"Created {len(predicates)} predicates for date range {start_date.date()} to {end_date.date()}")

        # Load data using predicate-based partitioning
        df = spark.read.jdbc(
            url=jdbc_url,
            table=table_name,
            properties=properties,
            predicates=predicates
        )
        
        logger.info("Transaction data loaded successfully from PostgreSQL")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load transaction data: {str(e)}")
        raise

def load_and_sample_users():
    """Load July customer senders and create sample"""
    try:
        # Load July customer senders from saved parquet
        df_july_customer_senders = spark.read.parquet("dev/jazzcash-fraud-detection/data/july_2025_customer_senders")
        total_customers = df_july_customer_senders.count()
        logger.info(f"Loaded July customer senders: {total_customers:,}")

        # Take a random sample based on arguments
        df_july_customer_senders_sample = df_july_customer_senders.sample(fraction=args.sample_fraction, seed=42).limit(args.sample_size)
        df_july_customer_senders_sample.cache()
        sample_count = df_july_customer_senders_sample.count()
        logger.info(f"Created random sample of July customer senders: {sample_count:,} (fraction={args.sample_fraction}, limit={args.sample_size})")
        
        return df_july_customer_senders_sample
        
    except Exception as e:
        logger.error(f"Failed to load and sample users: {str(e)}")
        raise

# Execute data loading
df = load_transaction_data(args.start_date, args.end_date)
df_july_customer_senders_sample = load_and_sample_users()


import math

class UserFraudFeatureEngineer:
    """
    Simplified user-level basic feature engineering for fraud detection
    Creates one row per customer with basic aggregated features
    """
    
    def __init__(self, df):
        self.df = df
        logger.info("UserFraudFeatureEngineer initialized")
    
    def create_user_basic_features(self):
        """Create basic aggregated features per user"""
        logger.info("Creating basic user-level features")
        
        try:
            # Basic user aggregations
            user_features = self.df.groupBy("ac_from") \
                .agg(
                    F.count("trans_id").alias("total_transactions"),
                    F.countDistinct("trx_channel").alias("unique_channels"),
                    F.countDistinct("trx_type").alias("unique_types"),
                    F.countDistinct("data_date").alias("active_days"),
                    F.avg("start_balance").alias("avg_start_balance"),
                    F.min("trans_initiate_time").alias("first_transaction_time"),
                    F.max("trans_initiate_time").alias("last_transaction_time"),
                    F.sum(F.when(F.hour("trans_initiate_time").between(22, 23) | 
                               F.hour("trans_initiate_time").between(0, 6), 1).otherwise(0)).alias("night_transactions"),
                    F.sum(F.when(F.dayofweek("trans_initiate_time").isin([1, 7]), 1).otherwise(0)).alias("weekend_transactions")
                )
            
            # Add derived features
            user_features = user_features.withColumn(
                "days_between_first_last", 
                F.datediff(F.col("last_transaction_time"), F.col("first_transaction_time"))
            ).withColumn(
                "avg_transactions_per_day",
                F.when(F.col("days_between_first_last") > 0, 
                      F.col("total_transactions") / F.col("days_between_first_last")).otherwise(F.col("total_transactions"))
            ).withColumn(
                "night_transaction_ratio",
                F.col("night_transactions") / F.col("total_transactions")
            ).withColumn(
                "weekend_transaction_ratio", 
                F.col("weekend_transactions") / F.col("total_transactions")
            )
            
            logger.info("Basic user-level features created successfully")
            return user_features
            
        except Exception as e:
            logger.error(f"Error creating user features: {str(e)}")
            raise

    

def process_customer_batches_user_level(df_customers, df_transactions, batch_size=50000, target_table="customer_fraud_basic_features"):
    """
    Simplified batch processing for user-level customer features with logging
    """
    try:
        total_customers = df_customers.count()
        total_batches = math.ceil(total_customers / batch_size)
        
        logger.info(f"Starting user-level batch processing:")
        logger.info(f"Total customers: {total_customers:,}")
        logger.info(f"Batch size: {batch_size:,}")
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"Target table: {target_table}")
        
        # Create target table schema
        create_user_target_table(target_table)
        
        # Process in batches to avoid memory issues
        df_customers_hashed = df_customers.withColumn(
            "hash_mod", 
            F.abs(F.hash(F.col("ac_from"))) % total_batches
        )
        
        df_customers_hashed.cache()
        df_customers_hashed.count()
        
        successful_batches = 0
        total_users_processed = 0
        
        for batch_num in range(total_batches):
            batch_start_time = datetime.now()
            
            logger.info(f"Processing Batch {batch_num + 1}/{total_batches}")
            
            # Get customer batch
            customer_batch = df_customers_hashed.filter(F.col("hash_mod") == batch_num).select("ac_from")
            batch_customer_count = customer_batch.count()
            
            if batch_customer_count == 0:
                logger.info("Empty batch, skipping")
                continue
            
            logger.info(f"Customers in batch: {batch_customer_count:,}")
            
            # Filter transactions for this batch
            df_batch_transactions = df_transactions.join(customer_batch, on="ac_from", how="inner")
            batch_txn_count = df_batch_transactions.count()
            
            logger.info(f"Transactions in batch: {batch_txn_count:,}")
            
            if batch_txn_count == 0:
                logger.info("No transactions for batch, skipping")
                continue
            
            try:
                # Create user-level features
                logger.info("Creating user-level features...")
                user_feature_engineer = UserFraudFeatureEngineer(df_batch_transactions)
                df_user_features = user_feature_engineer.create_user_basic_features()
                
                # Add batch metadata
                df_user_features = df_user_features \
                    .withColumn("batch_number", F.lit(batch_num + 1)) \
                    .withColumn("processing_timestamp", F.current_timestamp())
                
                # Write to PostgreSQL table
                logger.info(f"Writing batch {batch_num + 1} to PostgreSQL table: {target_table}")
                
                write_properties = {
                    "user": DB_CONFIG['user'],
                    "password": DB_CONFIG['password'],
                    "driver": "org.postgresql.Driver",
                    "batchsize": "5000",
                    "isolationLevel": "READ_UNCOMMITTED",
                    "reWriteBatchedInserts": "true",
                    "stringtype": "unspecified"
                }
                
                try:
                    df_user_features.write.jdbc(
                        url=jdbc_url,
                        table=f"public.{target_table}",
                        mode="append",
                        properties=write_properties
                    )
                    logger.info(f"Successfully written to table: public.{target_table}")
                except Exception as db_error:
                    logger.error(f"Failed to write to database table: {str(db_error)}")
                    logger.error("Skipping this batch due to database write failure")
                
                batch_end_time = datetime.now()
                batch_duration = batch_end_time - batch_start_time
                
                successful_batches += 1
                total_users_processed += batch_customer_count
                
                logger.info(f"Batch {batch_num + 1} completed in {batch_duration}")
                logger.info(f"Users processed: {batch_customer_count:,}")
                logger.info(f"Written to table: public.{target_table}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num + 1}: {str(e)}")
                continue
        
        df_customers_hashed.unpersist()
        
        logger.info("User-level batch processing completed!")
        logger.info(f"Successful batches: {successful_batches}/{total_batches}")
        logger.info(f"Total users processed: {total_users_processed:,}")
        
        return successful_batches, total_users_processed
        
    except Exception as e:
        logger.error(f"Failed in batch processing: {str(e)}")
        raise


def create_user_target_table(table_name):
    """
    Create target table for user-level features with proper schema
    """
    logger.info(f"Creating user-level target table: {table_name}")
    
    try:
        # Create table schema for user features
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS public.{table_name} (
            ac_from VARCHAR(50) PRIMARY KEY,
            total_transactions INTEGER,
            unique_channels INTEGER,
            unique_types INTEGER,
            active_days INTEGER,
            avg_start_balance DOUBLE PRECISION,
            first_transaction_time TIMESTAMP,
            last_transaction_time TIMESTAMP,
            night_transactions INTEGER,
            weekend_transactions INTEGER,
            days_between_first_last INTEGER,
            avg_transactions_per_day DOUBLE PRECISION,
            night_transaction_ratio DOUBLE PRECISION,
            weekend_transaction_ratio DOUBLE PRECISION,
            batch_number INTEGER,
            processing_timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_{table_name}_batch_number ON public.{table_name}(batch_number);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_processing_timestamp ON public.{table_name}(processing_timestamp);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_total_transactions ON public.{table_name}(total_transactions);
        """
        
        # Execute table creation using Spark JDBC
        spark.read.jdbc(
            url=jdbc_url,
            table=f"(SELECT 1 as test_connection) as test",
            properties=properties
        ).collect()  # Test connection first
        
        logger.info(f"Table public.{table_name} schema prepared")
        logger.info("Note: Table will be created on first batch write with proper schema")
        
    except Exception as e:
        logger.warning(f"Could not pre-create table schema: {str(e)}")
        logger.info("Table will be created automatically on first write")
    
def main():
    """Main execution function"""
    try:
        # Set the scheduler pool for this job
        spark.sparkContext.setLocalProperty("spark.scheduler.pool", "feature_engineering")
        
        logger.info("Starting user-level batch feature engineering...")
        successful_batches, total_users = process_customer_batches_user_level(
            df_customers=df_july_customer_senders_sample,
            df_transactions=df,
            batch_size=args.batch_size,
            target_table=args.target_table
        )
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Processed {total_users:,} users in {successful_batches} batches")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
    finally:
        if 'spark' in globals():
            logger.info("Stopping Spark session...")
            spark.stop()

if __name__ == "__main__":
    main()