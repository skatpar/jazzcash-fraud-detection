#!/usr/bin/env python3
"""
Transaction-Level Feature Generation Script
===========================================

This script generates transaction-level features for fraud detection.
Each transaction gets features based on:
- Time-based attributes (hour, day, weekend, night)
- Balance features (log transform, change, percentage)
- Historical lookback (3-day window before transaction)
- Risk indicators (high activity, multi-channel, amount deviation)
- Channel and type one-hot encoding

Features Generated per Transaction:
- Time-based: 6 features
- Balance: 3 features
- 3-day lookback: 10 features
- Risk indicators: 4 features
- Channel one-hot: 5 features
- Type one-hot: 4 features
- Total: ~32 features per transaction

Usage:
    python generate_transaction_features.py --cutoff-date 2025-07-01
    python generate_transaction_features.py --cutoff-date 2025-07-01 --verbose
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from clickhouse_driver import Client
from config_manager import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_date(date_string):
    """Validate date format and return datetime object."""
    try:
        date_obj = datetime.strptime(date_string, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_string}'. Use YYYY-MM-DD format."
        )


def generate_transaction_features(cutoff_date, config=None, lookback_days=None, 
                                  dry_run=False, verbose=False):
    """
    Generate transaction-level features for a specific cutoff date.
    
    Args:
        cutoff_date: Date string in YYYY-MM-DD format
        config: Config object (if None, loads from default location)
        lookback_days: Number of days to look back (default: from config or 3)
        dry_run: If True, only show the query without executing
        verbose: Enable verbose logging
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    if config is None:
        config = get_config()
    
    # Use config values or provided values
    if lookback_days is None:
        lookback_days = config.transaction_lookback_days
    
    host = config.host
    port = config.port
    database = config.database
    cluster = config.cluster
    source_table = config.source_table
    
    logger.info("=" * 80)
    logger.info("TRANSACTION-LEVEL FEATURE GENERATION")
    logger.info("=" * 80)
    logger.info(f"Cutoff Date: {cutoff_date}")
    logger.info(f"Lookback Days: {lookback_days}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Database: {database}")
    logger.info(f"Source Table: {source_table}")
    logger.info("=" * 80)
    
    # Calculate start date for lookback
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    lookback_start = (cutoff - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    logger.info(f"\nLookback Range: {lookback_start} to {cutoff_date}")
    logger.info(f"Features per transaction: ~32 (time + balance + lookback + risk + encoding)")
    
    try:
        # Connect to ClickHouse
        logger.info("\n📡 Connecting to ClickHouse...")
        client = Client(**config.get_connection_params())
        
        # Test connection
        result = client.execute('SELECT version()')
        version = result[0][0]
        logger.info(f"✅ Connected to ClickHouse {version}")
        
        # Check if transaction features already exist for this cutoff date
        logger.info(f"\n🔍 Checking existing data for {cutoff_date}...")
        check_query = f"""
        SELECT count() as count
        FROM {database}.transaction_features_distributed
        WHERE cutoff_date = toDate('{cutoff_date}')
        """
        
        existing_count = client.execute(check_query)[0][0]
        
        if existing_count > 0:
            logger.warning(f"⚠️  Found {existing_count:,} existing records for {cutoff_date}")
            response = input("Do you want to delete existing data and regenerate? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                delete_query = f"""
                ALTER TABLE {database}.transaction_features_distributed ON CLUSTER {cluster}
                DELETE WHERE cutoff_date = toDate('{cutoff_date}')
                """
                logger.info("🗑️  Deleting existing data...")
                if not dry_run:
                    client.execute(delete_query)
                    logger.info(f"✅ Deleted {existing_count:,} records")
            else:
                logger.info("❌ Operation cancelled by user")
                return
        else:
            logger.info("✅ No existing data found")
        
        # Check source data availability
        logger.info(f"\n📊 Checking source data for {cutoff_date}...")
        source_check = f"""
        SELECT count() as count
        FROM {database}.{source_table}
        WHERE data_date = toDate('{cutoff_date}')
        """
        
        source_count = client.execute(source_check)[0][0]
        logger.info(f"   • Transactions on {cutoff_date}: {source_count:,}")
        
        if source_count == 0:
            logger.error(f"❌ No source transactions found for {cutoff_date}")
            logger.error("   Please check the source table and date!")
            return
        
        # Generate transaction feature query
        logger.info("\n📝 Generating transaction feature query...")
        
        feature_query = f"""
INSERT INTO {database}.transaction_features_distributed
SELECT 
    -- IDENTIFIERS
    t1.trans_id,
    t1.ac_from,
    t1.ac_to,
    t1.data_date,
    t1.trans_initiate_time,
    toDate('{cutoff_date}') as cutoff_date,
    
    -- ORIGINAL TRANSACTION ATTRIBUTES
    t1.trx_channel,
    t1.trx_type,
    t1.start_balance,
    t1.end_balance,
    t1.trx_amt,
    
    -- TIME-BASED FEATURES
    toHour(t1.trans_initiate_time) as hour_of_day,
    toDayOfWeek(t1.trans_initiate_time) as day_of_week,
    if(toDayOfWeek(t1.trans_initiate_time) IN (6, 7), 1, 0) as is_weekend,
    if(toHour(t1.trans_initiate_time) IN (2, 3, 4, 5, 6), 1, 0) as is_night,
    if(toHour(t1.trans_initiate_time) BETWEEN 9 AND 17, 1, 0) as is_business_hours,
    if(toHour(t1.trans_initiate_time) BETWEEN 0 AND 6, 1, 0) as is_unusual_hour,
    
    -- BALANCE FEATURES
    if(t1.start_balance > 0, log(t1.start_balance + 1), 0) as start_balance_log,
    t1.end_balance - t1.start_balance as balance_change,
    if(t1.start_balance > 0, 
       ((t1.end_balance - t1.start_balance) / t1.start_balance) * 100, 
       0) as balance_change_pct,
    
    -- 3-DAY LOOKBACK FEATURES (historical context before this transaction)
    -- Count transactions in previous 3 days
    (SELECT count(*) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
    ) as txns_3d,
    
    -- Total amount in previous 3 days
    (SELECT sum(start_balance) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
    ) as total_amount_3d,
    
    -- Average amount in previous 3 days
    (SELECT avg(start_balance) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
    ) as avg_amount_3d,
    
    -- Max amount in previous 3 days
    (SELECT max(start_balance) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
    ) as max_amount_3d,
    
    -- Min amount in previous 3 days
    (SELECT min(start_balance) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
    ) as min_amount_3d,
    
    -- Unique recipients in previous 3 days
    (SELECT uniq(ac_to) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
       AND t2.ac_to != ''
    ) as unique_recipients_3d,
    
    -- Unique channels in previous 3 days
    (SELECT uniq(trx_channel) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
    ) as unique_channels_3d,
    
    -- Unique transaction types in previous 3 days
    (SELECT uniq(trx_type) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
    ) as unique_types_3d,
    
    -- Night transactions in previous 3 days
    (SELECT count(*) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
       AND toHour(t2.trans_initiate_time) IN (2, 3, 4, 5, 6)
    ) as night_txns_3d,
    
    -- Weekend transactions in previous 3 days
    (SELECT count(*) 
     FROM {database}.{source_table} t2
     WHERE t2.ac_from = t1.ac_from
       AND t2.data_date < t1.data_date
       AND t2.data_date >= toDate('{lookback_start}')
       AND toDayOfWeek(t2.trans_initiate_time) IN (6, 7)
    ) as weekend_txns_3d,
    
    -- RISK INDICATORS
    -- High activity flag (>10 transactions in 3d)
    if((SELECT count(*) 
        FROM {database}.{source_table} t2
        WHERE t2.ac_from = t1.ac_from
          AND t2.data_date < t1.data_date
          AND t2.data_date >= toDate('{lookback_start}')
       ) > 10, 1, 0) as is_high_activity_3d,
    
    -- Multi-channel flag (>1 channel in 3d)
    if((SELECT uniq(trx_channel) 
        FROM {database}.{source_table} t2
        WHERE t2.ac_from = t1.ac_from
          AND t2.data_date < t1.data_date
          AND t2.data_date >= toDate('{lookback_start}')
       ) > 1, 1, 0) as multi_channel_recent,
    
    -- Amount deviation from 3d average
    if((SELECT avg(start_balance) 
        FROM {database}.{source_table} t2
        WHERE t2.ac_from = t1.ac_from
          AND t2.data_date < t1.data_date
          AND t2.data_date >= toDate('{lookback_start}')
       ) > 0,
       abs(t1.start_balance - (
           SELECT avg(start_balance) 
           FROM {database}.{source_table} t2
           WHERE t2.ac_from = t1.ac_from
             AND t2.data_date < t1.data_date
             AND t2.data_date >= toDate('{lookback_start}')
       )),
       0) as amount_deviation_from_avg,
    
    -- Night and weekend combo
    if(toHour(t1.trans_initiate_time) IN (2, 3, 4, 5, 6) 
       AND toDayOfWeek(t1.trans_initiate_time) IN (6, 7), 1, 0) as night_weekend_combo,
    
    -- CHANNEL ONE-HOT ENCODING
    if(t1.trx_channel = 'New JC App', 1, 0) as channel_new_jc_app,
    if(t1.trx_channel = 'USSD', 1, 0) as channel_ussd,
    if(t1.trx_channel = 'USSD_API', 1, 0) as channel_ussd_api,
    if(t1.trx_channel = 'Payment Gateway', 1, 0) as channel_payment_gateway,
    if(t1.trx_channel = 'Mobile App', 1, 0) as channel_mobile_app,
    
    -- TYPE ONE-HOT ENCODING
    if(t1.trx_type = 'Transfer C2C', 1, 0) as type_transfer_c2c,
    if(t1.trx_type = 'Transfer C2B', 1, 0) as type_transfer_c2b,
    if(t1.trx_type = 'Bill Payment', 1, 0) as type_bill_payment,
    if(t1.trx_type = 'Mobile Load', 1, 0) as type_mobile_load,
    
    -- METADATA
    now() as processing_timestamp,
    toDate(now()) as created_at

FROM {database}.{source_table} t1
WHERE t1.data_date = toDate('{cutoff_date}')
ORDER BY t1.ac_from, t1.trans_initiate_time
"""
        
        if dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("DRY RUN - QUERY PREVIEW")
            logger.info("=" * 80)
            print("\n" + feature_query + "\n")
            logger.info("=" * 80)
            logger.info("✅ Dry run complete - no data was modified")
            return
        
        # Execute query
        logger.info(f"\n🚀 Executing transaction feature generation for {cutoff_date}...")
        logger.info("   This may take several minutes depending on data volume...")
        logger.info("   Note: Lookback features require self-joins which can be slow")
        
        start_time = datetime.now()
        client.execute(feature_query)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n✅ Query executed successfully!")
        logger.info(f"⏱️  Execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Verify insertion
        logger.info("\n🔍 Verifying inserted data...")
        new_count = client.execute(check_query)[0][0]
        logger.info(f"✅ Inserted {new_count:,} transaction records for {cutoff_date}")
        
        # Get sample statistics
        stats_query = f"""
        SELECT 
            avg(hour_of_day) as avg_hour,
            sum(is_weekend) as weekend_count,
            sum(is_night) as night_count,
            sum(is_high_activity_3d) as high_activity_count,
            avg(txns_3d) as avg_lookback_txns,
            count() as total_transactions
        FROM {database}.transaction_features_distributed
        WHERE cutoff_date = toDate('{cutoff_date}')
        """
        
        stats = client.execute(stats_query)[0]
        
        logger.info("\n📊 Transaction Statistics:")
        logger.info(f"   • Average hour of day: {stats[0]:.2f}")
        logger.info(f"   • Weekend transactions: {stats[1]:,} ({(stats[1]/stats[5]*100):.2f}%)")
        logger.info(f"   • Night transactions: {stats[2]:,} ({(stats[2]/stats[5]*100):.2f}%)")
        logger.info(f"   • High activity users: {stats[3]:,} ({(stats[3]/stats[5]*100):.2f}%)")
        logger.info(f"   • Average 3d lookback txns: {stats[4]:.2f}")
        logger.info(f"   • Total transactions: {stats[5]:,}")
        
        # Check for duplicates
        logger.info("\n🔍 Checking for duplicate transactions...")
        dup_query = f"""
        SELECT count(*) 
        FROM (
            SELECT trans_id
            FROM {database}.transaction_features_distributed
            WHERE cutoff_date = toDate('{cutoff_date}')
            GROUP BY trans_id
            HAVING count(*) > 1
        )
        """
        
        dup_count = client.execute(dup_query)[0][0]
        
        if dup_count > 0:
            logger.warning(f"⚠️  Found {dup_count:,} duplicate transaction IDs!")
        else:
            logger.info("✅ No duplicate transaction IDs found")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRANSACTION FEATURE GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Close connection
        client.disconnect()
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("❌ ERROR OCCURRED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=verbose)
        sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate transaction-level features for fraud detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--cutoff-date',
        type=validate_date,
        required=True,
        help='Cutoff date in YYYY-MM-DD format (required)'
    )
    
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=None,
        help='Number of days to look back for historical features (default: from config or 3)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config/clickhouse_config.ini)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview query without executing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = get_config(args.config)
        logger.info(f"✅ Loaded configuration from: {config.config_file}")
    except FileNotFoundError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Generate transaction features
    generate_transaction_features(
        cutoff_date=args.cutoff_date,
        config=config,
        lookback_days=args.lookback_days,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
