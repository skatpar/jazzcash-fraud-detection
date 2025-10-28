#!/usr/bin/env python3
"""
User-Level Feature Generation Script
====================================

This script generates user-level aggregated features for fraud detection.
Features are calculated using 3-day and 7-day lookback windows, excluding the cutoff date.

Features Generated:
- 3-day window: 9 features (transaction counts, amounts, channels, types, recipients)
- 7-day window: 29 features (aggregates + behavioral + channel/type diversity + time-based)
- Total: 38 features per user

Usage:
    python generate_user_features.py --cutoff-date 2025-07-01
    python generate_user_features.py --cutoff-date 2025-07-01 --lookback-days 7 --verbose
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


def generate_user_features(cutoff_date, config=None, lookback_days=None,
                          dry_run=False, verbose=False):
    """
    Generate user-level features for a specific cutoff date.
    
    Args:
        cutoff_date: Date string in YYYY-MM-DD format
        config: Config object (if None, loads from default location)
        lookback_days: Number of days to look back (default: from config or 7)
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
        lookback_days = config.default_lookback_days
    
    host = config.host
    port = config.port
    database = config.database
    cluster = config.cluster
    source_table = config.source_table
    
    logger.info("=" * 80)
    logger.info("USER-LEVEL FEATURE GENERATION")
    logger.info("=" * 80)
    logger.info(f"Cutoff Date: {cutoff_date}")
    logger.info(f"Lookback Days: {lookback_days}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Database: {database}")
    logger.info(f"Source Table: {source_table}")
    logger.info("=" * 80)
    
    # Calculate start date
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    start_date = (cutoff - timedelta(days=lookback_days - 1)).strftime('%Y-%m-%d')
    
    logger.info(f"\nDate Range: {start_date} to {cutoff_date}")
    logger.info(f"Feature Windows: 3-day (9 features) + 7-day (29 features) = 38 total")
    
    try:
        # Connect to ClickHouse
        logger.info("\n📡 Connecting to ClickHouse...")
        client = Client(**config.get_connection_params())
        
        # Test connection
        result = client.execute('SELECT version()')
        version = result[0][0]
        logger.info(f"✅ Connected to ClickHouse {version}")
        
        # Check if user features already exist for this cutoff date
        logger.info(f"\n🔍 Checking existing data for {cutoff_date}...")
        check_query = f"""
        SELECT count() as count
        FROM {database}.ac_from_features_distributed
        WHERE cutoff_date = toDate('{cutoff_date}')
        """
        
        existing_count = client.execute(check_query)[0][0]
        
        if existing_count > 0:
            logger.warning(f"⚠️  Found {existing_count:,} existing records for {cutoff_date}")
            response = input("Do you want to delete existing data and regenerate? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                delete_query = f"""
                ALTER TABLE {database}.ac_from_features_distributed ON CLUSTER {cluster}
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
        
        # Generate feature engineering query
        logger.info("\n📝 Generating user feature query...")
        
        feature_query = f"""
INSERT INTO {database}.ac_from_features_distributed
WITH 
-- Get users who transacted on cutoff date
active_users AS (
    SELECT DISTINCT ac_from
    FROM {database}.{source_table}
    WHERE data_date = toDate('{cutoff_date}')
      AND ac_from != ''
),
-- Pre-calculate top channels per user ({lookback_days}-day window)
user_channel_stats AS (
    SELECT 
        ac_from,
        trx_channel,
        count() as channel_count,
        row_number() OVER (PARTITION BY ac_from ORDER BY count() DESC) as channel_rank
    FROM (
        SELECT 
            ac_from,
            trx_channel
        FROM {database}.{source_table}
        WHERE data_date >= toDate('{start_date}')
          AND data_date <= toDate('{cutoff_date}')
          AND ac_from GLOBAL IN (SELECT ac_from FROM active_users)
    )
    GROUP BY ac_from, trx_channel
),
-- Pre-calculate top types per user
user_type_stats AS (
    SELECT 
        ac_from,
        trx_type,
        count() as type_count,
        row_number() OVER (PARTITION BY ac_from ORDER BY count() DESC) as type_rank
    FROM (
        SELECT 
            ac_from,
            trx_type
        FROM {database}.{source_table}
        WHERE data_date >= toDate('{start_date}')
          AND data_date <= toDate('{cutoff_date}')
          AND ac_from GLOBAL IN (SELECT ac_from FROM active_users)
    )
    GROUP BY ac_from, trx_type
),
-- Pre-calculate top recipients per user
user_recipient_stats AS (
    SELECT 
        ac_from,
        ac_to,
        count() as recipient_count,
        sum(start_balance) as total_to_recipient,
        row_number() OVER (PARTITION BY ac_from ORDER BY count() DESC) as recipient_rank
    FROM (
        SELECT 
            ac_from,
            ac_to,
            start_balance
        FROM {database}.{source_table}
        WHERE data_date >= toDate('{start_date}')
          AND data_date <= toDate('{cutoff_date}')
          AND ac_from GLOBAL IN (SELECT ac_from FROM active_users)
          AND ac_to != ''
    )
    GROUP BY ac_from, ac_to
)

SELECT 
    main.ac_from,
    toDate('{cutoff_date}') as cutoff_date,
    
    -- 3-DAY FEATURES (excludes cutoff date, uses days_back 1-3)
    sumIf(1, main.days_back BETWEEN 1 AND 3) as total_txns_3d,
    sumIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as total_amount_3d,
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as avg_amount_3d,
    quantileIf(0.5)(main.start_balance, main.days_back BETWEEN 1 AND 3) as median_amount_3d,
    maxIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as max_amount_3d,
    minIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as min_amount_3d,
    uniqIf(main.ac_to, main.days_back BETWEEN 1 AND 3) as unique_recipients_3d,
    uniqIf(main.trx_channel, main.days_back BETWEEN 1 AND 3) as unique_channels_3d,
    uniqIf(main.trx_type, main.days_back BETWEEN 1 AND 3) as unique_types_3d,
    
    -- 7-DAY FEATURES (excludes cutoff date, uses days_back 1-7)
    sumIf(1, main.days_back BETWEEN 1 AND 7) as total_txns_7d,
    sumIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as total_amount_7d,
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as avg_amount_7d,
    quantileIf(0.5)(main.start_balance, main.days_back BETWEEN 1 AND 7) as median_amount_7d,
    maxIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as max_amount_7d,
    minIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as min_amount_7d,
    uniqIf(main.ac_to, main.days_back BETWEEN 1 AND 7) as unique_recipients_7d,
    uniqIf(main.trx_channel, main.days_back BETWEEN 1 AND 7) as unique_channels_7d,
    uniqIf(main.trx_type, main.days_back BETWEEN 1 AND 7) as unique_types_7d,
    
    -- CHANNEL FEATURES (7-day)
    anyIf(ch.trx_channel, ch.channel_rank = 1) as most_used_channel_7d,
    argMax(main.trx_channel, main.trans_initiate_time) as last_used_channel,
    if(uniq(main.trx_channel) > 1, 
       1 - (max(ch.channel_count) / sum(ch.channel_count)), 0) as channel_diversity_score_7d,
    
    -- TYPE FEATURES (7-day)
    anyIf(ty.trx_type, ty.type_rank = 1) as most_used_type_7d,
    argMax(main.trx_type, main.trans_initiate_time) as last_used_type,
    if(uniq(main.trx_type) > 1, 
       1 - (max(ty.type_count) / sum(ty.type_count)), 0) as type_diversity_score_7d,
    
    -- TIME-BASED FEATURES (7-day)
    sumIf(1, toHour(main.trans_initiate_time) IN (2,3,4,5,6) AND main.days_back BETWEEN 1 AND 7) as night_txns_7d,
    sumIf(1, toDayOfWeek(main.trans_initiate_time) IN (6,7) AND main.days_back BETWEEN 1 AND 7) as weekend_txns_7d,
    sumIf(1, toHour(main.trans_initiate_time) BETWEEN 9 AND 17 AND main.days_back BETWEEN 1 AND 7) as peak_hour_txns_7d,
    sumIf(1, toHour(main.trans_initiate_time) NOT BETWEEN 9 AND 17 AND main.days_back BETWEEN 1 AND 7) as off_peak_hour_txns_7d,
    
    -- BALANCE FEATURES (7-day)
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as avg_start_balance_7d,
    avgIf(main.end_balance, main.days_back BETWEEN 1 AND 7) as avg_end_balance_7d,
    minIf(least(main.start_balance, main.end_balance), main.days_back BETWEEN 1 AND 7) as min_balance_7d,
    maxIf(greatest(main.start_balance, main.end_balance), main.days_back BETWEEN 1 AND 7) as max_balance_7d,
    stddevPopIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as balance_volatility_7d,
    
    -- RECIPIENT FEATURES (7-day)
    anyIf(rs.ac_to, rs.recipient_rank = 1) as top_recipient_7d,
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 7 AND main.ac_to != '') as avg_amount_per_recipient_7d,
    maxIf(main.start_balance, main.days_back BETWEEN 1 AND 7 AND main.ac_to != '') as max_amount_to_single_recipient_7d,
    if(count(main.ac_from) > 0, max(rs.recipient_count) / count(main.ac_from), 0) as recipient_concentration_ratio_7d,
    
    -- BEHAVIORAL FEATURES (7-day)
    if(count(main.ac_from) > 1,
       dateDiff('hour', min(main.trans_initiate_time), max(main.trans_initiate_time)) / (count(main.ac_from) - 1),
       0) as avg_time_between_txns_7d,
    count(main.ac_from) / greatest(dateDiff('day', min(main.data_date), max(main.data_date)) + 1, 1) as txn_frequency_score_7d,
    min(main.trans_initiate_time) as first_txn_time,
    max(main.trans_initiate_time) as last_txn_time,
    dateDiff('day', max(main.data_date), toDate('{cutoff_date}')) as days_since_last_txn,
    
    now() as processing_timestamp,
    toDate(now()) as created_at

FROM (
    SELECT 
        ac_from,
        ac_to,
        trans_id,
        start_balance,
        end_balance,
        trx_channel,
        trx_type,
        trans_initiate_time,
        data_date,
        dateDiff('day', data_date, toDate('{cutoff_date}')) as days_back
    FROM {database}.{source_table}
    WHERE data_date >= toDate('{start_date}')
      AND data_date <= toDate('{cutoff_date}')
      AND ac_from GLOBAL IN (SELECT ac_from FROM active_users)
    ORDER BY ac_from, trans_initiate_time
) main
GLOBAL LEFT JOIN user_channel_stats ch ON main.ac_from = ch.ac_from AND main.trx_channel = ch.trx_channel
GLOBAL LEFT JOIN user_type_stats ty ON main.ac_from = ty.ac_from AND main.trx_type = ty.trx_type  
GLOBAL LEFT JOIN user_recipient_stats rs ON main.ac_from = rs.ac_from AND main.ac_to = rs.ac_to
GROUP BY main.ac_from
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
        logger.info(f"\n🚀 Executing user feature generation for {cutoff_date}...")
        logger.info("   This may take several minutes...")
        
        start_time = datetime.now()
        client.execute(feature_query)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n✅ Query executed successfully!")
        logger.info(f"⏱️  Execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Verify insertion
        logger.info("\n🔍 Verifying inserted data...")
        new_count = client.execute(check_query)[0][0]
        logger.info(f"✅ Inserted {new_count:,} user records for {cutoff_date}")
        
        # Get sample statistics
        stats_query = f"""
        SELECT 
            avg(total_txns_7d) as avg_txns_7d,
            avg(total_amount_7d) as avg_amount_7d,
            count() as total_users
        FROM {database}.ac_from_features_distributed
        WHERE cutoff_date = toDate('{cutoff_date}')
        """
        
        stats = client.execute(stats_query)[0]
        
        logger.info("\n📊 User Statistics:")
        logger.info(f"   • Average transactions (7d): {stats[0]:.2f}")
        logger.info(f"   • Average amount (7d): {stats[1]:,.2f}")
        logger.info(f"   • Total users: {stats[2]:,}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ USER FEATURE GENERATION COMPLETED SUCCESSFULLY!")
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
        description='Generate user-level aggregated features for fraud detection',
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
        help='Number of days to look back (default: from config or 7)'
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
    
    # Generate user features
    generate_user_features(
        cutoff_date=args.cutoff_date,
        config=config,
        lookback_days=args.lookback_days,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
