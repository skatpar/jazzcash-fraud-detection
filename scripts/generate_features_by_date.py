#!/usr/bin/env python3
"""
Feature Generation Script for Fraud Detection
==============================================

This script generates features for a specific cutoff date and inserts them
into the combined_features table in ClickHouse.

Usage:
    python generate_features_by_date.py --cutoff-date 2025-07-01
    python generate_features_by_date.py --cutoff-date 2025-07-01 --cluster my_cluster_2shards
    python generate_features_by_date.py --cutoff-date 2025-07-01 --dry-run

Arguments:
    --cutoff-date    : Date in YYYY-MM-DD format (required)
    --cluster        : ClickHouse cluster name (default: my_cluster_2shards)
    --host           : ClickHouse host (default: localhost)
    --port           : ClickHouse port (default: 9000)
    --database       : Database name (default: public)
    --dry-run        : Preview query without executing (optional)
    --verbose        : Enable verbose logging (optional)

Example:
    python generate_features_by_date.py --cutoff-date 2025-07-15 --verbose
"""

import argparse
import sys
from datetime import datetime
from clickhouse_driver import Client
import logging

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


def get_combined_features_query(cutoff_date, cluster_name):
    """
    Generate the SQL query to create combined features for a specific cutoff date.
    
    Args:
        cutoff_date: Date string in YYYY-MM-DD format
        cluster_name: ClickHouse cluster name
        
    Returns:
        SQL query string
    """
    query = f"""
INSERT INTO combined_features_distributed
SELECT
    t.account_no,
    t.cutoff_date,
    
    -- Transaction Features
    t.total_transactions,
    t.total_amount,
    t.avg_amount,
    t.max_amount,
    t.min_amount,
    t.std_amount,
    t.unique_channels,
    t.unique_transaction_types,
    t.total_credits,
    t.total_debits,
    t.credit_amount,
    t.debit_amount,
    t.avg_credit_amount,
    t.avg_debit_amount,
    t.max_credit_amount,
    t.max_debit_amount,
    t.credit_debit_ratio,
    t.transaction_velocity_1d,
    t.transaction_velocity_7d,
    t.transaction_velocity_30d,
    t.amount_velocity_1d,
    t.amount_velocity_7d,
    t.amount_velocity_30d,
    t.night_transaction_ratio,
    t.weekend_transaction_ratio,
    t.high_value_transaction_count,
    t.high_value_transaction_ratio,
    t.failed_transaction_count,
    t.failed_transaction_ratio,
    t.avg_time_between_transactions,
    t.transaction_amount_variance,
    t.unique_counterparties,
    t.repeated_counterparty_ratio,
    t.cross_channel_activity,
    t.channel_diversity_score,
    t.transaction_type_diversity_score,
    t.sudden_amount_increase,
    t.sudden_transaction_increase,
    t.dormancy_period_days,
    t.reactivation_flag,
    t.first_transaction_date,
    t.last_transaction_date,
    t.account_age_days,
    t.days_since_last_transaction,
    t.transaction_frequency,
    t.avg_daily_transactions,
    t.max_daily_transactions,
    t.transaction_consistency_score,
    t.amount_consistency_score,
    t.unusual_hour_transactions,
    t.unusual_day_transactions,
    t.round_amount_transactions,
    t.round_amount_ratio,
    t.sequential_transaction_pattern,
    t.burst_transaction_count,
    t.largest_single_transaction,
    t.smallest_single_transaction,
    t.transaction_amount_range,
    t.coefficient_of_variation,
    t.transaction_entropy,
    t.amount_entropy,
    t.channel_switch_frequency,
    t.transaction_type_switch_frequency,
    t.peak_hour_transaction_ratio,
    t.off_peak_transaction_ratio,
    t.business_hours_ratio,
    t.monthly_transaction_trend,
    t.weekly_transaction_trend,
    t.average_transaction_gap_hours,
    t.max_transaction_gap_hours,
    t.min_transaction_gap_hours,
    t.rapid_succession_count,
    t.micro_transaction_count,
    t.micro_transaction_ratio,
    t.large_round_amount_count,
    t.withdrawal_deposit_pattern,
    t.balance_volatility_proxy,
    
    -- Account From Features
    u.total_transactions AS ac_from_total_transactions,
    u.total_amount AS ac_from_total_amount,
    u.avg_amount AS ac_from_avg_amount,
    u.max_amount AS ac_from_max_amount,
    u.min_amount AS ac_from_min_amount,
    u.unique_channels AS ac_from_unique_channels,
    u.unique_transaction_types AS ac_from_unique_transaction_types,
    u.total_credits AS ac_from_total_credits,
    u.total_debits AS ac_from_total_debits,
    u.credit_amount AS ac_from_credit_amount,
    u.debit_amount AS ac_from_debit_amount,
    u.transaction_velocity_7d AS ac_from_transaction_velocity_7d,
    u.amount_velocity_7d AS ac_from_amount_velocity_7d,
    u.night_transaction_ratio AS ac_from_night_transaction_ratio,
    u.weekend_transaction_ratio AS ac_from_weekend_transaction_ratio,
    u.high_value_transaction_count AS ac_from_high_value_transaction_count,
    u.failed_transaction_count AS ac_from_failed_transaction_count,
    u.unique_counterparties AS ac_from_unique_counterparties,
    u.cross_channel_activity AS ac_from_cross_channel_activity,
    u.account_age_days AS ac_from_account_age_days,
    u.days_since_last_transaction AS ac_from_days_since_last_transaction,
    u.unusual_hour_transactions AS ac_from_unusual_hour_transactions,
    u.round_amount_ratio AS ac_from_round_amount_ratio,
    
    -- Target Variable (is_fraud)
    CASE 
        WHEN f.account_no IS NOT NULL THEN 1
        WHEN v.account_no IS NOT NULL THEN 1
        WHEN c.account_no IS NOT NULL THEN 1
        ELSE 0
    END AS is_fraud,
    
    -- Fraud Source (for analysis)
    CASE 
        WHEN f.account_no IS NOT NULL THEN 'fraud_msisdn'
        WHEN v.account_no IS NOT NULL THEN 'victim_msisdn'
        WHEN c.account_no IS NOT NULL THEN 'complaint_msisdn'
        ELSE 'legitimate'
    END AS fraud_source

FROM transaction_features_distributed AS t
LEFT JOIN ac_from_features_distributed AS u 
    ON t.account_no = u.account_no 
    AND u.cutoff_date = toDate('{cutoff_date}')
LEFT JOIN fraud_accounts_with_types AS f 
    ON t.account_no = f.account_no
LEFT JOIN victim_accounts_with_types AS v 
    ON t.account_no = v.account_no
LEFT JOIN complaint_accounts_with_types AS c 
    ON t.account_no = c.account_no
WHERE t.cutoff_date = toDate('{cutoff_date}')
SETTINGS distributed_product_mode = 'global'
"""
    return query.strip()


def check_existing_data(client, cutoff_date, database):
    """Check if data already exists for the given cutoff date."""
    query = f"""
    SELECT count() as count
    FROM {database}.combined_features_distributed
    WHERE cutoff_date = toDate('{cutoff_date}')
    """
    
    result = client.execute(query)
    count = result[0][0] if result else 0
    return count


def get_source_data_counts(client, cutoff_date, database):
    """Get counts from source tables for the cutoff date."""
    counts = {}
    
    # Transaction features count
    query = f"""
    SELECT count() as count
    FROM {database}.transaction_features_distributed
    WHERE cutoff_date = toDate('{cutoff_date}')
    """
    result = client.execute(query)
    counts['transaction_features'] = result[0][0] if result else 0
    
    # Account from features count
    query = f"""
    SELECT count() as count
    FROM {database}.ac_from_features_distributed
    WHERE cutoff_date = toDate('{cutoff_date}')
    """
    result = client.execute(query)
    counts['ac_from_features'] = result[0][0] if result else 0
    
    return counts


def generate_features(cutoff_date, cluster='my_cluster_2shards', host='localhost', 
                     port=9000, database='public', dry_run=False, verbose=False):
    """
    Generate features for a specific cutoff date and insert into combined_features table.
    
    Args:
        cutoff_date: Date string in YYYY-MM-DD format
        cluster: ClickHouse cluster name
        host: ClickHouse host
        port: ClickHouse port
        database: Database name
        dry_run: If True, only show the query without executing
        verbose: Enable verbose logging
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("FRAUD DETECTION FEATURE GENERATION")
    logger.info("=" * 80)
    logger.info(f"Cutoff Date: {cutoff_date}")
    logger.info(f"Cluster: {cluster}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Database: {database}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 80)
    
    try:
        # Connect to ClickHouse
        logger.info("\n📡 Connecting to ClickHouse...")
        client = Client(
            host=host,
            port=port,
            database=database,
            password='DfsTeChB1',
            settings={'use_numpy': True}
        )
        
        # Test connection
        result = client.execute('SELECT version()')
        version = result[0][0]
        logger.info(f"✅ Connected to ClickHouse {version}")
        
        # Check existing data
        logger.info(f"\n🔍 Checking existing data for {cutoff_date}...")
        existing_count = check_existing_data(client, cutoff_date, database)
        
        if existing_count > 0:
            logger.warning(f"⚠️  Found {existing_count:,} existing records for {cutoff_date}")
            response = input("Do you want to delete existing data and regenerate? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                delete_query = f"""
                ALTER TABLE {database}.combined_features_distributed ON CLUSTER {cluster}
                DELETE WHERE cutoff_date = toDate('{cutoff_date}')
                """
                logger.info("🗑️  Deleting existing data...")
                if not dry_run:
                    client.execute(delete_query)
                    logger.info(f"✅ Deleted {existing_count:,} records")
                else:
                    logger.info(f"[DRY RUN] Would delete {existing_count:,} records")
            else:
                logger.info("❌ Operation cancelled by user")
                return
        else:
            logger.info("✅ No existing data found")
        
        # Check source data availability
        logger.info(f"\n📊 Checking source data availability...")
        source_counts = get_source_data_counts(client, cutoff_date, database)
        
        logger.info(f"  • Transaction features: {source_counts['transaction_features']:,} records")
        logger.info(f"  • Account from features: {source_counts['ac_from_features']:,} records")
        
        if source_counts['transaction_features'] == 0:
            logger.error(f"❌ No transaction features found for {cutoff_date}")
            logger.error("   Please generate transaction features first!")
            return
        
        # Generate query
        logger.info("\n📝 Generating combined features query...")
        query = get_combined_features_query(cutoff_date, cluster)
        
        if dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("DRY RUN - QUERY PREVIEW")
            logger.info("=" * 80)
            print("\n" + query + "\n")
            logger.info("=" * 80)
            logger.info("✅ Dry run complete - no data was modified")
            return
        
        # Execute query
        logger.info(f"\n🚀 Executing feature generation for {cutoff_date}...")
        logger.info("   This may take several minutes depending on data volume...")
        
        start_time = datetime.now()
        client.execute(query)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Verify insertion
        logger.info("\n✅ Query executed successfully!")
        logger.info(f"⏱️  Execution time: {duration:.2f} seconds")
        
        # Count inserted records
        logger.info("\n🔍 Verifying inserted data...")
        new_count = check_existing_data(client, cutoff_date, database)
        logger.info(f"✅ Inserted {new_count:,} records for {cutoff_date}")
        
        # Get fraud statistics
        fraud_query = f"""
        SELECT 
            fraud_source,
            count() as count,
            round(count() * 100.0 / {new_count}, 2) as percentage
        FROM {database}.combined_features_distributed
        WHERE cutoff_date = toDate('{cutoff_date}')
        GROUP BY fraud_source
        ORDER BY count DESC
        """
        
        fraud_stats = client.execute(fraud_query)
        
        logger.info("\n📊 Fraud Distribution:")
        logger.info("-" * 80)
        for row in fraud_stats:
            source, count, pct = row
            logger.info(f"  • {source:20s}: {count:>10,} ({pct:>6.2f}%)")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ FEATURE GENERATION COMPLETED SUCCESSFULLY!")
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
        description='Generate fraud detection features for a specific cutoff date',
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
        '--cluster',
        type=str,
        default='my_cluster_2shards',
        help='ClickHouse cluster name (default: my_cluster_2shards)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='ClickHouse host (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=9000,
        help='ClickHouse port (default: 9000)'
    )
    
    parser.add_argument(
        '--database',
        type=str,
        default='public',
        help='Database name (default: public)'
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
    
    # Generate features
    generate_features(
        cutoff_date=args.cutoff_date,
        cluster=args.cluster,
        host=args.host,
        port=args.port,
        database=args.database,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
