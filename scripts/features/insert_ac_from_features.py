#!/usr/bin/env python3
"""
Insert AC_FROM (User-Level) Features Script
Processes user-level aggregated features for a date range and inserts into ClickHouse.

Usage:
    python insert_ac_from_features.py 2025-09-01 2025-09-30
"""

import sys
import time
import configparser
from pathlib import Path
from datetime import datetime, timedelta, date
from clickhouse_driver import Client


def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def get_clickhouse_client():
    """Initialize ClickHouse client from config"""
    config = configparser.ConfigParser()
    config_path = Path('/root/research-dir/dev/jazzcash-fraud-detection/config/clickhouse_config.ini')
    
    if config_path.exists():
        config.read(config_path)
        ch_config = {
            'host': config['clickhouse']['host'],
            'port': int(config['clickhouse']['port']),
            'database': config['clickhouse']['database'],
            'user': config['clickhouse']['user'],
            'password': config['clickhouse']['password']
        }
        print(f"✅ Configuration loaded from {config_path}")
    else:
        ch_config = {
            'host': 'localhost',
            'port': 9000,
            'database': 'public',
            'user': 'default',
            'password': 'DfsTeChB1'
        }
        print("⚠️  Config file not found, using default configuration")
    
    client = Client(
        host=ch_config['host'],
        port=ch_config['port'],
        database=ch_config['database'],
        user=ch_config['user'],
        password=ch_config['password'],
        settings={
            'max_execution_time': 7200,
            'send_timeout': 600,
            'receive_timeout': 600,
            'connect_timeout': 10
        }
    )
    
    # Test connection
    version = client.execute('SELECT version()')[0][0]
    print(f"✅ Connected to ClickHouse {version}")
    
    return client


def build_insert_query(cutoff_date, lookback_days=7):
    """Build the INSERT query for user-level features"""
    cutoff_dt = datetime.strptime(cutoff_date, '%Y-%m-%d')
    start_date = (cutoff_dt - timedelta(days=lookback_days - 1)).strftime('%Y-%m-%d')
    
    return f"""
INSERT INTO public.ac_from_features_distributed
WITH 
-- Get users who transacted on cutoff date
active_users AS (
    SELECT DISTINCT ac_from
    FROM public.stixor_iar_distributed
    WHERE data_date = toDate('{cutoff_date}')
      AND ac_from != ''
),
-- Pre-calculate top channels and types per user
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
        FROM public.stixor_iar_distributed
        WHERE data_date >= toDate('{start_date}')
          AND data_date <= toDate('{cutoff_date}')
          AND ac_from GLOBAL IN (SELECT ac_from FROM active_users)
    )
    GROUP BY ac_from, trx_channel
),
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
        FROM public.stixor_iar_distributed
        WHERE data_date >= toDate('{start_date}')
          AND data_date <= toDate('{cutoff_date}')
          AND ac_from GLOBAL IN (SELECT ac_from FROM active_users)
    )
    GROUP BY ac_from, trx_type
),
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
        FROM public.stixor_iar_distributed
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
    
    -- 3-day features (excludes cutoff date, uses days_back 1-3)
    sumIf(1, main.days_back BETWEEN 1 AND 3) as total_txns_3d,
    sumIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as total_amount_3d,
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as avg_amount_3d,
    quantileIf(0.5)(main.start_balance, main.days_back BETWEEN 1 AND 3) as median_amount_3d,
    maxIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as max_amount_3d,
    minIf(main.start_balance, main.days_back BETWEEN 1 AND 3) as min_amount_3d,
    uniqIf(main.ac_to, main.days_back BETWEEN 1 AND 3) as unique_recipients_3d,
    uniqIf(main.trx_channel, main.days_back BETWEEN 1 AND 3) as unique_channels_3d,
    uniqIf(main.trx_type, main.days_back BETWEEN 1 AND 3) as unique_types_3d,
    
    -- 7-day features (excludes cutoff date, uses days_back 1-7)
    sumIf(1, main.days_back BETWEEN 1 AND 7) as total_txns_7d,
    sumIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as total_amount_7d,
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as avg_amount_7d,
    quantileIf(0.5)(main.start_balance, main.days_back BETWEEN 1 AND 7) as median_amount_7d,
    maxIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as max_amount_7d,
    minIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as min_amount_7d,
    uniqIf(main.ac_to, main.days_back BETWEEN 1 AND 7) as unique_recipients_7d,
    uniqIf(main.trx_channel, main.days_back BETWEEN 1 AND 7) as unique_channels_7d,
    uniqIf(main.trx_type, main.days_back BETWEEN 1 AND 7) as unique_types_7d,
    
    -- Channel features (7-day)
    anyIf(ch.trx_channel, ch.channel_rank = 1) as most_used_channel_7d,
    argMax(main.trx_channel, main.trans_initiate_time) as last_used_channel,
    if(uniq(main.trx_channel) > 1, 
       1 - (max(ch.channel_count) / sum(ch.channel_count)), 0) as channel_diversity_score_7d,
    
    -- Type features (7-day)
    anyIf(ty.trx_type, ty.type_rank = 1) as most_used_type_7d,
    argMax(main.trx_type, main.trans_initiate_time) as last_used_type,
    if(uniq(main.trx_type) > 1, 
       1 - (max(ty.type_count) / sum(ty.type_count)), 0) as type_diversity_score_7d,
    
    -- Time-based features (7-day)
    sumIf(1, toHour(main.trans_initiate_time) IN (2,3,4,5,6) AND main.days_back BETWEEN 1 AND 7) as night_txns_7d,
    sumIf(1, toDayOfWeek(main.trans_initiate_time) IN (6,7) AND main.days_back BETWEEN 1 AND 7) as weekend_txns_7d,
    sumIf(1, toHour(main.trans_initiate_time) BETWEEN 9 AND 17 AND main.days_back BETWEEN 1 AND 7) as peak_hour_txns_7d,
    sumIf(1, toHour(main.trans_initiate_time) NOT BETWEEN 9 AND 17 AND main.days_back BETWEEN 1 AND 7) as off_peak_hour_txns_7d,
    
    -- Balance features (7-day)
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as avg_start_balance_7d,
    avgIf(main.end_balance, main.days_back BETWEEN 1 AND 7) as avg_end_balance_7d,
    minIf(least(main.start_balance, main.end_balance), main.days_back BETWEEN 1 AND 7) as min_balance_7d,
    maxIf(greatest(main.start_balance, main.end_balance), main.days_back BETWEEN 1 AND 7) as max_balance_7d,
    stddevPopIf(main.start_balance, main.days_back BETWEEN 1 AND 7) as balance_volatility_7d,
    
    -- Recipient features (7-day)
    anyIf(rs.ac_to, rs.recipient_rank = 1) as top_recipient_7d,
    avgIf(main.start_balance, main.days_back BETWEEN 1 AND 7 AND main.ac_to != '') as avg_amount_per_recipient_7d,
    maxIf(main.start_balance, main.days_back BETWEEN 1 AND 7 AND main.ac_to != '') as max_amount_to_single_recipient_7d,
    if(count(main.ac_from) > 0, max(rs.recipient_count) / count(main.ac_from), 0) as recipient_concentration_ratio_7d,
    
    -- Behavioral features (7-day)
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
    FROM public.stixor_iar_distributed
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


def process_date_range(start_date, end_date):
    """Process all dates in the given range"""
    print(f"\n{'='*80}")
    print(f"USER-LEVEL (AC_FROM) FEATURE INSERTION")
    print(f"{'='*80}")
    print(f"📅 Date Range: {start_date} to {end_date}")
    
    # Generate date list
    dates_to_process = []
    current = start_date
    while current <= end_date:
        dates_to_process.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print(f"📊 Total dates to process: {len(dates_to_process)}")
    print(f"🎯 Target table: public.ac_from_features_distributed")
    print(f"📈 Features: 38 user-level aggregated features (3d + 7d)")
    print(f"   • 3-day: 9 aggregate features")
    print(f"   • 7-day: 29 behavioral features")
    
    # Initialize ClickHouse client
    client = get_clickhouse_client()
    
    # Process each date
    total_start_time = time.time()
    successful_dates = []
    failed_dates = []
    
    for idx, cutoff_date in enumerate(dates_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Processing {idx}/{len(dates_to_process)}: {cutoff_date}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            query = build_insert_query(cutoff_date, lookback_days=7)
            
            print(f"🔄 Executing INSERT query...")
            client.execute(query)
            
            elapsed = time.time() - start_time
            print(f"✅ Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            successful_dates.append(cutoff_date)
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ Failed after {elapsed:.2f} seconds")
            print(f"   Error: {str(e)}")
            failed_dates.append((cutoff_date, str(e)))
    
    # Summary
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"✅ Successful: {len(successful_dates)}/{len(dates_to_process)}")
    print(f"❌ Failed: {len(failed_dates)}/{len(dates_to_process)}")
    
    if failed_dates:
        print(f"\n❌ Failed dates:")
        for date_str, error in failed_dates:
            print(f"   - {date_str}: {error[:100]}...")
    
    # Final statistics
    try:
        total_records = client.execute("SELECT count(*) FROM public.ac_from_features_distributed")[0][0]
        print(f"\n📊 Total user records in table: {total_records:,}")
        
        unique_users = client.execute("SELECT uniq(ac_from) FROM public.ac_from_features_distributed")[0][0]
        print(f"👥 Unique users: {unique_users:,}")
    except Exception as e:
        print(f"\n⚠️  Could not retrieve statistics: {str(e)}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python insert_ac_from_features.py START_DATE END_DATE")
        print("Example: python insert_ac_from_features.py 2025-09-01 2025-09-30")
        sys.exit(1)
    
    try:
        start_date = parse_date(sys.argv[1])
        end_date = parse_date(sys.argv[2])
        
        if start_date > end_date:
            print("❌ Error: Start date must be before or equal to end date")
            sys.exit(1)
        
        process_date_range(start_date, end_date)
        
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
