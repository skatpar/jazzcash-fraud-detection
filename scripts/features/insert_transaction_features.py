#!/usr/bin/env python3
"""
Insert Transaction Features Script
Processes transaction-level features for a date range and inserts into ClickHouse.

Usage:
    python insert_transaction_features.py 2025-09-01 2025-09-30
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


def build_insert_query(txn_date):
    """Build the INSERT query for transaction-level features"""
    return f"""
INSERT INTO public.transaction_features_distributed
SELECT 
    -- Identifiers
    curr.trans_id,
    curr.ac_from,
    curr.ac_to,
    curr.data_date,
    curr.trans_initiate_time,
    toDate('{txn_date}') as cutoff_date,
    
    -- Original transaction attributes
    curr.trx_channel,
    curr.trx_type,
    curr.start_balance,
    curr.end_balance,
    curr.trx_amt,
    
    -- Time-based features
    toHour(curr.trans_initiate_time) as hour_of_day,
    toDayOfWeek(curr.trans_initiate_time) as day_of_week,
    if(toDayOfWeek(curr.trans_initiate_time) IN (6, 7), 1, 0) as is_weekend,
    if(toHour(curr.trans_initiate_time) >= 22 OR toHour(curr.trans_initiate_time) <= 6, 1, 0) as is_night,
    if(toHour(curr.trans_initiate_time) >= 9 AND toHour(curr.trans_initiate_time) <= 17, 1, 0) as is_business_hours,
    if(toHour(curr.trans_initiate_time) < 6 OR toHour(curr.trans_initiate_time) > 23, 1, 0) as is_unusual_hour,
    
    -- Risk indicators (current transaction)
    if((toHour(curr.trans_initiate_time) >= 22 OR toHour(curr.trans_initiate_time) <= 6) 
       AND toDayOfWeek(curr.trans_initiate_time) IN (6, 7), 1, 0) as night_weekend_combo,
    
    -- Balance features
    log(greatest(curr.start_balance, 1)) as start_balance_log,
    curr.end_balance - curr.start_balance as balance_change,
    if(curr.start_balance > 0, (curr.end_balance - curr.start_balance) / curr.start_balance, 0) as balance_change_pct,
    
    -- 3-day historical features (excluding current transaction day)
    coalesce(hist.txns_3d, 0) as txns_3d,
    coalesce(hist.total_amount_3d, 0) as total_amount_3d,
    coalesce(hist.avg_amount_3d, 0) as avg_amount_3d,
    coalesce(hist.max_amount_3d, 0) as max_amount_3d,
    coalesce(hist.min_amount_3d, 0) as min_amount_3d,
    coalesce(hist.unique_recipients_3d, 0) as unique_recipients_3d,
    coalesce(hist.unique_channels_3d, 0) as unique_channels_3d,
    coalesce(hist.unique_types_3d, 0) as unique_types_3d,
    if(coalesce(hist.txns_3d, 0) > 10, 1, 0) as is_high_activity_3d,
    if(coalesce(hist.unique_channels_3d, 0) > 1, 1, 0) as multi_channel_recent,
    if(coalesce(hist.avg_amount_3d, 0) > 0, 
       (curr.trx_amt - hist.avg_amount_3d) / hist.avg_amount_3d, 0) as amount_deviation_from_avg,
    coalesce(hist.night_txns_3d, 0) as night_txns_3d,
    coalesce(hist.weekend_txns_3d, 0) as weekend_txns_3d,
    
    -- Channel one-hot encoding
    if(curr.trx_channel = 'NEW_JC_APP', 1, 0) as channel_new_jc_app,
    if(curr.trx_channel = 'USSD', 1, 0) as channel_ussd,
    if(curr.trx_channel = 'USSD_API', 1, 0) as channel_ussd_api,
    if(curr.trx_channel = 'Payment Gateway', 1, 0) as channel_payment_gateway,
    if(curr.trx_channel = 'Mobile App', 1, 0) as channel_mobile_app,
    
    -- Type one-hot encoding
    if(curr.trx_type = 'Transfer(C2C)', 1, 0) as type_transfer_c2c,
    if(curr.trx_type = 'Transfer(C2B)', 1, 0) as type_transfer_c2b,
    if(curr.trx_type = 'Bill Payment', 1, 0) as type_bill_payment,
    if(curr.trx_type LIKE '%Load%', 1, 0) as type_mobile_load,
    
    -- Metadata
    now() as processing_timestamp,
    today() as created_at

FROM (
    SELECT *
    FROM public.stixor_iar_distributed
    WHERE data_date = toDate('{txn_date}')
      AND ac_from != ''
) AS curr

GLOBAL LEFT JOIN (
    SELECT 
        ac_from,
        count() as txns_3d,
        sum(trx_amt) as total_amount_3d,
        avg(trx_amt) as avg_amount_3d,
        max(trx_amt) as max_amount_3d,
        min(trx_amt) as min_amount_3d,
        uniq(ac_to) as unique_recipients_3d,
        uniq(trx_channel) as unique_channels_3d,
        uniq(trx_type) as unique_types_3d,
        sumIf(1, toHour(trans_initiate_time) >= 22 OR toHour(trans_initiate_time) <= 6) as night_txns_3d,
        sumIf(1, toDayOfWeek(trans_initiate_time) IN (6, 7)) as weekend_txns_3d
    FROM public.stixor_iar_distributed
    WHERE data_date >= toDate('{txn_date}') - INTERVAL 3 DAY
      AND data_date < toDate('{txn_date}')
      AND ac_from != ''
    GROUP BY ac_from
) AS hist ON curr.ac_from = hist.ac_from
"""


def process_date_range(start_date, end_date):
    """Process all dates in the given range"""
    print(f"\n{'='*80}")
    print(f"TRANSACTION-LEVEL FEATURE INSERTION")
    print(f"{'='*80}")
    print(f"📅 Date Range: {start_date} to {end_date}")
    
    # Generate date list
    dates_to_process = []
    current = start_date
    while current <= end_date:
        dates_to_process.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print(f"📊 Total dates to process: {len(dates_to_process)}")
    print(f"🎯 Target table: public.transaction_features_distributed")
    print(f"📈 Features: 39 transaction-level features")
    print(f"   • Time-based: 6 features")
    print(f"   • Balance: 3 features")
    print(f"   • 3-day historical: 13 features")
    print(f"   • One-hot encodings: 9 features (5 channels + 4 types)")
    print(f"   • Risk indicators: 1 feature")
    
    # Initialize ClickHouse client
    client = get_clickhouse_client()
    
    # Process each date
    total_start_time = time.time()
    successful_dates = []
    failed_dates = []
    
    for idx, txn_date in enumerate(dates_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Processing {idx}/{len(dates_to_process)}: {txn_date}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            query = build_insert_query(txn_date)
            
            print(f"🔄 Executing INSERT query...")
            client.execute(query)
            
            elapsed = time.time() - start_time
            print(f"✅ Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            successful_dates.append(txn_date)
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ Failed after {elapsed:.2f} seconds")
            print(f"   Error: {str(e)}")
            failed_dates.append((txn_date, str(e)))
    
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
        total_records = client.execute("SELECT count(*) FROM public.transaction_features_distributed")[0][0]
        print(f"\n📊 Total transaction records in table: {total_records:,}")
        
        date_range = client.execute(
            "SELECT min(data_date), max(data_date) FROM public.transaction_features_distributed"
        )[0]
        print(f"📅 Date range in table: {date_range[0]} to {date_range[1]}")
    except Exception as e:
        print(f"\n⚠️  Could not retrieve statistics: {str(e)}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python insert_transaction_features.py START_DATE END_DATE")
        print("Example: python insert_transaction_features.py 2025-09-01 2025-09-30")
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
