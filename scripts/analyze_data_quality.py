#!/usr/bin/env python3
"""
Fraud Features Data Quality Analysis Script
Executes comprehensive data quality checks on the fraud features table
"""

import clickhouse_connect
import pandas as pd
from datetime import datetime
import os
import json

# ClickHouse connection configuration
CH_CONFIG = {
    'host': '10.205.161.108',
    'port': 8123,
    'username': 'default',
    'password': '',
    'database': 'default'
}

# Output directory
OUTPUT_DIR = '../analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def connect_to_clickhouse():
    """Establish connection to ClickHouse"""
    print("Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host=CH_CONFIG['host'],
        port=CH_CONFIG['port'],
        username=CH_CONFIG['username'],
        password=CH_CONFIG['password'],
        database=CH_CONFIG['database']
    )
    print(f"✅ Connected to ClickHouse at {CH_CONFIG['host']}:{CH_CONFIG['port']}")
    return client

def run_query(client, query, query_name):
    """Execute a query and return results as DataFrame"""
    print(f"\n{'='*80}")
    print(f"Running: {query_name}")
    print(f"{'='*80}")
    try:
        result = client.query_df(query)
        print(f"✅ Query completed: {len(result)} rows returned")
        return result
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")
        return None

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FRAUD FEATURES DATA QUALITY ANALYSIS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to ClickHouse
    client = connect_to_clickhouse()
    
    # Store all results
    results = {}
    
    # 1. NULL Values Analysis
    query_1 = """
    SELECT 
        'NULL Values Count' as check_type,
        countIf(trans_id IS NULL) as trans_id_nulls,
        countIf(cutoff_date IS NULL) as cutoff_date_nulls,
        countIf(fraud_flag IS NULL) as fraud_flag_nulls,
        countIf(trx_channel IS NULL) as trx_channel_nulls,
        countIf(trx_type IS NULL) as trx_type_nulls,
        countIf(start_balance IS NULL) as start_balance_nulls,
        countIf(end_balance IS NULL) as end_balance_nulls,
        countIf(trx_amt IS NULL) as trx_amt_nulls
    FROM stixor_fraud_features_distributed
    """
    results['null_values'] = run_query(client, query_1, "NULL Values Analysis")
    
    # 2. Infinite Values Analysis
    query_2 = """
    SELECT 
        'Infinite Values' as check_type,
        countIf(isInfinite(start_balance)) as start_balance_inf,
        countIf(isInfinite(end_balance)) as end_balance_inf,
        countIf(isInfinite(trx_amt)) as trx_amt_inf,
        countIf(isInfinite(distance_mean)) as distance_mean_inf,
        countIf(isInfinite(distance_std)) as distance_std_inf
    FROM stixor_fraud_features_distributed
    """
    results['infinite_values'] = run_query(client, query_2, "Infinite Values Analysis")
    
    # 3. NaN Values Analysis
    query_3 = """
    SELECT 
        'NaN Values' as check_type,
        countIf(isNaN(start_balance)) as start_balance_nan,
        countIf(isNaN(end_balance)) as end_balance_nan,
        countIf(isNaN(trx_amt)) as trx_amt_nan,
        countIf(isNaN(distance_mean)) as distance_mean_nan,
        countIf(isNaN(distance_std)) as distance_std_nan
    FROM stixor_fraud_features_distributed
    """
    results['nan_values'] = run_query(client, query_3, "NaN Values Analysis")
    
    # 4. Negative Values Check
    query_4 = """
    SELECT 
        'Negative Values' as check_type,
        countIf(trx_amt < 0) as negative_trx_amt,
        countIf(trx_count_1d < 0) as negative_trx_count_1d,
        countIf(trx_count_7d < 0) as negative_trx_count_7d,
        countIf(trx_count_30d < 0) as negative_trx_count_30d,
        countIf(distance_mean < 0) as negative_distance_mean
    FROM stixor_fraud_features_distributed
    """
    results['negative_values'] = run_query(client, query_4, "Negative Values Check")
    
    # 5. Fraud Flag Validation
    query_5 = """
    SELECT 
        'Fraud Flag' as check_type,
        count(*) as total_records,
        countIf(fraud_flag = 0) as fraud_flag_0,
        countIf(fraud_flag = 1) as fraud_flag_1,
        countIf(fraud_flag NOT IN (0, 1)) as invalid_fraud_flag,
        countIf(fraud_flag IS NULL) as null_fraud_flag
    FROM stixor_fraud_features_distributed
    """
    results['fraud_flag'] = run_query(client, query_5, "Fraud Flag Validation")
    
    # 6. Date Range Validation
    query_6 = """
    SELECT 
        'Date Range' as check_type,
        min(cutoff_date) as earliest_date,
        max(cutoff_date) as latest_date,
        countIf(cutoff_date < '2020-01-01') as dates_before_2020,
        countIf(cutoff_date > today()) as dates_in_future,
        countIf(cutoff_date IS NULL) as null_dates
    FROM stixor_fraud_features_distributed
    """
    results['date_range'] = run_query(client, query_6, "Date Range Validation")
    
    # 7. Hour and Day Validation
    query_7 = """
    SELECT 
        'Time Validation' as check_type,
        countIf(hour_of_day < 0 OR hour_of_day > 23) as invalid_hours,
        countIf(day_of_week < 0 OR day_of_week > 7) as invalid_days,
        min(hour_of_day) as min_hour,
        max(hour_of_day) as max_hour,
        min(day_of_week) as min_day,
        max(day_of_week) as max_day
    FROM stixor_fraud_features_distributed
    """
    results['time_validation'] = run_query(client, query_7, "Hour/Day Validation")
    
    # 8. Transaction Count Consistency
    query_8 = """
    SELECT 
        'Transaction Counts' as check_type,
        countIf(trx_count_1d > trx_count_7d) as count_1d_gt_7d,
        countIf(trx_count_7d > trx_count_30d) as count_7d_gt_30d,
        countIf(trx_count_1d > 1000) as extreme_count_1d,
        countIf(trx_count_30d > 20000) as extreme_count_30d
    FROM stixor_fraud_features_distributed
    """
    results['trx_count_consistency'] = run_query(client, query_8, "Transaction Count Consistency")
    
    # 9. Transaction Amount Consistency
    query_9 = """
    SELECT 
        'Transaction Amounts' as check_type,
        countIf(trx_amt_sum_1d > trx_amt_sum_7d) as amt_1d_gt_7d,
        countIf(trx_amt_sum_7d > trx_amt_sum_30d) as amt_7d_gt_30d,
        countIf(trx_amt > 10000000) as extreme_trx_amt,
        countIf(trx_amt = 0) as zero_trx_amt
    FROM stixor_fraud_features_distributed
    """
    results['trx_amt_consistency'] = run_query(client, query_9, "Transaction Amount Consistency")
    
    # 10. Duplicate Transaction IDs
    query_10 = """
    SELECT 
        'Duplicates' as check_type,
        count(*) as total_records,
        count(DISTINCT trans_id) as unique_trans_ids,
        count(*) - count(DISTINCT trans_id) as duplicate_count
    FROM stixor_fraud_features_distributed
    """
    results['duplicates'] = run_query(client, query_10, "Duplicate Check")
    
    # 11. Summary Statistics
    query_11 = """
    SELECT 
        count(*) as total_records,
        avg(trx_amt) as avg_trx_amt,
        stddevPop(trx_amt) as std_trx_amt,
        min(trx_amt) as min_trx_amt,
        max(trx_amt) as max_trx_amt,
        quantile(0.25)(trx_amt) as q1_trx_amt,
        quantile(0.50)(trx_amt) as median_trx_amt,
        quantile(0.75)(trx_amt) as q3_trx_amt
    FROM stixor_fraud_features_distributed
    """
    results['summary_stats'] = run_query(client, query_11, "Summary Statistics")
    
    # 12. Categorical Value Distributions
    query_12 = """
    SELECT 
        trx_channel,
        count(*) as record_count,
        countIf(fraud_flag = 1) as fraud_count,
        round(countIf(fraud_flag = 1) * 100.0 / count(*), 2) as fraud_rate_pct
    FROM stixor_fraud_features_distributed
    GROUP BY trx_channel
    ORDER BY record_count DESC
    """
    results['trx_channel_dist'] = run_query(client, query_12, "Transaction Channel Distribution")
    
    # 13. Overall Data Quality Score
    query_13 = """
    SELECT 
        count(*) as total_records,
        countIf(
            fraud_flag IN (0, 1) 
            AND trx_amt >= 0 
            AND NOT isInfinite(trx_amt) 
            AND NOT isNaN(trx_amt)
            AND hour_of_day BETWEEN 0 AND 23
            AND day_of_week BETWEEN 0 AND 7
            AND trx_count_1d <= trx_count_7d
            AND trx_count_7d <= trx_count_30d
        ) as clean_records,
        round(countIf(
            fraud_flag IN (0, 1) 
            AND trx_amt >= 0 
            AND NOT isInfinite(trx_amt) 
            AND NOT isNaN(trx_amt)
            AND hour_of_day BETWEEN 0 AND 23
            AND day_of_week BETWEEN 0 AND 7
            AND trx_count_1d <= trx_count_7d
            AND trx_count_7d <= trx_count_30d
        ) * 100.0 / count(*), 2) as quality_score_pct
    FROM stixor_fraud_features_distributed
    """
    results['quality_score'] = run_query(client, query_13, "Overall Data Quality Score")
    
    # 14. Sample Problematic Records
    query_14 = """
    SELECT 
        trans_id,
        cutoff_date,
        fraud_flag,
        trx_amt,
        start_balance,
        trx_count_1d,
        trx_count_7d,
        hour_of_day
    FROM stixor_fraud_features_distributed
    WHERE 
        isInfinite(trx_amt) OR isNaN(trx_amt) OR trx_amt < 0
        OR isInfinite(start_balance) OR isNaN(start_balance)
        OR trx_count_1d > trx_count_7d
        OR hour_of_day > 23 OR hour_of_day < 0
        OR fraud_flag NOT IN (0, 1)
    LIMIT 50
    """
    results['problematic_records'] = run_query(client, query_14, "Sample Problematic Records")
    
    # Print Summary Report
    print("\n" + "="*80)
    print("DATA QUALITY SUMMARY REPORT")
    print("="*80)
    
    if results['quality_score'] is not None and len(results['quality_score']) > 0:
        score = results['quality_score'].iloc[0]
        print(f"\n📊 Overall Quality Score: {score['quality_score_pct']:.2f}%")
        print(f"   Total Records: {score['total_records']:,}")
        print(f"   Clean Records: {score['clean_records']:,}")
        print(f"   Problematic Records: {score['total_records'] - score['clean_records']:,}")
    
    # Save results to files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n💾 Saving results to {OUTPUT_DIR}/")
    for key, df in results.items():
        if df is not None and len(df) > 0:
            filename = f"{OUTPUT_DIR}/data_quality_{key}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"   ✅ {filename}")
    
    # Create summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'quality_score': float(results['quality_score'].iloc[0]['quality_score_pct']) if results['quality_score'] is not None else None,
        'total_records': int(results['quality_score'].iloc[0]['total_records']) if results['quality_score'] is not None else None,
        'checks_performed': list(results.keys())
    }
    
    summary_file = f"{OUTPUT_DIR}/data_quality_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ {summary_file}")
    
    print("\n✅ Data quality analysis complete!")
    
    # Close connection
    client.close()

if __name__ == '__main__':
    main()
