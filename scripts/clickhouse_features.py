# %%
# ClickHouse Connection Setup
from clickhouse_driver import Client
import pandas as pd
import configparser
from pathlib import Path

# Load configuration from config file
config = configparser.ConfigParser()
config_path = Path('/root/research-dir/dev/jazzcash-fraud-detection/config/clickhouse_config.ini')

if config_path.exists():
    config.read(config_path)
    CLICKHOUSE_CONFIG = {
        'host': config['clickhouse']['host'],
        'port': int(config['clickhouse']['port']),
        'database': config['clickhouse']['database'],
        'user': config['clickhouse']['user'],
        'password': config['clickhouse']['password']
    }
    print("✅ Configuration loaded from clickhouse_config.ini")
else:
    # Fallback to hardcoded config
    CLICKHOUSE_CONFIG = {
        'host': 'localhost',
        'port': 9000,
        'database': 'public',
        'user': 'default',
        'password': 'DfsTeChB1'
    }
    print("⚠️  Config file not found, using default configuration")

print("\n🔌 Connecting to ClickHouse...")
print(f"📍 Host: {CLICKHOUSE_CONFIG['host']}:{CLICKHOUSE_CONFIG['port']}")
print(f"🗄️  Database: {CLICKHOUSE_CONFIG['database']}")

# Create ClickHouse client
clickhouse_client = Client(
    host=CLICKHOUSE_CONFIG['host'],
    port=CLICKHOUSE_CONFIG['port'],
    database=CLICKHOUSE_CONFIG['database'],
    user=CLICKHOUSE_CONFIG['user'],
    password=CLICKHOUSE_CONFIG['password'],
    settings={
        'max_execution_time': 7200,  # 2 hour timeout for batch processing
        'send_timeout': 600,
        'receive_timeout': 600,
        'connect_timeout': 10
    }
)
    

# %%
from datetime import datetime, timedelta
import time
from datetime import date
dates_to_process = []
start = date(2025, 2,16)
end = date(2025, 4, 30)  # April has 30 days

delta = end - start
for i in range(delta.days + 1):
    d = start + timedelta(days=i)
    dates_to_process.append(d.strftime('%Y-%m-%d'))

print(f"📅 Total dates to process: {len(dates_to_process)}")
print(f"First date: {dates_to_process[0]}")
print(f"Last date: {dates_to_process[-1]}")
print(f"\nExcluded dates: 2025-06-01, 2025-06-05, 2025-06-10, 2025-06-15")
print(f"\nStarting batch processing...")

total_start_time = time.time()
successful_dates = []
failed_dates = []

for idx, CUTOFF_DATE in enumerate(dates_to_process, 1):
    print(f"\n{'='*80}")
    print(f"Processing {idx}/{len(dates_to_process)}: {CUTOFF_DATE}")
    print(f"{'='*80}")
    
    try:
        # CORRECTED: Final combined query with ALL user features (3d + 7d), MBAR, and fraud labels
        corrected_final_query = f"""
        INSERT INTO public.stixor_fraud_features_distributed
        SELECT 
            -- Transaction identifiers
            t.trans_id,
            t.ac_from,
            t.ac_to,
            t.data_date,
            t.trans_initiate_time,
            toDate('{CUTOFF_DATE}') as cutoff_date,
            
            -- FRAUD LABEL: 1 if trans_id exists in fraud table, 0 otherwise
            -- FIXED: Check for empty string ('') instead of NULL because ClickHouse LEFT JOIN returns '' for non-matches
            if(f.trans_id != '', 1, 0) as fraud_flag,
            
            -- Original transaction attributes
            t.trx_channel,
            t.trx_type,
            t.start_balance,
            t.end_balance,
            t.trx_amt,
            
            -- MBAR Account Information (ALL 24 columns)
            coalesce(m.a_c_reference, '') as mbar_a_c_reference,
            coalesce(m.region, '') as mbar_region,
            coalesce(m.city, '') as mbar_city,
            coalesce(m.registered_channel, '') as mbar_registered_channel,
            coalesce(m.registered_date_time, toDateTime('1970-01-01 00:00:00')) as mbar_registered_date_time,
            coalesce(m.a_c_status, '') as mbar_a_c_status,
            coalesce(m.a_c_level, '') as mbar_a_c_level,
            coalesce(m.agent_group, '') as mbar_agent_group,
            coalesce(m.limit_group, '') as mbar_limit_group,
            coalesce(m.charge_profile, '') as mbar_charge_profile,
            coalesce(m.credit_dl_ml_yl, 0) as mbar_credit_dl_ml_yl,
            coalesce(m.debit_dl_ml_yl, 0) as mbar_debit_dl_ml_yl,
            coalesce(m.year_of_birth, 0) as mbar_year_of_birth,
            coalesce(m.last_modified_date_time, toDateTime('1970-01-01 00:00:00')) as mbar_last_modified_date_time,
            coalesce(m.dormant_date, toDate('1970-01-01')) as mbar_dormant_date,
            coalesce(m.re_active_date, toDate('1970-01-01')) as mbar_re_active_date,
            coalesce(m.place_of_birth, '') as mbar_place_of_birth,
            coalesce(m.account_type_name, '') as mbar_account_type_name,
            coalesce(m.mpin_status, '') as mbar_mpin_status,
            coalesce(m.filer, '') as mbar_filer,
            coalesce(m.prov, '') as mbar_prov,
            coalesce(m.year_mdob, '') as mbar_year_mdob,
            coalesce(m.gmsisdn, '') as mbar_gmsisdn,
            coalesce(m.trust_level, '') as mbar_trust_level,
            
            -- Transaction-level time-based features
            t.hour_of_day,
            t.day_of_week,
            t.is_weekend,
            t.is_night,
            t.is_business_hours,
            t.is_unusual_hour,
            
            -- Transaction-level risk indicators
            t.night_weekend_combo,
            
            -- Transaction-level balance features
            t.start_balance_log,
            t.balance_change,
            t.balance_change_pct,
            
            -- Transaction-level 3-day historical features
            t.txns_3d as txn_txns_3d,
            t.total_amount_3d as txn_total_amount_3d,
            t.avg_amount_3d as txn_avg_amount_3d,
            t.max_amount_3d as txn_max_amount_3d,
            t.min_amount_3d as txn_min_amount_3d,
            t.unique_recipients_3d as txn_unique_recipients_3d,
            t.unique_channels_3d as txn_unique_channels_3d,
            t.unique_types_3d as txn_unique_types_3d,
            t.is_high_activity_3d as txn_is_high_activity_3d,
            t.multi_channel_recent as txn_multi_channel_recent,
            t.amount_deviation_from_avg as txn_amount_deviation_from_avg,
            t.night_txns_3d as txn_night_txns_3d,
            t.weekend_txns_3d as txn_weekend_txns_3d,
            
            -- Transaction-level channel one-hot encoding
            t.channel_new_jc_app,
            t.channel_ussd,
            t.channel_ussd_api,
            t.channel_payment_gateway,
            t.channel_mobile_app,
            
            -- Transaction-level type one-hot encoding
            t.type_transfer_c2c,
            t.type_transfer_c2b,
            t.type_bill_payment,
            t.type_mobile_load,
            
            -- User-level 3-day aggregate features (9 features)
            coalesce(u.total_txns_3d, 0) as user_total_txns_3d,
            coalesce(u.total_amount_3d, 0) as user_total_amount_3d,
            coalesce(u.avg_amount_3d, 0) as user_avg_amount_3d,
            coalesce(u.median_amount_3d, 0) as user_median_amount_3d,
            coalesce(u.max_amount_3d, 0) as user_max_amount_3d,
            coalesce(u.min_amount_3d, 0) as user_min_amount_3d,
            coalesce(u.unique_recipients_3d, 0) as user_unique_recipients_3d,
            coalesce(u.unique_channels_3d, 0) as user_unique_channels_3d,
            coalesce(u.unique_types_3d, 0) as user_unique_types_3d,
            
            -- User-level 7-day aggregate features (9 features)
            coalesce(u.total_txns_7d, 0) as user_total_txns_7d,
            coalesce(u.total_amount_7d, 0) as user_total_amount_7d,
            coalesce(u.avg_amount_7d, 0) as user_avg_amount_7d,
            coalesce(u.median_amount_7d, 0) as user_median_amount_7d,
            coalesce(u.max_amount_7d, 0) as user_max_amount_7d,
            coalesce(u.min_amount_7d, 0) as user_min_amount_7d,
            coalesce(u.unique_recipients_7d, 0) as user_unique_recipients_7d,
            coalesce(u.unique_channels_7d, 0) as user_unique_channels_7d,
            coalesce(u.unique_types_7d, 0) as user_unique_types_7d,
            
            -- User-level channel features (7-day)
            coalesce(u.most_used_channel_7d, '') as user_most_used_channel_7d,
            coalesce(u.last_used_channel, '') as user_last_used_channel,
            coalesce(u.channel_diversity_score_7d, 0) as user_channel_diversity_score_7d,
            
            -- User-level type features (7-day)
            coalesce(u.most_used_type_7d, '') as user_most_used_type_7d,
            coalesce(u.last_used_type, '') as user_last_used_type,
            coalesce(u.type_diversity_score_7d, 0) as user_type_diversity_score_7d,
            
            -- User-level time-based features (7-day)
            coalesce(u.night_txns_7d, 0) as user_night_txns_7d,
            coalesce(u.weekend_txns_7d, 0) as user_weekend_txns_7d,
            coalesce(u.peak_hour_txns_7d, 0) as user_peak_hour_txns_7d,
            coalesce(u.off_peak_hour_txns_7d, 0) as user_off_peak_hour_txns_7d,
            
            -- User-level balance features (7-day)
            coalesce(u.avg_start_balance_7d, 0) as user_avg_start_balance_7d,
            coalesce(u.avg_end_balance_7d, 0) as user_avg_end_balance_7d,
            coalesce(u.min_balance_7d, 0) as user_min_balance_7d,
            coalesce(u.max_balance_7d, 0) as user_max_balance_7d,
            coalesce(u.balance_volatility_7d, 0) as user_balance_volatility_7d,
            
            -- User-level recipient features (7-day)
            coalesce(u.top_recipient_7d, '') as user_top_recipient_7d,
            coalesce(u.avg_amount_per_recipient_7d, 0) as user_avg_amount_per_recipient_7d,
            coalesce(u.max_amount_to_single_recipient_7d, 0) as user_max_amount_to_single_recipient_7d,
            coalesce(u.recipient_concentration_ratio_7d, 0) as user_recipient_concentration_ratio_7d,
            
            -- User-level behavioral features (7-day)
            coalesce(u.avg_time_between_txns_7d, 0) as user_avg_time_between_txns_7d,
            coalesce(u.txn_frequency_score_7d, 0) as user_txn_frequency_score_7d,
            coalesce(u.first_txn_time, toDateTime('1970-01-01 00:00:00')) as user_first_txn_time,
            coalesce(u.last_txn_time, toDateTime('1970-01-01 00:00:00')) as user_last_txn_time,
            coalesce(u.days_since_last_txn, 0) as user_days_since_last_txn,
            
            -- Metadata
            now() as processing_timestamp,
            today() as created_at

        FROM (SELECT * FROM public.transaction_features_distributed 
        WHERE cutoff_date = toDate('{CUTOFF_DATE}') ) AS t

        -- Join with user-level features (ALL 38 features: 9 x 3d + 29 x 7d)
        GLOBAL LEFT JOIN public.ac_from_features_distributed AS u
            ON t.ac_from = u.ac_from
            AND u.cutoff_date = toDate('{CUTOFF_DATE}')

        -- Join with MBAR data (24 columns)
        GLOBAL LEFT JOIN (
            SELECT DISTINCT
                a_c_reference,
                region,
                city,
                registered_channel,
                registered_date_time,
                a_c_status,
                a_c_level,
                agent_group,
                limit_group,
                charge_profile,
                credit_dl_ml_yl,
                debit_dl_ml_yl,
                year_of_birth,
                last_modified_date_time,
                dormant_date,
                re_active_date,
                place_of_birth,
                account_type_name,
                mpin_status,
                filer,
                prov,
                year_mdob,
                gmsisdn,
                trust_level
            FROM public.stixor_mbar_v_distributed
        ) AS m
            ON t.ac_from = m.a_c_reference

        -- Join with fraud labels
        GLOBAL LEFT JOIN (
            SELECT DISTINCT trans_id
            FROM public.fraud_distributed
        ) AS f
            ON t.trans_id = f.trans_id
        """
        
        print("\n📝 CORRECTED combined query with ALL user features prepared")
        print(f"   • Query length: {len(corrected_final_query):,} characters")
        print(f"   • Target table: public.combined_features_distributed")
        print(f"   • Cutoff date filter: {CUTOFF_DATE}")
        print(f"\n   🔗 Join Operations:")
        print(f"     1. transaction_features ⟶ ac_from_features (ALL 38 user features) [filtered by cutoff_date]")
        print(f"     2. transaction_features ⟶ stixor_mbar_v (24 MBAR columns)")
        print(f"     3. transaction_features ⟶ fraud (fraud labels)")
        print(f"\n   📊 Complete Feature Summary:")
        print(f"     • Fraud Flag: 1 column (target)")
        print(f"     • Transaction features: 39 columns")
        print(f"     • User 3d features: 9 columns")
        print(f"     • User 7d features: 29 columns (NOW INCLUDED!)")
        print(f"     • MBAR features: 24 columns")
        print(f"     • Total: 1 + 39 + 9 + 29 + 24 = 102 columns")
        print(f"\n   ✅ FIXED: Now includes ALL user-level features (both 3d and 7d)")
        print(f"   ✅ FILTERED: Both transaction and user features filtered by cutoff_date = {CUTOFF_DATE}")
        
        start_time = time.time()
        
        # # DELETE existing data for this cutoff_date (INSERT OVERWRITE behavior)
        # delete_query = f"ALTER TABLE public.combined_features_distributed DELETE WHERE cutoff_date = toDate('{CUTOFF_DATE}')"
        # try:
        #     print(f"\n🗑️  Deleting existing records for cutoff_date = {CUTOFF_DATE}...")
        #     clickhouse_client.execute(delete_query)
        #     print(f"   ✅ Existing records deleted (if any)")
        # except Exception as delete_error:
        #     print(f"   ⚠️  Delete warning: {str(delete_error)} (continuing...)")
        
        # Execute the INSERT query
        clickhouse_client.execute(corrected_final_query)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Combined feature engineering completed successfully!")
        print(f"⏱️  Execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        successful_dates.append(CUTOFF_DATE)
        
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"\n❌ Combined feature engineering failed after {elapsed_time:.2f} seconds")
        print(f"   Error: {str(e)}")
        failed_dates.append((CUTOFF_DATE, str(e)))

# Summary
total_elapsed_time = time.time() - total_start_time
print(f"\n{'='*80}")
print(f"BATCH PROCESSING COMPLETE")
print(f"{'='*80}")
print(f"⏱️  Total execution time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
print(f"✅ Successful: {len(successful_dates)}/{len(dates_to_process)}")
print(f"❌ Failed: {len(failed_dates)}/{len(dates_to_process)}")

if failed_dates:
    print(f"\nFailed dates:")
    for date, error in failed_dates:
        print(f"   - {date}: {error[:100]}...")
   