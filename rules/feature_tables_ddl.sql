-- ====================================================================
-- FRAUD DETECTION FEATURE TABLES - DDL DEFINITIONS
-- ====================================================================
-- Purpose: Create all tables needed for fraud detection feature engineering
-- Author: JazzCash Fraud Detection Team
-- Created: 2025-10-27
-- 
-- Tables Created:
--   1. ac_from_features_local/distributed - User-level aggregated features
--   2. transaction_features_local/distributed - Transaction-level features
--   3. combined_features_local/distributed - Combined features with MBAR data
--   4. combined_features_balanced_* - Balanced datasets for training
-- 
-- Usage:
--   clickhouse-client --multiquery < feature_tables_ddl.sql
-- ====================================================================

-- ====================================================================
-- 1. USER-LEVEL FEATURES (AC_FROM_FEATURES)
-- ====================================================================
-- Purpose: Store aggregated user behavior features with 3d and 7d windows
-- Granularity: One row per user per cutoff date
-- Features: 38 total (9 x 3d + 29 x 7d)
-- ====================================================================

-- Drop existing tables
DROP TABLE IF EXISTS public.ac_from_features_distributed ON CLUSTER my_cluster_2shards;
DROP TABLE IF EXISTS public.ac_from_features_local ON CLUSTER my_cluster_2shards;

-- Create local table (on each shard)
CREATE TABLE IF NOT EXISTS public.ac_from_features_local
ON CLUSTER my_cluster_2shards
(
    -- Identifiers
    ac_from String COMMENT 'User account identifier (primary key)',
    cutoff_date Date COMMENT 'Feature cutoff date for temporal filtering',
    
    -- 3-DAY WINDOW FEATURES (excluding cutoff date, days_back 1-3)
    total_txns_3d UInt32 DEFAULT 0 COMMENT 'Total transactions in 3-day window',
    total_amount_3d Float64 DEFAULT 0 COMMENT 'Total transaction amount in 3d',
    avg_amount_3d Float64 DEFAULT 0 COMMENT 'Average transaction amount in 3d',
    median_amount_3d Float64 DEFAULT 0 COMMENT 'Median transaction amount in 3d',
    max_amount_3d Float64 DEFAULT 0 COMMENT 'Maximum transaction amount in 3d',
    min_amount_3d Float64 DEFAULT 0 COMMENT 'Minimum transaction amount in 3d',
    unique_recipients_3d UInt32 DEFAULT 0 COMMENT 'Count of unique recipients in 3d',
    unique_channels_3d UInt32 DEFAULT 0 COMMENT 'Count of unique channels used in 3d',
    unique_types_3d UInt32 DEFAULT 0 COMMENT 'Count of unique transaction types in 3d',
    
    -- 7-DAY WINDOW FEATURES (excluding cutoff date, days_back 1-7)
    total_txns_7d UInt32 DEFAULT 0 COMMENT 'Total transactions in 7-day window',
    total_amount_7d Float64 DEFAULT 0 COMMENT 'Total transaction amount in 7d',
    avg_amount_7d Float64 DEFAULT 0 COMMENT 'Average transaction amount in 7d',
    median_amount_7d Float64 DEFAULT 0 COMMENT 'Median transaction amount in 7d',
    max_amount_7d Float64 DEFAULT 0 COMMENT 'Maximum transaction amount in 7d',
    min_amount_7d Float64 DEFAULT 0 COMMENT 'Minimum transaction amount in 7d',
    unique_recipients_7d UInt32 DEFAULT 0 COMMENT 'Count of unique recipients in 7d',
    unique_channels_7d UInt32 DEFAULT 0 COMMENT 'Count of unique channels used in 7d',
    unique_types_7d UInt32 DEFAULT 0 COMMENT 'Count of unique transaction types in 7d',
    
    -- CHANNEL FEATURES (7-day window)
    most_used_channel_7d String DEFAULT '' COMMENT 'Most frequently used channel in 7d',
    last_used_channel String DEFAULT '' COMMENT 'Channel used in most recent transaction',
    channel_diversity_score_7d Float64 DEFAULT 0 COMMENT 'Channel usage diversity (1 - max/total)',
    
    -- TYPE FEATURES (7-day window)
    most_used_type_7d String DEFAULT '' COMMENT 'Most frequently used transaction type in 7d',
    last_used_type String DEFAULT '' COMMENT 'Transaction type of most recent transaction',
    type_diversity_score_7d Float64 DEFAULT 0 COMMENT 'Transaction type diversity (1 - max/total)',
    
    -- TIME-BASED FEATURES (7-day window)
    night_txns_7d UInt32 DEFAULT 0 COMMENT 'Transactions during night hours (2-6 AM) in 7d',
    weekend_txns_7d UInt32 DEFAULT 0 COMMENT 'Transactions during weekends in 7d',
    peak_hour_txns_7d UInt32 DEFAULT 0 COMMENT 'Transactions during peak hours (9 AM - 5 PM) in 7d',
    off_peak_hour_txns_7d UInt32 DEFAULT 0 COMMENT 'Transactions outside peak hours in 7d',
    
    -- BALANCE FEATURES (7-day window)
    avg_start_balance_7d Float64 DEFAULT 0 COMMENT 'Average starting balance in 7d',
    avg_end_balance_7d Float64 DEFAULT 0 COMMENT 'Average ending balance in 7d',
    min_balance_7d Float64 DEFAULT 0 COMMENT 'Minimum balance (start or end) in 7d',
    max_balance_7d Float64 DEFAULT 0 COMMENT 'Maximum balance (start or end) in 7d',
    balance_volatility_7d Float64 DEFAULT 0 COMMENT 'Standard deviation of start balance in 7d',
    
    -- RECIPIENT FEATURES (7-day window)
    top_recipient_7d String DEFAULT '' COMMENT 'Most frequent recipient account in 7d',
    avg_amount_per_recipient_7d Float64 DEFAULT 0 COMMENT 'Average amount sent per recipient in 7d',
    max_amount_to_single_recipient_7d Float64 DEFAULT 0 COMMENT 'Maximum amount sent to any recipient in 7d',
    recipient_concentration_ratio_7d Float64 DEFAULT 0 COMMENT 'Ratio of transactions to top recipient vs total',
    
    -- BEHAVIORAL FEATURES (7-day window)
    avg_time_between_txns_7d Float64 DEFAULT 0 COMMENT 'Average hours between consecutive transactions in 7d',
    txn_frequency_score_7d Float64 DEFAULT 0 COMMENT 'Average daily transaction frequency in 7d',
    first_txn_time DateTime COMMENT 'Timestamp of first transaction in window',
    last_txn_time DateTime COMMENT 'Timestamp of last transaction in window',
    days_since_last_txn Int32 DEFAULT 0 COMMENT 'Days between last transaction and cutoff date',
    
    -- METADATA
    processing_timestamp DateTime DEFAULT now() COMMENT 'When this record was created',
    created_at Date DEFAULT today() COMMENT 'Date this record was created'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(cutoff_date)
ORDER BY (ac_from, cutoff_date)
SETTINGS index_granularity = 8192
COMMENT 'User-level aggregated features with 3d and 7d lookback windows';

-- Create distributed table (query across all shards)
CREATE TABLE IF NOT EXISTS public.ac_from_features_distributed AS public.ac_from_features_local
ENGINE = Distributed(my_cluster_2shards, public, ac_from_features_local, cityHash64(ac_from))
COMMENT 'Distributed table for user-level features across cluster shards';


-- ====================================================================
-- 2. TRANSACTION-LEVEL FEATURES (TRANSACTION_FEATURES)
-- ====================================================================
-- Purpose: Store transaction-level features with historical lookback
-- Granularity: One row per transaction
-- Features: Transaction attributes + lookback features + risk indicators
-- ====================================================================

-- Drop existing tables
DROP TABLE IF EXISTS public.transaction_features_distributed ON CLUSTER my_cluster_2shards;
DROP TABLE IF EXISTS public.transaction_features_local ON CLUSTER my_cluster_2shards;

-- Create local table (on each shard)
CREATE TABLE IF NOT EXISTS public.transaction_features_local
ON CLUSTER my_cluster_2shards
(
    -- IDENTIFIERS
    trans_id String COMMENT 'Unique transaction identifier (primary key)',
    ac_from String COMMENT 'Sender account',
    ac_to String COMMENT 'Recipient account',
    data_date Date COMMENT 'Transaction date',
    trans_initiate_time DateTime COMMENT 'Transaction timestamp',
    cutoff_date Date COMMENT 'Feature cutoff date for temporal filtering',
    
    -- ORIGINAL TRANSACTION ATTRIBUTES
    trx_channel String DEFAULT '' COMMENT 'Transaction channel (USSD, Mobile App, etc.)',
    trx_type String DEFAULT '' COMMENT 'Transaction type (C2C, C2B, Bill Payment, etc.)',
    start_balance Float64 DEFAULT 0 COMMENT 'Account balance before transaction',
    end_balance Float64 DEFAULT 0 COMMENT 'Account balance after transaction',
    trx_amt Float64 DEFAULT 0 COMMENT 'Transaction amount',
    
    -- TIME-BASED FEATURES
    hour_of_day UInt8 COMMENT 'Hour of day (0-23)',
    day_of_week UInt8 COMMENT 'Day of week (1=Monday, 7=Sunday)',
    is_weekend UInt8 DEFAULT 0 COMMENT 'Flag: 1 if Saturday or Sunday',
    is_night UInt8 DEFAULT 0 COMMENT 'Flag: 1 if hour between 2-6 AM',
    is_business_hours UInt8 DEFAULT 0 COMMENT 'Flag: 1 if hour between 9 AM - 5 PM',
    is_unusual_hour UInt8 DEFAULT 0 COMMENT 'Flag: 1 if hour between 12 AM - 6 AM',
    
    -- BALANCE FEATURES
    start_balance_log Float64 DEFAULT 0 COMMENT 'Log transform of start balance',
    balance_change Float64 DEFAULT 0 COMMENT 'Change in balance (end - start)',
    balance_change_pct Float64 DEFAULT 0 COMMENT 'Percentage change in balance',
    
    -- 3-DAY LOOKBACK FEATURES (historical context)
    txns_3d UInt32 DEFAULT 0 COMMENT 'Count of transactions in previous 3 days',
    total_amount_3d Float64 DEFAULT 0 COMMENT 'Total transaction amount in previous 3 days',
    avg_amount_3d Float64 DEFAULT 0 COMMENT 'Average transaction amount in previous 3 days',
    max_amount_3d Float64 DEFAULT 0 COMMENT 'Maximum transaction amount in previous 3 days',
    min_amount_3d Float64 DEFAULT 0 COMMENT 'Minimum transaction amount in previous 3 days',
    unique_recipients_3d UInt32 DEFAULT 0 COMMENT 'Unique recipients in previous 3 days',
    unique_channels_3d UInt32 DEFAULT 0 COMMENT 'Unique channels in previous 3 days',
    unique_types_3d UInt32 DEFAULT 0 COMMENT 'Unique transaction types in previous 3 days',
    night_txns_3d UInt32 DEFAULT 0 COMMENT 'Night transactions (2-6 AM) in previous 3 days',
    weekend_txns_3d UInt32 DEFAULT 0 COMMENT 'Weekend transactions in previous 3 days',
    
    -- RISK INDICATORS
    is_high_activity_3d UInt8 DEFAULT 0 COMMENT 'Flag: 1 if >10 transactions in 3d',
    multi_channel_recent UInt8 DEFAULT 0 COMMENT 'Flag: 1 if >1 channel used in 3d',
    amount_deviation_from_avg Float64 DEFAULT 0 COMMENT 'How much current txn deviates from 3d average',
    night_weekend_combo UInt8 DEFAULT 0 COMMENT 'Flag: 1 if transaction is both night and weekend',
    
    -- CHANNEL ONE-HOT ENCODING
    channel_new_jc_app UInt8 DEFAULT 0 COMMENT 'Flag: 1 if channel is New JC App',
    channel_ussd UInt8 DEFAULT 0 COMMENT 'Flag: 1 if channel is USSD',
    channel_ussd_api UInt8 DEFAULT 0 COMMENT 'Flag: 1 if channel is USSD API',
    channel_payment_gateway UInt8 DEFAULT 0 COMMENT 'Flag: 1 if channel is Payment Gateway',
    channel_mobile_app UInt8 DEFAULT 0 COMMENT 'Flag: 1 if channel is Mobile App',
    
    -- TYPE ONE-HOT ENCODING
    type_transfer_c2c UInt8 DEFAULT 0 COMMENT 'Flag: 1 if type is Transfer C2C',
    type_transfer_c2b UInt8 DEFAULT 0 COMMENT 'Flag: 1 if type is Transfer C2B',
    type_bill_payment UInt8 DEFAULT 0 COMMENT 'Flag: 1 if type is Bill Payment',
    type_mobile_load UInt8 DEFAULT 0 COMMENT 'Flag: 1 if type is Mobile Load',
    
    -- METADATA
    processing_timestamp DateTime DEFAULT now() COMMENT 'When this record was created',
    created_at Date DEFAULT today() COMMENT 'Date this record was created'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(data_date)
ORDER BY (trans_id, ac_from, data_date)
SETTINGS index_granularity = 8192
COMMENT 'Transaction-level features with historical lookback and risk indicators';

-- Create distributed table (query across all shards)
CREATE TABLE IF NOT EXISTS public.transaction_features_distributed AS public.transaction_features_local
ENGINE = Distributed(my_cluster_2shards, public, transaction_features_local, cityHash64(trans_id))
COMMENT 'Distributed table for transaction-level features across cluster shards';


-- ====================================================================
-- 3. COMBINED FEATURES (TRANSACTION + USER + MBAR + FRAUD LABELS)
-- ====================================================================
-- Purpose: Join all feature sources into a single table for model training
-- Granularity: One row per transaction
-- Features: Transaction features + User features + MBAR data + Fraud labels
-- ====================================================================

-- Drop existing tables
DROP TABLE IF EXISTS public.combined_features_distributed ON CLUSTER my_cluster_2shards;
DROP TABLE IF EXISTS public.combined_features_local ON CLUSTER my_cluster_2shards;

-- Create local table (on each shard)
CREATE TABLE IF NOT EXISTS public.combined_features_local
ON CLUSTER my_cluster_2shards
(
    -- TRANSACTION IDENTIFIERS
    trans_id String COMMENT 'Unique transaction identifier',
    ac_from String COMMENT 'Sender account',
    ac_to String COMMENT 'Recipient account',
    data_date Date COMMENT 'Transaction date',
    trans_initiate_time DateTime COMMENT 'Transaction timestamp',
    cutoff_date Date DEFAULT '2025-07-01' COMMENT 'Feature cutoff date',
    
    -- FRAUD LABEL (TARGET VARIABLE)
    fraud_flag UInt8 DEFAULT 0 COMMENT 'Target: 1 if fraud, 0 if legitimate',
    
    -- ORIGINAL TRANSACTION ATTRIBUTES
    trx_channel String DEFAULT '' COMMENT 'Transaction channel',
    trx_type String DEFAULT '' COMMENT 'Transaction type',
    start_balance Float64 DEFAULT 0 COMMENT 'Balance before transaction',
    end_balance Float64 DEFAULT 0 COMMENT 'Balance after transaction',
    trx_amt Float64 DEFAULT 0 COMMENT 'Transaction amount',
    
    -- MBAR ACCOUNT INFORMATION (24 columns from stixor_mbar_v)
    mbar_a_c_reference String DEFAULT '' COMMENT 'MBAR account reference',
    mbar_region String DEFAULT '' COMMENT 'MBAR region',
    mbar_city String DEFAULT '' COMMENT 'MBAR city',
    mbar_registered_channel String DEFAULT '' COMMENT 'MBAR registration channel',
    mbar_registered_date_time DateTime DEFAULT toDateTime('1970-01-01 00:00:00') COMMENT 'MBAR registration date',
    mbar_a_c_status String DEFAULT '' COMMENT 'MBAR account status',
    mbar_a_c_level String DEFAULT '' COMMENT 'MBAR account level',
    mbar_agent_group String DEFAULT '' COMMENT 'MBAR agent group',
    mbar_limit_group String DEFAULT '' COMMENT 'MBAR limit group',
    mbar_charge_profile String DEFAULT '' COMMENT 'MBAR charge profile',
    mbar_credit_dl_ml_yl Decimal(18, 2) DEFAULT 0 COMMENT 'MBAR credit limits',
    mbar_debit_dl_ml_yl Decimal(18, 2) DEFAULT 0 COMMENT 'MBAR debit limits',
    mbar_year_of_birth Int32 DEFAULT 0 COMMENT 'MBAR year of birth',
    mbar_last_modified_date_time DateTime DEFAULT toDateTime('1970-01-01 00:00:00') COMMENT 'MBAR last modified',
    mbar_dormant_date Date DEFAULT toDate('1970-01-01') COMMENT 'MBAR dormant date',
    mbar_re_active_date Date DEFAULT toDate('1970-01-01') COMMENT 'MBAR reactivation date',
    mbar_place_of_birth String DEFAULT '' COMMENT 'MBAR place of birth',
    mbar_account_type_name String DEFAULT '' COMMENT 'MBAR account type',
    mbar_mpin_status String DEFAULT '' COMMENT 'MBAR MPIN status',
    mbar_filer String DEFAULT '' COMMENT 'MBAR filer status',
    mbar_prov String DEFAULT '' COMMENT 'MBAR province',
    mbar_year_mdob String DEFAULT '' COMMENT 'MBAR year MDOB',
    mbar_gmsisdn String DEFAULT '' COMMENT 'MBAR GMSISDN',
    mbar_trust_level String DEFAULT '' COMMENT 'MBAR trust level',
    
    -- TRANSACTION-LEVEL TIME-BASED FEATURES
    hour_of_day UInt8 COMMENT 'Hour of transaction',
    day_of_week UInt8 COMMENT 'Day of week',
    is_weekend UInt8 DEFAULT 0 COMMENT 'Weekend flag',
    is_night UInt8 DEFAULT 0 COMMENT 'Night hours flag',
    is_business_hours UInt8 DEFAULT 0 COMMENT 'Business hours flag',
    is_unusual_hour UInt8 DEFAULT 0 COMMENT 'Unusual hours flag',
    
    -- TRANSACTION-LEVEL RISK INDICATORS
    night_weekend_combo UInt8 DEFAULT 0 COMMENT 'Night and weekend combination',
    
    -- TRANSACTION-LEVEL BALANCE FEATURES
    start_balance_log Float64 DEFAULT 0 COMMENT 'Log of start balance',
    balance_change Float64 DEFAULT 0 COMMENT 'Balance change',
    balance_change_pct Float64 DEFAULT 0 COMMENT 'Balance change percentage',
    
    -- TRANSACTION-LEVEL 3D LOOKBACK FEATURES (prefix: txn_)
    txn_txns_3d UInt32 DEFAULT 0 COMMENT 'Transaction count in 3d lookback',
    txn_total_amount_3d Float64 DEFAULT 0 COMMENT 'Total amount in 3d lookback',
    txn_avg_amount_3d Float64 DEFAULT 0 COMMENT 'Average amount in 3d lookback',
    txn_max_amount_3d Float64 DEFAULT 0 COMMENT 'Max amount in 3d lookback',
    txn_min_amount_3d Float64 DEFAULT 0 COMMENT 'Min amount in 3d lookback',
    txn_unique_recipients_3d UInt32 DEFAULT 0 COMMENT 'Unique recipients in 3d lookback',
    txn_unique_channels_3d UInt32 DEFAULT 0 COMMENT 'Unique channels in 3d lookback',
    txn_unique_types_3d UInt32 DEFAULT 0 COMMENT 'Unique types in 3d lookback',
    txn_is_high_activity_3d UInt8 DEFAULT 0 COMMENT 'High activity flag (>10 txns in 3d)',
    txn_multi_channel_recent UInt8 DEFAULT 0 COMMENT 'Multiple channels used in 3d',
    txn_amount_deviation_from_avg Float64 DEFAULT 0 COMMENT 'Deviation from 3d average',
    txn_night_txns_3d UInt32 DEFAULT 0 COMMENT 'Night transactions in 3d',
    txn_weekend_txns_3d UInt32 DEFAULT 0 COMMENT 'Weekend transactions in 3d',
    
    -- CHANNEL ONE-HOT ENCODING
    channel_new_jc_app UInt8 DEFAULT 0,
    channel_ussd UInt8 DEFAULT 0,
    channel_ussd_api UInt8 DEFAULT 0,
    channel_payment_gateway UInt8 DEFAULT 0,
    channel_mobile_app UInt8 DEFAULT 0,
    
    -- TYPE ONE-HOT ENCODING
    type_transfer_c2c UInt8 DEFAULT 0,
    type_transfer_c2b UInt8 DEFAULT 0,
    type_bill_payment UInt8 DEFAULT 0,
    type_mobile_load UInt8 DEFAULT 0,
    
    -- USER-LEVEL 3D AGGREGATE FEATURES (prefix: user_)
    user_total_txns_3d UInt32 DEFAULT 0 COMMENT 'User total transactions in 3d',
    user_total_amount_3d Float64 DEFAULT 0 COMMENT 'User total amount in 3d',
    user_avg_amount_3d Float64 DEFAULT 0 COMMENT 'User average amount in 3d',
    user_median_amount_3d Float64 DEFAULT 0 COMMENT 'User median amount in 3d',
    user_max_amount_3d Float64 DEFAULT 0 COMMENT 'User max amount in 3d',
    user_min_amount_3d Float64 DEFAULT 0 COMMENT 'User min amount in 3d',
    user_unique_recipients_3d UInt32 DEFAULT 0 COMMENT 'User unique recipients in 3d',
    user_unique_channels_3d UInt32 DEFAULT 0 COMMENT 'User unique channels in 3d',
    user_unique_types_3d UInt32 DEFAULT 0 COMMENT 'User unique types in 3d',
    
    -- USER-LEVEL 7D AGGREGATE FEATURES
    user_total_txns_7d UInt32 DEFAULT 0 COMMENT 'User total transactions in 7d',
    user_total_amount_7d Float64 DEFAULT 0 COMMENT 'User total amount in 7d',
    user_avg_amount_7d Float64 DEFAULT 0 COMMENT 'User average amount in 7d',
    user_median_amount_7d Float64 DEFAULT 0 COMMENT 'User median amount in 7d',
    user_max_amount_7d Float64 DEFAULT 0 COMMENT 'User max amount in 7d',
    user_min_amount_7d Float64 DEFAULT 0 COMMENT 'User min amount in 7d',
    user_unique_recipients_7d UInt32 DEFAULT 0 COMMENT 'User unique recipients in 7d',
    user_unique_channels_7d UInt32 DEFAULT 0 COMMENT 'User unique channels in 7d',
    user_unique_types_7d UInt32 DEFAULT 0 COMMENT 'User unique types in 7d',
    
    -- USER-LEVEL 7D CHANNEL FEATURES
    user_most_used_channel_7d String DEFAULT '' COMMENT 'Most used channel in 7d',
    user_last_used_channel String DEFAULT '' COMMENT 'Last used channel',
    user_channel_diversity_score_7d Float64 DEFAULT 0 COMMENT 'Channel diversity score',
    
    -- USER-LEVEL 7D TYPE FEATURES
    user_most_used_type_7d String DEFAULT '' COMMENT 'Most used type in 7d',
    user_last_used_type String DEFAULT '' COMMENT 'Last used type',
    user_type_diversity_score_7d Float64 DEFAULT 0 COMMENT 'Type diversity score',
    
    -- USER-LEVEL 7D TIME-BASED FEATURES
    user_night_txns_7d UInt32 DEFAULT 0 COMMENT 'User night transactions in 7d',
    user_weekend_txns_7d UInt32 DEFAULT 0 COMMENT 'User weekend transactions in 7d',
    user_peak_hour_txns_7d UInt32 DEFAULT 0 COMMENT 'User peak hour transactions in 7d',
    user_off_peak_hour_txns_7d UInt32 DEFAULT 0 COMMENT 'User off-peak transactions in 7d',
    
    -- USER-LEVEL 7D BALANCE FEATURES
    user_avg_start_balance_7d Float64 DEFAULT 0 COMMENT 'User average start balance in 7d',
    user_avg_end_balance_7d Float64 DEFAULT 0 COMMENT 'User average end balance in 7d',
    user_min_balance_7d Float64 DEFAULT 0 COMMENT 'User minimum balance in 7d',
    user_max_balance_7d Float64 DEFAULT 0 COMMENT 'User maximum balance in 7d',
    user_balance_volatility_7d Float64 DEFAULT 0 COMMENT 'User balance volatility in 7d',
    
    -- USER-LEVEL 7D RECIPIENT FEATURES
    user_top_recipient_7d String DEFAULT '' COMMENT 'User top recipient in 7d',
    user_avg_amount_per_recipient_7d Float64 DEFAULT 0 COMMENT 'User avg amount per recipient in 7d',
    user_max_amount_to_single_recipient_7d Float64 DEFAULT 0 COMMENT 'User max amount to single recipient in 7d',
    user_recipient_concentration_ratio_7d Float64 DEFAULT 0 COMMENT 'User recipient concentration in 7d',
    
    -- USER-LEVEL 7D BEHAVIORAL FEATURES
    user_avg_time_between_txns_7d Float64 DEFAULT 0 COMMENT 'User average time between transactions in 7d',
    user_txn_frequency_score_7d Float64 DEFAULT 0 COMMENT 'User transaction frequency score in 7d',
    user_first_txn_time DateTime COMMENT 'User first transaction time in window',
    user_last_txn_time DateTime COMMENT 'User last transaction time in window',
    user_days_since_last_txn Int32 DEFAULT 0 COMMENT 'Days since user last transaction',
    
    -- METADATA
    processing_timestamp DateTime DEFAULT now() COMMENT 'Record creation timestamp',
    created_at Date DEFAULT today() COMMENT 'Record creation date'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(data_date)
ORDER BY (fraud_flag, trans_id, ac_from)
SETTINGS index_granularity = 8192
COMMENT 'Combined features: Transaction + User + MBAR + Fraud labels for model training';

-- Create distributed table (query across all shards)
CREATE TABLE IF NOT EXISTS public.combined_features_distributed AS public.combined_features_local
ENGINE = Distributed(my_cluster_2shards, public, combined_features_local, cityHash64(trans_id))
COMMENT 'Distributed table for combined features across cluster shards';


-- ====================================================================
-- 4. BALANCED DATASETS (FOR MODEL TRAINING)
-- ====================================================================
-- Purpose: Create balanced versions of combined features with different ratios
-- Ratios: 1:1, 1:5, 1:10, 1:20 (fraud:legitimate)
-- Note: Tables use same schema as combined_features but with ORDER BY optimized for fraud_flag
-- ====================================================================

-- 4.1 Balanced Dataset 1:1 (Equal fraud and legitimate)
DROP TABLE IF EXISTS public.combined_features_balanced_1_1_distributed ON CLUSTER my_cluster_2shards;
DROP TABLE IF EXISTS public.combined_features_balanced_1_1_local ON CLUSTER my_cluster_2shards;

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_1_local ON CLUSTER my_cluster_2shards
AS public.combined_features_local
ENGINE = MergeTree()
PARTITION BY toYYYYMM(data_date)
ORDER BY (fraud_flag, trans_id)
COMMENT 'Balanced dataset with 1:1 ratio (fraud:legitimate)';

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_1_distributed
AS public.combined_features_balanced_1_1_local
ENGINE = Distributed(my_cluster_2shards, public, combined_features_balanced_1_1_local, cityHash64(trans_id))
COMMENT 'Distributed balanced dataset 1:1';

-- 4.2 Balanced Dataset 1:5 (5 legitimate per fraud)
DROP TABLE IF EXISTS public.combined_features_balanced_1_5_distributed ON CLUSTER my_cluster_2shards;
DROP TABLE IF EXISTS public.combined_features_balanced_1_5_local ON CLUSTER my_cluster_2shards;

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_5_local ON CLUSTER my_cluster_2shards
AS public.combined_features_local
ENGINE = MergeTree()
PARTITION BY toYYYYMM(data_date)
ORDER BY (fraud_flag, trans_id)
COMMENT 'Balanced dataset with 1:5 ratio (fraud:legitimate)';

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_5_distributed
AS public.combined_features_balanced_1_5_local
ENGINE = Distributed(my_cluster_2shards, public, combined_features_balanced_1_5_local, cityHash64(trans_id))
COMMENT 'Distributed balanced dataset 1:5';

-- 4.3 Balanced Dataset 1:10 (10 legitimate per fraud)
DROP TABLE IF EXISTS public.combined_features_balanced_1_10_distributed ON CLUSTER my_cluster_2shards;
DROP TABLE IF EXISTS public.combined_features_balanced_1_10_local ON CLUSTER my_cluster_2shards;

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_10_local ON CLUSTER my_cluster_2shards
AS public.combined_features_local
ENGINE = MergeTree()
PARTITION BY toYYYYMM(data_date)
ORDER BY (fraud_flag, trans_id)
COMMENT 'Balanced dataset with 1:10 ratio (fraud:legitimate)';

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_10_distributed
AS public.combined_features_balanced_1_10_local
ENGINE = Distributed(my_cluster_2shards, public, combined_features_balanced_1_10_local, cityHash64(trans_id))
COMMENT 'Distributed balanced dataset 1:10';

-- 4.4 Balanced Dataset 1:20 (20 legitimate per fraud)
DROP TABLE IF EXISTS public.combined_features_balanced_1_20_distributed ON CLUSTER my_cluster_2shards;
DROP TABLE IF EXISTS public.combined_features_balanced_1_20_local ON CLUSTER my_cluster_2shards;

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_20_local ON CLUSTER my_cluster_2shards
AS public.combined_features_local
ENGINE = MergeTree()
PARTITION BY toYYYYMM(data_date)
ORDER BY (fraud_flag, trans_id)
COMMENT 'Balanced dataset with 1:20 ratio (fraud:legitimate)';

CREATE TABLE IF NOT EXISTS public.combined_features_balanced_1_20_distributed
AS public.combined_features_balanced_1_20_local
ENGINE = Distributed(my_cluster_2shards, public, combined_features_balanced_1_20_local, cityHash64(trans_id))
COMMENT 'Distributed balanced dataset 1:20';


-- ====================================================================
-- VERIFICATION QUERIES
-- ====================================================================

-- Check all tables created
SELECT 
    database,
    name as table_name,
    engine,
    total_rows,
    formatReadableSize(total_bytes) as size
FROM system.tables
WHERE database = 'public'
  AND name LIKE '%features%'
ORDER BY name;

-- ====================================================================
-- END OF DDL
-- ====================================================================
