-- Updated table DDL for 1d, 3d, 7d windows only
DROP TABLE IF EXISTS public.ac_from_features_local;
DROP TABLE IF EXISTS public.ac_from_features_distributed;

CREATE TABLE IF NOT EXISTS public.ac_from_features_local
ON CLUSTER my_cluster_2shards
(
    ac_from String,
    cutoff_date Date,
    
    -- 1 day features
    total_txns_1d UInt32 DEFAULT 0,
    total_amount_1d Float64 DEFAULT 0,
    avg_amount_1d Float64 DEFAULT 0,
    median_amount_1d Float64 DEFAULT 0,
    max_amount_1d Float64 DEFAULT 0,
    min_amount_1d Float64 DEFAULT 0,
    unique_recipients_1d UInt32 DEFAULT 0,
    unique_channels_1d UInt32 DEFAULT 0,
    unique_types_1d UInt32 DEFAULT 0,
    
    -- 3 day features  
    total_txns_3d UInt32 DEFAULT 0,
    total_amount_3d Float64 DEFAULT 0,
    avg_amount_3d Float64 DEFAULT 0,
    median_amount_3d Float64 DEFAULT 0,
    max_amount_3d Float64 DEFAULT 0,
    min_amount_3d Float64 DEFAULT 0,
    unique_recipients_3d UInt32 DEFAULT 0,
    unique_channels_3d UInt32 DEFAULT 0,
    unique_types_3d UInt32 DEFAULT 0,
    
    -- 7 day features
    total_txns_7d UInt32 DEFAULT 0,
    total_amount_7d Float64 DEFAULT 0,
    avg_amount_7d Float64 DEFAULT 0,
    median_amount_7d Float64 DEFAULT 0,
    max_amount_7d Float64 DEFAULT 0,
    min_amount_7d Float64 DEFAULT 0,
    unique_recipients_7d UInt32 DEFAULT 0,
    unique_channels_7d UInt32 DEFAULT 0,
    unique_types_7d UInt32 DEFAULT 0,
    
    -- Channel preference features (7-day)
    most_used_channel_7d String,
    last_used_channel String,
    channel_diversity_score_7d Float64 DEFAULT 0,
    
    -- Transaction type features (7-day)
    most_used_type_7d String,
    last_used_type String,
    type_diversity_score_7d Float64 DEFAULT 0,
    
    -- Time-based features (7-day)
    night_txns_7d UInt32 DEFAULT 0,
    weekend_txns_7d UInt32 DEFAULT 0,
    peak_hour_txns_7d UInt32 DEFAULT 0,
    off_peak_hour_txns_7d UInt32 DEFAULT 0,
    
    -- Balance features (7-day)
    avg_start_balance_7d Float64 DEFAULT 0,
    avg_end_balance_7d Float64 DEFAULT 0,
    min_balance_7d Float64 DEFAULT 0,
    max_balance_7d Float64 DEFAULT 0,
    balance_volatility_7d Float64 DEFAULT 0,
    
    -- Recipient features (7-day)
    top_recipient_7d String,
    avg_amount_per_recipient_7d Float64 DEFAULT 0,
    max_amount_to_single_recipient_7d Float64 DEFAULT 0,
    recipient_concentration_ratio_7d Float64 DEFAULT 0,
    
    -- Behavioral features (7-day)
    avg_time_between_txns_7d Float64 DEFAULT 0,
    txn_frequency_score_7d Float64 DEFAULT 0,
    first_txn_time DateTime,
    last_txn_time DateTime,
    days_since_first_txn UInt32 DEFAULT 0,
    days_since_last_txn UInt32 DEFAULT 0,
    
    processing_timestamp DateTime DEFAULT now(),
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY ac_from
PARTITION BY toYYYYMM(processing_timestamp);

-- Create distributed table
CREATE TABLE IF NOT EXISTS public.ac_from_features_distributed AS public.ac_from_features_local
ENGINE = Distributed(my_cluster_2shards , public, ac_from_features_local, cityHash64(ac_from));


-- Updated INSERT query for users who transacted on 2025-05-31
-- Features calculated over 7-day window (2025-05-25 to 2025-05-31)
INSERT INTO public.ac_from_features_distributed
WITH 
-- Get users who transacted on cutoff date
active_users AS (
    SELECT DISTINCT ac_from
    FROM public.stixor_iar_distributed
    WHERE data_date = toDate('2025-05-31')
      AND ac_from != ''
),
-- Pre-calculate top channels and types per user (7-day window)
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
        WHERE data_date >= toDate('2025-05-25')
          AND data_date <= toDate('2025-05-31')
          AND ac_from IN (SELECT ac_from FROM active_users)
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
        WHERE data_date >= toDate('2025-05-25')
          AND data_date <= toDate('2025-05-31')
          AND ac_from IN (SELECT ac_from FROM active_users)
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
        WHERE data_date >= toDate('2025-05-25')
          AND data_date <= toDate('2025-05-31')
          AND ac_from IN (SELECT ac_from FROM active_users)
          AND ac_to != ''
    )
    GROUP BY ac_from, ac_to
)

SELECT 
    main.ac_from,
    toDate('2025-05-31') as cutoff_date,
    
    -- 1-day features (only 2025-05-31)
    sumIf(1, main.days_back = 0) as total_txns_1d,
    sumIf(main.start_balance, main.days_back = 0) as total_amount_1d,
    avgIf(main.start_balance, main.days_back = 0) as avg_amount_1d,
    quantileIf(0.5)(main.start_balance, main.days_back = 0) as median_amount_1d,
    maxIf(main.start_balance, main.days_back = 0) as max_amount_1d,
    minIf(main.start_balance, main.days_back = 0) as min_amount_1d,
    uniqIf(main.ac_to, main.days_back = 0) as unique_recipients_1d,
    uniqIf(main.trx_channel, main.days_back = 0) as unique_channels_1d,
    uniqIf(main.trx_type, main.days_back = 0) as unique_types_1d,
    
    -- 3-day features (2025-05-29 to 2025-05-31)
    sumIf(1, main.days_back <= 2) as total_txns_3d,
    sumIf(main.start_balance, main.days_back <= 2) as total_amount_3d,
    avgIf(main.start_balance, main.days_back <= 2) as avg_amount_3d,
    quantileIf(0.5)(main.start_balance, main.days_back <= 2) as median_amount_3d,
    maxIf(main.start_balance, main.days_back <= 2) as max_amount_3d,
    minIf(main.start_balance, main.days_back <= 2) as min_amount_3d,
    uniqIf(main.ac_to, main.days_back <= 2) as unique_recipients_3d,
    uniqIf(main.trx_channel, main.days_back <= 2) as unique_channels_3d,
    uniqIf(main.trx_type, main.days_back <= 2) as unique_types_3d,
    
    -- 7-day features (2025-05-25 to 2025-05-31)
    sumIf(1, main.days_back <= 6) as total_txns_7d,
    sumIf(main.start_balance, main.days_back <= 6) as total_amount_7d,
    avgIf(main.start_balance, main.days_back <= 6) as avg_amount_7d,
    quantileIf(0.5)(main.start_balance, main.days_back <= 6) as median_amount_7d,
    maxIf(main.start_balance, main.days_back <= 6) as max_amount_7d,
    minIf(main.start_balance, main.days_back <= 6) as min_amount_7d,
    uniqIf(main.ac_to, main.days_back <= 6) as unique_recipients_7d,
    uniqIf(main.trx_channel, main.days_back <= 6) as unique_channels_7d,
    uniqIf(main.trx_type, main.days_back <= 6) as unique_types_7d,
    
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
    sumIf(1, toHour(main.trans_initiate_time) IN (2,3,4,5,6) AND main.days_back <= 6) as night_txns_7d,
    sumIf(1, toDayOfWeek(main.trans_initiate_time) IN (6,7) AND main.days_back <= 6) as weekend_txns_7d,
    sumIf(1, toHour(main.trans_initiate_time) BETWEEN 9 AND 17 AND main.days_back <= 6) as peak_hour_txns_7d,
    sumIf(1, toHour(main.trans_initiate_time) NOT BETWEEN 9 AND 17 AND main.days_back <= 6) as off_peak_hour_txns_7d,
    
    -- Balance features (7-day)
    avgIf(main.start_balance, main.days_back <= 6) as avg_start_balance_7d,
    avgIf(main.end_balance, main.days_back <= 6) as avg_end_balance_7d,
    minIf(least(main.start_balance, main.end_balance), main.days_back <= 6) as min_balance_7d,
    maxIf(greatest(main.start_balance, main.end_balance), main.days_back <= 6) as max_balance_7d,
    stddevPopIf(main.start_balance, main.days_back <= 6) as balance_volatility_7d,
    
    -- Recipient features (7-day)
    anyIf(rs.ac_to, rs.recipient_rank = 1) as top_recipient_7d,
    avgIf(main.start_balance, main.days_back <= 6 AND main.ac_to != '') as avg_amount_per_recipient_7d,
    maxIf(main.start_balance, main.days_back <= 6 AND main.ac_to != '') as max_amount_to_single_recipient_7d,
    if(count(main.ac_from) > 0, max(rs.recipient_count) / count(main.ac_from), 0) as recipient_concentration_ratio_7d,
    
    -- Behavioral features (7-day)
    if(count(main.ac_from) > 1,
       dateDiff('hour', min(main.trans_initiate_time), max(main.trans_initiate_time)) / (count(main.ac_from) - 1),
       0) as avg_time_between_txns_7d,
    count(main.ac_from) / greatest(dateDiff('day', min(main.data_date), max(main.data_date)) + 1, 1) as txn_frequency_score_7d,
    min(main.trans_initiate_time) as first_txn_time,
    max(main.trans_initiate_time) as last_txn_time,
    dateDiff('day', min(main.data_date), toDate('2025-05-31')) as days_since_first_txn,
    dateDiff('day', max(main.data_date), toDate('2025-05-31')) as days_since_last_txn,
    
    now() as processing_timestamp

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
        dateDiff('day', data_date, toDate('2025-05-31')) as days_back
    FROM public.stixor_iar_distributed
    WHERE data_date >= toDate('2025-05-25')
      AND data_date <= toDate('2025-05-31')
      AND ac_from IN (SELECT ac_from FROM active_users)
    ORDER BY ac_from, trans_initiate_time
) main
LEFT JOIN user_channel_stats ch ON main.ac_from = ch.ac_from AND main.trx_channel = ch.trx_channel
LEFT JOIN user_type_stats ty ON main.ac_from = ty.ac_from AND main.trx_type = ty.trx_type  
LEFT JOIN user_recipient_stats rs ON main.ac_from = rs.ac_from AND main.ac_to = rs.ac_to
GROUP BY main.ac_from;



INSERT INTO public.transaction_features_distributed
SELECT 
    -- Identifiers
    trans_id,
    ac_from,
    ac_to,
    data_date,
    trans_initiate_time,
    
    -- Original transaction attributes
    trx_channel,
    trx_type,
    start_balance,
    end_balance,
    trx_amt,
    
    -- Time-based features
    toHour(trans_initiate_time) as hour_of_day,
    toDayOfWeek(trans_initiate_time) as day_of_week,
    if(toDayOfWeek(trans_initiate_time) IN (6, 7), 1, 0) as is_weekend,
    if(toHour(trans_initiate_time) >= 22 OR toHour(trans_initiate_time) <= 6, 1, 0) as is_night,
    if(toHour(trans_initiate_time) >= 9 AND toHour(trans_initiate_time) <= 17, 1, 0) as is_business_hours,
    if(toHour(trans_initiate_time) < 6 OR toHour(trans_initiate_time) > 23, 1, 0) as is_unusual_hour,
    
    -- Risk indicators (current transaction only)
    0 as is_high_activity_3d,
    if((toHour(trans_initiate_time) >= 22 OR toHour(trans_initiate_time) <= 6) 
       AND toDayOfWeek(trans_initiate_time) IN (6, 7), 1, 0) as night_weekend_combo,
    0 as multi_channel_recent,
    0 as amount_deviation_from_avg,
    
    -- Balance features
    log(greatest(start_balance, 1)) as start_balance_log,
    end_balance - start_balance as balance_change,
    if(start_balance > 0, (end_balance - start_balance) / start_balance, 0) as balance_change_pct,
    
    -- Channel one-hot encoding
    if(trx_channel = 'NEW_JC_APP', 1, 0) as channel_new_jc_app,
    if(trx_channel = 'USSD', 1, 0) as channel_ussd,
    if(trx_channel = 'USSD_API', 1, 0) as channel_ussd_api,
    if(trx_channel = 'Payment Gateway', 1, 0) as channel_payment_gateway,
    if(trx_channel = 'Mobile App', 1, 0) as channel_mobile_app,
    
    -- Type one-hot encoding
    if(trx_type = 'Transfer(C2C)', 1, 0) as type_transfer_c2c,
    if(trx_type = 'Transfer(C2B)', 1, 0) as type_transfer_c2b,
    if(trx_type = 'Bill Payment', 1, 0) as type_bill_payment,
    if(trx_type LIKE '%Load%', 1, 0) as type_mobile_load,
    
    -- Metadata
    now() as processing_timestamp,
    today() as created_at

FROM public.stixor_iar_distributed
WHERE data_date = toDate('{TXN_DATE}')
  AND ac_from != ''