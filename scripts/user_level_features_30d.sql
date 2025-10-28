-- Corrected INSERT query without risk features
INSERT INTO public.ac_from_features_distributed
WITH 
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
        WHERE data_date > toDate('2025-05-31') - INTERVAL 30 DAY
          AND data_date <= toDate('2025-05-31')
          AND ac_from != ''
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
        WHERE data_date > toDate('2025-05-31') - INTERVAL 30 DAY
          AND data_date <= toDate('2025-05-31')
          AND ac_from != ''
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
        WHERE data_date > toDate('2025-05-31') - INTERVAL 30 DAY
          AND data_date <= toDate('2025-05-31')
          AND ac_from != ''
          AND ac_to != ''
    )
    GROUP BY ac_from, ac_to
)

SELECT 
    main.ac_from,
    toDate('2025-05-31') as cutoff_date,
    
    -- 1-day features
    sumIf(1, main.days_back = 1) as total_txns_1d,
    sumIf(main.start_balance, main.days_back = 1) as total_amount_1d,
    avgIf(main.start_balance, main.days_back = 1) as avg_amount_1d,
    quantileIf(0.5)(main.start_balance, main.days_back = 1) as median_amount_1d,
    maxIf(main.start_balance, main.days_back = 1) as max_amount_1d,
    minIf(main.start_balance, main.days_back = 1) as min_amount_1d,
    uniqIf(main.ac_to, main.days_back = 1) as unique_recipients_1d,
    uniqIf(main.trx_channel, main.days_back = 1) as unique_channels_1d,
    uniqIf(main.trx_type, main.days_back = 1) as unique_types_1d,
    
    -- 3-day features
    sumIf(1, main.days_back <= 3) as total_txns_3d,
    sumIf(main.start_balance, main.days_back <= 3) as total_amount_3d,
    avgIf(main.start_balance, main.days_back <= 3) as avg_amount_3d,
    quantileIf(0.5)(main.start_balance, main.days_back <= 3) as median_amount_3d,
    maxIf(main.start_balance, main.days_back <= 3) as max_amount_3d,
    minIf(main.start_balance, main.days_back <= 3) as min_amount_3d,
    uniqIf(main.ac_to, main.days_back <= 3) as unique_recipients_3d,
    uniqIf(main.trx_channel, main.days_back <= 3) as unique_channels_3d,
    uniqIf(main.trx_type, main.days_back <= 3) as unique_types_3d,
    
    -- 5-day features
    sumIf(1, main.days_back <= 5) as total_txns_5d,
    sumIf(main.start_balance, main.days_back <= 5) as total_amount_5d,
    avgIf(main.start_balance, main.days_back <= 5) as avg_amount_5d,
    quantileIf(0.5)(main.start_balance, main.days_back <= 5) as median_amount_5d,
    maxIf(main.start_balance, main.days_back <= 5) as max_amount_5d,
    minIf(main.start_balance, main.days_back <= 5) as min_amount_5d,
    uniqIf(main.ac_to, main.days_back <= 5) as unique_recipients_5d,
    uniqIf(main.trx_channel, main.days_back <= 5) as unique_channels_5d,
    uniqIf(main.trx_type, main.days_back <= 5) as unique_types_5d,
    
    -- 10-day features
    sumIf(1, main.days_back <= 10) as total_txns_10d,
    sumIf(main.start_balance, main.days_back <= 10) as total_amount_10d,
    avgIf(main.start_balance, main.days_back <= 10) as avg_amount_10d,
    quantileIf(0.5)(main.start_balance, main.days_back <= 10) as median_amount_10d,
    maxIf(main.start_balance, main.days_back <= 10) as max_amount_10d,
    minIf(main.start_balance, main.days_back <= 10) as min_amount_10d,
    uniqIf(main.ac_to, main.days_back <= 10) as unique_recipients_10d,
    uniqIf(main.trx_channel, main.days_back <= 10) as unique_channels_10d,
    uniqIf(main.trx_type, main.days_back <= 10) as unique_types_10d,
    
    -- 20-day features
    sumIf(1, main.days_back <= 20) as total_txns_20d,
    sumIf(main.start_balance, main.days_back <= 20) as total_amount_20d,
    avgIf(main.start_balance, main.days_back <= 20) as avg_amount_20d,
    quantileIf(0.5)(main.start_balance, main.days_back <= 20) as median_amount_20d,
    maxIf(main.start_balance, main.days_back <= 20) as max_amount_20d,
    minIf(main.start_balance, main.days_back <= 20) as min_amount_20d,
    uniqIf(main.ac_to, main.days_back <= 20) as unique_recipients_20d,
    uniqIf(main.trx_channel, main.days_back <= 20) as unique_channels_20d,
    uniqIf(main.trx_type, main.days_back <= 20) as unique_types_20d,
    
    -- 30-day features
    sumIf(1, main.days_back <= 30) as total_txns_30d,
    sumIf(main.start_balance, main.days_back <= 30) as total_amount_30d,
    avgIf(main.start_balance, main.days_back <= 30) as avg_amount_30d,
    quantileIf(0.5)(main.start_balance, main.days_back <= 30) as median_amount_30d,
    maxIf(main.start_balance, main.days_back <= 30) as max_amount_30d,
    minIf(main.start_balance, main.days_back <= 30) as min_amount_30d,
    uniqIf(main.ac_to, main.days_back <= 30) as unique_recipients_30d,
    uniqIf(main.trx_channel, main.days_back <= 30) as unique_channels_30d,
    uniqIf(main.trx_type, main.days_back <= 30) as unique_types_30d,
    
    -- Channel features (using pre-calculated stats)
    anyIf(ch.trx_channel, ch.channel_rank = 1) as most_used_channel_30d,
    argMax(main.trx_channel, main.trans_initiate_time) as last_used_channel,
    if(uniq(main.trx_channel) > 1, 
       1 - (max(ch.channel_count) / sum(ch.channel_count)), 0) as channel_diversity_score_30d,
    
    -- Type features (using pre-calculated stats)
    anyIf(ty.trx_type, ty.type_rank = 1) as most_used_type_30d,
    argMax(main.trx_type, main.trans_initiate_time) as last_used_type,
    if(uniq(main.trx_type) > 1, 
       1 - (max(ty.type_count) / sum(ty.type_count)), 0) as type_diversity_score_30d,
    
    -- Time-based features (30-day)
    sumIf(1, toHour(main.trans_initiate_time) IN (2,3,4,5,6) AND main.days_back <= 30) as night_txns_30d,
    sumIf(1, toDayOfWeek(main.trans_initiate_time) IN (6,7) AND main.days_back <= 30) as weekend_txns_30d,
    sumIf(1, toHour(main.trans_initiate_time) BETWEEN 9 AND 17 AND main.days_back <= 30) as peak_hour_txns_30d,
    sumIf(1, toHour(main.trans_initiate_time) NOT BETWEEN 9 AND 17 AND main.days_back <= 30) as off_peak_hour_txns_30d,
    
    -- Balance features (30-day)
    avgIf(main.start_balance, main.days_back <= 30) as avg_start_balance_30d,
    avgIf(main.end_balance, main.days_back <= 30) as avg_end_balance_30d,
    minIf(least(main.start_balance, main.end_balance), main.days_back <= 30) as min_balance_30d,
    maxIf(greatest(main.start_balance, main.end_balance), main.days_back <= 30) as max_balance_30d,
    stddevPopIf(main.start_balance, main.days_back <= 30) as balance_volatility_30d,
    
    -- Recipient features (using pre-calculated stats)
    anyIf(rs.ac_to, rs.recipient_rank = 1) as top_recipient_30d,
    avgIf(main.start_balance, main.days_back <= 30 AND main.ac_to != '') as avg_amount_per_recipient_30d,
    maxIf(main.start_balance, main.days_back <= 30 AND main.ac_to != '') as max_amount_to_single_recipient_30d,
    if(count(main.ac_from) > 0, max(rs.recipient_count) / count(main.ac_from), 0) as recipient_concentration_ratio_30d,
    
    -- Behavioral features
    if(count(main.ac_from) > 1,
       dateDiff('hour', min(main.trans_initiate_time), max(main.trans_initiate_time)) / (count(main.ac_from) - 1),
       0) as avg_time_between_txns_30d,
    count(main.ac_from) / greatest(dateDiff('day', min(main.data_date), max(main.data_date)) + 1, 1) as txn_frequency_score_30d,
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
    WHERE data_date > toDate('2025-05-31') - INTERVAL 30 DAY
      AND data_date <= toDate('2025-05-31')
      AND ac_from != ''
    ORDER BY ac_from, trans_initiate_time
) main
LEFT JOIN user_channel_stats ch ON main.ac_from = ch.ac_from AND main.trx_channel = ch.trx_channel
LEFT JOIN user_type_stats ty ON main.ac_from = ty.ac_from AND main.trx_type = ty.trx_type  
LEFT JOIN user_recipient_stats rs ON main.ac_from = rs.ac_from AND main.ac_to = rs.ac_to
GROUP BY main.ac_from;