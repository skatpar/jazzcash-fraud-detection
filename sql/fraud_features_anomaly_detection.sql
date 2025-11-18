-- ============================================================================
-- FRAUD FEATURES TABLE - ANOMALY & ERROR DETECTION QUERIES
-- ============================================================================
-- Purpose: Identify data quality issues, outliers, and erroneous values
-- Table: stixor_fraud_features_distributed
-- ============================================================================

-- ============================================================================
-- 1. NULL VALUES ANALYSIS
-- ============================================================================
-- Check for NULL values in critical columns
SELECT 
    'NULL Values Count' as check_type,
    countIf(trans_id IS NULL) as trans_id_nulls,
    countIf(cutoff_date IS NULL) as cutoff_date_nulls,
    countIf(fraud_flag IS NULL) as fraud_flag_nulls,
    countIf(trx_channel IS NULL) as trx_channel_nulls,
    countIf(trx_type IS NULL) as trx_type_nulls,
    countIf(start_balance IS NULL) as start_balance_nulls,
    countIf(end_balance IS NULL) as end_balance_nulls,
    countIf(trx_amt IS NULL) as trx_amt_nulls,
    countIf(mbar_registered_channel IS NULL) as mbar_registered_channel_nulls,
    countIf(distance_mean IS NULL) as distance_mean_nulls,
    countIf(distance_std IS NULL) as distance_std_nulls,
    countIf(hour_of_day IS NULL) as hour_of_day_nulls,
    countIf(day_of_week IS NULL) as day_of_week_nulls
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 2. INFINITE VALUES ANALYSIS
-- ============================================================================
-- Check for infinite values in numeric columns
SELECT 
    'Infinite Values Count' as check_type,
    countIf(isInfinite(start_balance)) as start_balance_inf,
    countIf(isInfinite(end_balance)) as end_balance_inf,
    countIf(isInfinite(trx_amt)) as trx_amt_inf,
    countIf(isInfinite(distance_mean)) as distance_mean_inf,
    countIf(isInfinite(distance_std)) as distance_std_inf,
    countIf(isInfinite(trx_count_1d)) as trx_count_1d_inf,
    countIf(isInfinite(trx_count_7d)) as trx_count_7d_inf,
    countIf(isInfinite(trx_count_30d)) as trx_count_30d_inf,
    countIf(isInfinite(trx_amt_sum_1d)) as trx_amt_sum_1d_inf,
    countIf(isInfinite(trx_amt_sum_7d)) as trx_amt_sum_7d_inf,
    countIf(isInfinite(trx_amt_sum_30d)) as trx_amt_sum_30d_inf
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 3. NaN VALUES ANALYSIS
-- ============================================================================
-- Check for NaN values in numeric columns
SELECT 
    'NaN Values Count' as check_type,
    countIf(isNaN(start_balance)) as start_balance_nan,
    countIf(isNaN(end_balance)) as end_balance_nan,
    countIf(isNaN(trx_amt)) as trx_amt_nan,
    countIf(isNaN(distance_mean)) as distance_mean_nan,
    countIf(isNaN(distance_std)) as distance_std_nan,
    countIf(isNaN(trx_count_1d)) as trx_count_1d_nan,
    countIf(isNaN(trx_count_7d)) as trx_count_7d_nan,
    countIf(isNaN(trx_count_30d)) as trx_count_30d_nan
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 4. NEGATIVE VALUES WHERE THEY SHOULDN'T BE
-- ============================================================================
-- Check for negative values in columns that should be positive
SELECT 
    'Negative Values Count' as check_type,
    countIf(trx_amt < 0) as negative_trx_amt,
    countIf(trx_count_1d < 0) as negative_trx_count_1d,
    countIf(trx_count_7d < 0) as negative_trx_count_7d,
    countIf(trx_count_30d < 0) as negative_trx_count_30d,
    countIf(trx_amt_sum_1d < 0) as negative_trx_amt_sum_1d,
    countIf(trx_amt_sum_7d < 0) as negative_trx_amt_sum_7d,
    countIf(trx_amt_sum_30d < 0) as negative_trx_amt_sum_30d,
    countIf(distance_mean < 0) as negative_distance_mean,
    countIf(distance_std < 0) as negative_distance_std
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 5. FRAUD FLAG VALIDATION
-- ============================================================================
-- Check for invalid fraud_flag values (should be 0 or 1 only)
SELECT 
    'Fraud Flag Validation' as check_type,
    count(*) as total_records,
    countIf(fraud_flag = 0) as fraud_flag_0,
    countIf(fraud_flag = 1) as fraud_flag_1,
    countIf(fraud_flag NOT IN (0, 1)) as invalid_fraud_flag,
    countIf(fraud_flag IS NULL) as null_fraud_flag
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 6. DATE RANGE VALIDATION
-- ============================================================================
-- Check for invalid or suspicious dates
SELECT 
    'Date Range Analysis' as check_type,
    min(cutoff_date) as earliest_date,
    max(cutoff_date) as latest_date,
    countIf(cutoff_date < '2020-01-01') as dates_before_2020,
    countIf(cutoff_date > today()) as dates_in_future,
    countIf(cutoff_date IS NULL) as null_dates
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 7. CATEGORICAL VALUES VALIDATION
-- ============================================================================
-- Check for empty strings or unusual values in categorical columns
SELECT 
    'Categorical Empty Values' as check_type,
    countIf(trx_channel = '') as empty_trx_channel,
    countIf(trx_type = '') as empty_trx_type,
    countIf(mbar_registered_channel = '') as empty_mbar_channel,
    countIf(mbar_account_type_name = '') as empty_account_type,
    countIf(length(trx_channel) = 0) as zero_length_channel,
    countIf(length(trx_type) = 0) as zero_length_type
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 8. HOUR OF DAY VALIDATION
-- ============================================================================
-- Check for invalid hour values (should be 0-23)
SELECT 
    'Hour of Day Validation' as check_type,
    count(*) as total_records,
    countIf(hour_of_day < 0) as negative_hours,
    countIf(hour_of_day > 23) as invalid_hours_gt_23,
    countIf(hour_of_day IS NULL) as null_hours,
    min(hour_of_day) as min_hour,
    max(hour_of_day) as max_hour
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 9. DAY OF WEEK VALIDATION
-- ============================================================================
-- Check for invalid day of week values (should be 0-6 or 1-7)
SELECT 
    'Day of Week Validation' as check_type,
    count(*) as total_records,
    countIf(day_of_week < 0) as negative_days,
    countIf(day_of_week > 7) as invalid_days_gt_7,
    countIf(day_of_week IS NULL) as null_days,
    min(day_of_week) as min_day,
    max(day_of_week) as max_day
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 10. BALANCE CONSISTENCY CHECK
-- ============================================================================
-- Check for suspicious balance patterns
SELECT 
    'Balance Anomalies' as check_type,
    countIf(start_balance < 0 AND end_balance > 0) as negative_start_positive_end,
    countIf(abs(end_balance - start_balance) > 10000000) as huge_balance_changes,
    countIf(start_balance = end_balance AND trx_amt != 0) as balance_mismatch,
    countIf(start_balance IS NULL AND end_balance IS NOT NULL) as start_null_end_not,
    countIf(start_balance IS NOT NULL AND end_balance IS NULL) as start_not_null_end_null
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 11. TRANSACTION COUNT CONSISTENCY
-- ============================================================================
-- Check for logical inconsistencies in transaction counts
SELECT 
    'Transaction Count Anomalies' as check_type,
    countIf(trx_count_1d > trx_count_7d) as count_1d_gt_7d,
    countIf(trx_count_7d > trx_count_30d) as count_7d_gt_30d,
    countIf(trx_count_1d > 1000) as extreme_count_1d,
    countIf(trx_count_7d > 5000) as extreme_count_7d,
    countIf(trx_count_30d > 20000) as extreme_count_30d
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 12. TRANSACTION AMOUNT CONSISTENCY
-- ============================================================================
-- Check for logical inconsistencies in transaction amounts
SELECT 
    'Transaction Amount Anomalies' as check_type,
    countIf(trx_amt_sum_1d > trx_amt_sum_7d) as amt_1d_gt_7d,
    countIf(trx_amt_sum_7d > trx_amt_sum_30d) as amt_7d_gt_30d,
    countIf(trx_amt > trx_amt_sum_1d) as current_gt_sum_1d,
    countIf(trx_amt > 10000000) as extreme_trx_amt,
    countIf(trx_amt = 0) as zero_trx_amt
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 13. DISTANCE STATISTICS VALIDATION
-- ============================================================================
-- Check for invalid distance statistics
SELECT 
    'Distance Anomalies' as check_type,
    countIf(distance_std < 0) as negative_std,
    countIf(distance_std > distance_mean AND distance_mean > 0) as std_gt_mean,
    countIf(distance_mean > 10000) as extreme_distance_mean,
    countIf(distance_std > 10000) as extreme_distance_std,
    countIf(distance_mean = 0 AND distance_std > 0) as zero_mean_positive_std
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 14. DUPLICATE TRANSACTION IDS
-- ============================================================================
-- Check for duplicate transaction IDs
SELECT 
    'Duplicate Transaction IDs' as check_type,
    count(*) as total_records,
    count(DISTINCT trans_id) as unique_trans_ids,
    count(*) - count(DISTINCT trans_id) as duplicate_count
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 15. OUTLIERS IN NUMERIC COLUMNS (using IQR method)
-- ============================================================================
-- Identify extreme outliers in key numeric columns
WITH stats AS (
    SELECT
        quantile(0.25)(trx_amt) as q1_trx_amt,
        quantile(0.75)(trx_amt) as q3_trx_amt,
        quantile(0.25)(start_balance) as q1_start_balance,
        quantile(0.75)(start_balance) as q3_start_balance,
        quantile(0.25)(trx_count_7d) as q1_trx_count_7d,
        quantile(0.75)(trx_count_7d) as q3_trx_count_7d
    FROM stixor_fraud_features_distributed
)
SELECT 
    'Outliers (3*IQR)' as check_type,
    countIf(trx_amt < q1_trx_amt - 3*(q3_trx_amt - q1_trx_amt) OR 
            trx_amt > q3_trx_amt + 3*(q3_trx_amt - q1_trx_amt)) as trx_amt_outliers,
    countIf(start_balance < q1_start_balance - 3*(q3_start_balance - q1_start_balance) OR 
            start_balance > q3_start_balance + 3*(q3_start_balance - q1_start_balance)) as start_balance_outliers,
    countIf(trx_count_7d < q1_trx_count_7d - 3*(q3_trx_count_7d - q1_trx_count_7d) OR 
            trx_count_7d > q3_trx_count_7d + 3*(q3_trx_count_7d - q1_trx_count_7d)) as trx_count_7d_outliers
FROM stixor_fraud_features_distributed, stats;

-- ============================================================================
-- 16. SUMMARY STATISTICS FOR KEY COLUMNS
-- ============================================================================
-- Get summary statistics to identify data distribution issues
SELECT 
    'Summary Statistics' as metric,
    count(*) as total_records,
    avg(trx_amt) as avg_trx_amt,
    stddevPop(trx_amt) as std_trx_amt,
    min(trx_amt) as min_trx_amt,
    max(trx_amt) as max_trx_amt,
    avg(start_balance) as avg_start_balance,
    stddevPop(start_balance) as std_start_balance,
    avg(trx_count_7d) as avg_trx_count_7d,
    max(trx_count_7d) as max_trx_count_7d
FROM stixor_fraud_features_distributed;

-- ============================================================================
-- 17. FIND SPECIFIC RECORDS WITH MULTIPLE ISSUES
-- ============================================================================
-- Get sample records that have multiple data quality issues
SELECT 
    trans_id,
    cutoff_date,
    fraud_flag,
    trx_amt,
    start_balance,
    end_balance,
    'Multiple Issues' as issue_type,
    multiIf(
        isInfinite(trx_amt), 'trx_amt is infinite',
        isNaN(trx_amt), 'trx_amt is NaN',
        trx_amt < 0, 'trx_amt is negative',
        isInfinite(start_balance), 'start_balance is infinite',
        isNaN(start_balance), 'start_balance is NaN',
        trx_count_1d > trx_count_7d, 'count_1d > count_7d',
        trx_amt_sum_1d > trx_amt_sum_7d, 'amt_sum_1d > amt_sum_7d',
        hour_of_day > 23 OR hour_of_day < 0, 'invalid hour_of_day',
        fraud_flag NOT IN (0, 1), 'invalid fraud_flag',
        'Unknown issue'
    ) as specific_issue
FROM stixor_fraud_features_distributed
WHERE 
    isInfinite(trx_amt) OR isNaN(trx_amt) OR trx_amt < 0
    OR isInfinite(start_balance) OR isNaN(start_balance)
    OR trx_count_1d > trx_count_7d
    OR trx_amt_sum_1d > trx_amt_sum_7d
    OR hour_of_day > 23 OR hour_of_day < 0
    OR fraud_flag NOT IN (0, 1)
LIMIT 100;

-- ============================================================================
-- 18. DATA COMPLETENESS BY DATE
-- ============================================================================
-- Check data quality trends over time
SELECT 
    cutoff_date,
    count(*) as record_count,
    countIf(fraud_flag = 1) as fraud_count,
    countIf(trx_amt IS NULL) as null_trx_amt,
    countIf(isInfinite(trx_amt) OR isNaN(trx_amt)) as invalid_trx_amt,
    countIf(start_balance IS NULL) as null_start_balance,
    countIf(trx_count_1d > trx_count_7d) as logical_errors
FROM stixor_fraud_features_distributed
WHERE cutoff_date >= '2025-06-01'
GROUP BY cutoff_date
ORDER BY cutoff_date DESC
LIMIT 100;

-- ============================================================================
-- 19. CATEGORICAL VALUE DISTRIBUTIONS
-- ============================================================================
-- Check for unusual or unexpected categorical values
SELECT 
    'trx_channel' as column_name,
    trx_channel as value,
    count(*) as record_count,
    countIf(fraud_flag = 1) as fraud_count,
    round(countIf(fraud_flag = 1) * 100.0 / count(*), 2) as fraud_rate_pct
FROM stixor_fraud_features_distributed
GROUP BY trx_channel
ORDER BY record_count DESC;

SELECT 
    'trx_type' as column_name,
    trx_type as value,
    count(*) as record_count,
    countIf(fraud_flag = 1) as fraud_count,
    round(countIf(fraud_flag = 1) * 100.0 / count(*), 2) as fraud_rate_pct
FROM stixor_fraud_features_distributed
GROUP BY trx_type
ORDER BY record_count DESC;

SELECT 
    'mbar_registered_channel' as column_name,
    mbar_registered_channel as value,
    count(*) as record_count,
    countIf(fraud_flag = 1) as fraud_count,
    round(countIf(fraud_flag = 1) * 100.0 / count(*), 2) as fraud_rate_pct
FROM stixor_fraud_features_distributed
GROUP BY mbar_registered_channel
ORDER BY record_count DESC;

-- ============================================================================
-- 20. COMPREHENSIVE DATA QUALITY SCORE
-- ============================================================================
-- Overall data quality assessment
SELECT 
    'Data Quality Summary' as report,
    count(*) as total_records,
    round(countIf(
        fraud_flag IN (0, 1) 
        AND trx_amt >= 0 
        AND NOT isInfinite(trx_amt) 
        AND NOT isNaN(trx_amt)
        AND hour_of_day BETWEEN 0 AND 23
        AND day_of_week BETWEEN 0 AND 7
        AND trx_count_1d <= trx_count_7d
        AND trx_count_7d <= trx_count_30d
        AND trx_channel != ''
        AND trx_type != ''
    ) * 100.0 / count(*), 2) as quality_score_pct,
    count(*) - countIf(
        fraud_flag IN (0, 1) 
        AND trx_amt >= 0 
        AND NOT isInfinite(trx_amt) 
        AND NOT isNaN(trx_amt)
        AND hour_of_day BETWEEN 0 AND 23
        AND day_of_week BETWEEN 0 AND 7
        AND trx_count_1d <= trx_count_7d
        AND trx_count_7d <= trx_count_30d
        AND trx_channel != ''
        AND trx_type != ''
    ) as problematic_records
FROM stixor_fraud_features_distributed;
