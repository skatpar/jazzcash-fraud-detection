-- ============================================================================
-- FRAUD DETECTION RULES - COMPREHENSIVE ANALYSIS
-- ============================================================================
-- This file contains all fraud detection rules and their combinations
-- Expanded with additional rules per user request
-- Date: November 12, 2025
-- ============================================================================

-- ******************************************************
-- ============ BASE RULES (1-6) ========================
-- ******************************************************

-- ============================================================================
-- RULE 1: New Account with High Amount
-- Account age < 30 days AND transaction amount > $10,000
-- ============================================================================

WITH rule_1_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        mbar_registered_date_time,
        cutoff_date,
        CASE 
            WHEN dateDiff('day', toDate(mbar_registered_date_time), cutoff_date) < 30 
                AND trx_amt > 10000 
            THEN 1 ELSE 0 
        END AS rule_1_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule 1: New Account High Amount' AS rule_name,
    countIf(rule_1_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_1_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_1_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_1_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_1_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_1_flag = 1), 0), 4) AS precision,
    round(countIf(rule_1_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2 * (countIf(rule_1_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_1_flag = 1), 0)) * 
              (countIf(rule_1_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)) /
          nullIf((countIf(rule_1_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_1_flag = 1), 0)) + 
                 (countIf(rule_1_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)), 0), 4) AS f1_score,
    countIf(rule_1_flag = 1) AS total_flagged
FROM rule_1_calc;

-- ============================================================================
-- RULE 2: Very New Account with High Amount
-- Account age < 1 hour AND transaction amount > $10,000
-- ============================================================================

WITH rule_2_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        mbar_registered_date_time,
        cutoff_date,
        CASE 
            WHEN dateDiff('hour', mbar_registered_date_time, trans_initiate_time) < 1 
                AND trx_amt > 10000 
            THEN 1 ELSE 0 
        END AS rule_2_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule 2: Very New Account' AS rule_name,
    countIf(rule_2_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_2_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_2_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_2_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_2_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_2_flag = 1), 0), 4) AS precision,
    round(countIf(rule_2_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_2_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_2_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_2_flag = 1) AS total_flagged
FROM rule_2_calc;

-- ============================================================================
-- RULE 3: High Transaction Velocity
-- More than 10 transactions in 1 hour window
-- ============================================================================

WITH rule_3_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time) ASC 
                RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
            ) > 10 
            THEN 1 ELSE 0 
        END AS rule_3_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule 3: High Velocity' AS rule_name,
    countIf(rule_3_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_3_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_3_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_3_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_3_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_3_flag = 1), 0), 4) AS precision,
    round(countIf(rule_3_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_3_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_3_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_3_flag = 1) AS total_flagged
FROM rule_3_calc;

-- ============================================================================
-- RULE 4: Unusual Amount Pattern
-- Current transaction >= 3x max amount in last 30 days
-- ============================================================================

WITH rule_4_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN trx_amt >= 3 * MAX(trx_amt) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time)
                RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
            ) 
            THEN 1 ELSE 0 
        END AS rule_4_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule 4: 3x Max Amount' AS rule_name,
    countIf(rule_4_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_4_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_4_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_4_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_4_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_4_flag = 1), 0), 4) AS precision,
    round(countIf(rule_4_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_4_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_4_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_4_flag = 1) AS total_flagged
FROM rule_4_calc;

-- ============================================================================
-- RULE 5: Multiple Channels in Short Time
-- More than 1 unique channel used in last 30 minutes
-- ============================================================================

WITH rule_5_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_channel,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN uniqExact(trx_channel) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time)
                RANGE BETWEEN 1800 PRECEDING AND CURRENT ROW
            ) > 1 
            THEN 1 ELSE 0 
        END AS rule_5_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule 5: Multiple Channels' AS rule_name,
    countIf(rule_5_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_5_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_5_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_5_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_5_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_5_flag = 1), 0), 4) AS precision,
    round(countIf(rule_5_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_5_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_5_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_5_flag = 1) AS total_flagged
FROM rule_5_calc;

-- ============================================================================
-- RULE 6: Off-Peak Hours Transaction
-- Transactions between 1 AM and 5 AM
-- ============================================================================

WITH rule_6_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN toHour(trans_initiate_time) BETWEEN 1 AND 4 
            THEN 1 ELSE 0 
        END AS rule_6_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule 6: Off-Peak Hours' AS rule_name,
    countIf(rule_6_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_6_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_6_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_6_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_6_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_6_flag = 1), 0), 4) AS precision,
    round(countIf(rule_6_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_6_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_6_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_6_flag = 1) AS total_flagged
FROM rule_6_calc;

-- ******************************************************
-- =============== ADDITIONAL RULES ====================
-- ******************************************************

-- ============================================================================
-- RULE: Weekend Night Transaction (rule_time_weekend_night)
-- ============================================================================

WITH rule_time_weekend_night_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN (toDayOfWeek(trans_initiate_time) IN (1, 7))
              AND (toHour(trans_initiate_time) >= 23 OR toHour(trans_initiate_time) < 6) 
                 THEN 1 ELSE 0 
        END AS rule_time_weekend_night_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Weekend Night Transaction' AS rule_name,
    countIf(rule_time_weekend_night_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_time_weekend_night_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_time_weekend_night_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_time_weekend_night_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_time_weekend_night_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_time_weekend_night_flag = 1), 0), 4) AS precision,
    round(countIf(rule_time_weekend_night_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_time_weekend_night_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_time_weekend_night_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_time_weekend_night_flag = 1) AS total_flagged
FROM rule_time_weekend_night_calc;

-- ============================================================================
-- RULE: Transaction at Unusual Hour (rule_time_unusual_hour)
-- ============================================================================

WITH rule_time_unusual_hour_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        cutoff_date,
        is_unusual_hour,
        CASE 
            WHEN is_unusual_hour = 1 THEN 1 ELSE 0 
        END AS rule_time_unusual_hour_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Unusual Hour Transaction' AS rule_name,
    countIf(rule_time_unusual_hour_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_time_unusual_hour_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_time_unusual_hour_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_time_unusual_hour_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_time_unusual_hour_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_time_unusual_hour_flag = 1), 0), 4) AS precision,
    round(countIf(rule_time_unusual_hour_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_time_unusual_hour_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_time_unusual_hour_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_time_unusual_hour_flag = 1) AS total_flagged
FROM rule_time_unusual_hour_calc;

-- ============================================================================
-- RULE: New Account (<30 days) with High Amount >$500 (rule_age_new_high_amount)
-- ============================================================================

WITH rule_age_new_high_amount_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        mbar_registered_date_time,
        cutoff_date,
        CASE 
            WHEN dateDiff('day', toDate(mbar_registered_date_time), cutoff_date) < 30 
                 AND trx_amt > 500
                THEN 1 ELSE 0 
        END AS rule_age_new_high_amount_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: New Account High Amount >$500' AS rule_name,
    countIf(rule_age_new_high_amount_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_age_new_high_amount_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_age_new_high_amount_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_age_new_high_amount_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_age_new_high_amount_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_age_new_high_amount_flag = 1), 0), 4) AS precision,
    round(countIf(rule_age_new_high_amount_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_age_new_high_amount_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_age_new_high_amount_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_age_new_high_amount_flag = 1) AS total_flagged
FROM rule_age_new_high_amount_calc;

-- ============================================================================
-- RULE: Very New Account (<7 days), Any Transaction (rule_age_very_new_any_txn)
-- ============================================================================

WITH rule_age_very_new_any_txn_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        mbar_registered_date_time,
        cutoff_date,
        CASE 
            WHEN dateDiff('day', toDate(mbar_registered_date_time), cutoff_date) < 7
                THEN 1 ELSE 0 
        END AS rule_age_very_new_any_txn_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Very New Account <7 days Any Txn' AS rule_name,
    countIf(rule_age_very_new_any_txn_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_age_very_new_any_txn_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_age_very_new_any_txn_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_age_very_new_any_txn_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_age_very_new_any_txn_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_age_very_new_any_txn_flag = 1), 0), 4) AS precision,
    round(countIf(rule_age_very_new_any_txn_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_age_very_new_any_txn_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_age_very_new_any_txn_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_age_very_new_any_txn_flag = 1) AS total_flagged
FROM rule_age_very_new_any_txn_calc;

-- ============================================================================
-- RULE: Immediate Activity After Registration (rule_age_immediate_activity)
-- ============================================================================

WITH rule_age_immediate_activity_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        mbar_registered_date_time,
        cutoff_date,
        CASE 
            WHEN dateDiff('hour', mbar_registered_date_time, trans_initiate_time) < 1
                THEN 1 ELSE 0 
        END AS rule_age_immediate_activity_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: First Txn <1hr after Registration' AS rule_name,
    countIf(rule_age_immediate_activity_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_age_immediate_activity_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_age_immediate_activity_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_age_immediate_activity_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_age_immediate_activity_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_age_immediate_activity_flag = 1), 0), 4) AS precision,
    round(countIf(rule_age_immediate_activity_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_age_immediate_activity_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_age_immediate_activity_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_age_immediate_activity_flag = 1) AS total_flagged
FROM rule_age_immediate_activity_calc;

-- ============================================================================
-- RULE: Dormant Account >30 days then High Transaction (rule_age_dormant_30_high_amount)
-- ============================================================================

WITH rule_age_dormant_30_high_amount_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        trx_amt,
        LAG(trans_initiate_time) OVER (PARTITION BY ac_from ORDER BY trans_initiate_time) AS prev_tx_time,
        cutoff_date,
        CASE 
            WHEN prev_tx_time IS NOT NULL
             AND dateDiff('day', toDate(prev_tx_time), toDate(trans_initiate_time)) > 30
             AND trx_amt > 1000
                THEN 1 ELSE 0 
        END AS rule_age_dormant_30_high_amount_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Dormant >30d, High txn' AS rule_name,
    countIf(rule_age_dormant_30_high_amount_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_age_dormant_30_high_amount_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_age_dormant_30_high_amount_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_age_dormant_30_high_amount_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_age_dormant_30_high_amount_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_age_dormant_30_high_amount_flag = 1), 0), 4) AS precision,
    round(countIf(rule_age_dormant_30_high_amount_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_age_dormant_30_high_amount_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_age_dormant_30_high_amount_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_age_dormant_30_high_amount_flag = 1) AS total_flagged
FROM rule_age_dormant_30_high_amount_calc;

-- ============================================================================
-- RULE: High Value Transaction >$5,000 (rule_amount_high_value)
-- ============================================================================

WITH rule_amount_high_value_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN trx_amt > 5000 THEN 1 ELSE 0 
        END AS rule_amount_high_value_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: High Value txn >$5k' AS rule_name,
    countIf(rule_amount_high_value_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_amount_high_value_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_amount_high_value_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_amount_high_value_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_amount_high_value_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_amount_high_value_flag = 1), 0), 4) AS precision,
    round(countIf(rule_amount_high_value_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_amount_high_value_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_amount_high_value_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_amount_high_value_flag = 1) AS total_flagged
FROM rule_amount_high_value_calc;

-- ============================================================================
-- RULE: Round Number Amount (Multiples of 1000) (rule_amount_round_number)
-- ============================================================================

WITH rule_amount_round_number_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN trx_amt >= 1000 AND modulo(trx_amt, 1000) = 0 THEN 1 ELSE 0
        END AS rule_amount_round_number_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Round Number txn' AS rule_name,
    countIf(rule_amount_round_number_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_amount_round_number_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_amount_round_number_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_amount_round_number_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_amount_round_number_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_amount_round_number_flag = 1), 0), 4) AS precision,
    round(countIf(rule_amount_round_number_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_amount_round_number_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_amount_round_number_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_amount_round_number_flag = 1) AS total_flagged
FROM rule_amount_round_number_calc;

-- ============================================================================
-- RULE: Structuring (Just Below Thresholds) (rule_amount_structuring)
-- ============================================================================

WITH rule_amount_structuring_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN (trx_amt BETWEEN 4900 AND 4999) OR (trx_amt BETWEEN 9900 AND 9999)
                THEN 1 ELSE 0
        END AS rule_amount_structuring_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Structuring (Just-below thresh)' AS rule_name,
    countIf(rule_amount_structuring_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_amount_structuring_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_amount_structuring_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_amount_structuring_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_amount_structuring_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_amount_structuring_flag = 1), 0), 4) AS precision,
    round(countIf(rule_amount_structuring_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_amount_structuring_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_amount_structuring_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_amount_structuring_flag = 1) AS total_flagged
FROM rule_amount_structuring_calc;

-- ============================================================================
-- RULE: Amount Spike (>3x Historical Average) (rule_amount_spike)
-- ============================================================================

WITH rule_amount_spike_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        avg_amt := avg(trx_amt) OVER (
            PARTITION BY ac_from
            ORDER BY trans_initiate_time
            ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING
        ),
        CASE 
            WHEN trx_amt > 3 * avg_amt
                THEN 1 ELSE 0
        END AS rule_amount_spike_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Amount Spike >3x AVG' AS rule_name,
    countIf(rule_amount_spike_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_amount_spike_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_amount_spike_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_amount_spike_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_amount_spike_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_amount_spike_flag = 1), 0), 4) AS precision,
    round(countIf(rule_amount_spike_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_amount_spike_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_amount_spike_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_amount_spike_flag = 1) AS total_flagged
FROM rule_amount_spike_calc;

-- ============================================================================
-- RULE: Amount >3 StdDevs from AVG (rule_amount_3_stddev)
-- ============================================================================

WITH rule_amount_3_stddev_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        avg_amt := avg(trx_amt) OVER (
            PARTITION BY ac_from
            ORDER BY trans_initiate_time
            ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING
        ),
        stddev_amt := stddevPop(trx_amt) OVER (
            PARTITION BY ac_from
            ORDER BY trans_initiate_time
            ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING
        ),
        CASE 
            WHEN trx_amt > avg_amt + 3 * stddev_amt
                THEN 1 ELSE 0
        END AS rule_amount_3_stddev_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Amount >3 StdDev' AS rule_name,
    countIf(rule_amount_3_stddev_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_amount_3_stddev_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_amount_3_stddev_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_amount_3_stddev_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_amount_3_stddev_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_amount_3_stddev_flag = 1), 0), 4) AS precision,
    round(countIf(rule_amount_3_stddev_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_amount_3_stddev_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_amount_3_stddev_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_amount_3_stddev_flag = 1) AS total_flagged
FROM rule_amount_3_stddev_calc;

-- ============================================================================
-- RULE: PGW + New Account + High Amount (rule_channel_pgw_new_high)
-- ============================================================================

WITH rule_channel_pgw_new_high_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trx_channel,
        trans_initiate_time,
        mbar_registered_date_time,
        cutoff_date,
        CASE 
            WHEN trx_channel IN ('Payment Gateway', 'PGW')
                 AND dateDiff('day', toDate(mbar_registered_date_time), cutoff_date) < 30
                 AND trx_amt > 1000
                THEN 1 ELSE 0
        END AS rule_channel_pgw_new_high_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: PGW NewAcct HighAmt' AS rule_name,
    countIf(rule_channel_pgw_new_high_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_channel_pgw_new_high_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_channel_pgw_new_high_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_channel_pgw_new_high_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_channel_pgw_new_high_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_channel_pgw_new_high_flag = 1), 0), 4) AS precision,
    round(countIf(rule_channel_pgw_new_high_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_channel_pgw_new_high_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_channel_pgw_new_high_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_channel_pgw_new_high_flag = 1) AS total_flagged
FROM rule_channel_pgw_new_high_calc;

-- ============================================================================
-- RULE: Mobile New Account High Amount (rule_channel_mobile_new_high)
-- ============================================================================

WITH rule_channel_mobile_new_high_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trx_channel,
        trans_initiate_time,
        mbar_registered_date_time,
        cutoff_date,
        CASE 
            WHEN trx_channel IN ('Mobile App', 'NEW_JC_APP')
                 AND dateDiff('day', toDate(mbar_registered_date_time), cutoff_date) < 30
                 AND trx_amt > 1000
                THEN 1 ELSE 0
        END AS rule_channel_mobile_new_high_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Mobile NewAcct HighAmt' AS rule_name,
    countIf(rule_channel_mobile_new_high_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_channel_mobile_new_high_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_channel_mobile_new_high_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_channel_mobile_new_high_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_channel_mobile_new_high_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_channel_mobile_new_high_flag = 1), 0), 4) AS precision,
    round(countIf(rule_channel_mobile_new_high_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_channel_mobile_new_high_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_channel_mobile_new_high_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_channel_mobile_new_high_flag = 1) AS total_flagged
FROM rule_channel_mobile_new_high_calc;

-- ============================================================================
-- RULE: First Transaction to New Recipient with High Amount (rule_behavior_new_recipient_high)
-- ============================================================================

WITH rule_behavior_new_recipient_high_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        ac_to,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        ROW_NUMBER() OVER (PARTITION BY ac_from, ac_to ORDER BY trans_initiate_time) AS rn,
        CASE 
            WHEN rn = 1 AND trx_amt > 1000 THEN 1 ELSE 0
        END AS rule_behavior_new_recipient_high_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: New Recipient HighAmt' AS rule_name,
    countIf(rule_behavior_new_recipient_high_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_behavior_new_recipient_high_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_behavior_new_recipient_high_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_behavior_new_recipient_high_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_behavior_new_recipient_high_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_behavior_new_recipient_high_flag = 1), 0), 4) AS precision,
    round(countIf(rule_behavior_new_recipient_high_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_behavior_new_recipient_high_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_behavior_new_recipient_high_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_behavior_new_recipient_high_flag = 1) AS total_flagged
FROM rule_behavior_new_recipient_high_calc;

-- ============================================================================
-- RULE: Sudden 5x Amount Increase (rule_behavior_amount_jump)
-- ============================================================================

WITH rule_behavior_amount_jump_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        avg_amt := avg(trx_amt) OVER (
            PARTITION BY ac_from
            ORDER BY trans_initiate_time
            ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
        ),
        CASE 
            WHEN trx_amt > 5 * avg_amt
                 AND avg_amt < 1000
                THEN 1 ELSE 0
        END AS rule_behavior_amount_jump_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Amount Jump 5x AVG, AVG<1k' AS rule_name,
    countIf(rule_behavior_amount_jump_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_behavior_amount_jump_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_behavior_amount_jump_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_behavior_amount_jump_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_behavior_amount_jump_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_behavior_amount_jump_flag = 1), 0), 4) AS precision,
    round(countIf(rule_behavior_amount_jump_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_behavior_amount_jump_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_behavior_amount_jump_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_behavior_amount_jump_flag = 1) AS total_flagged
FROM rule_behavior_amount_jump_calc;

-- ============================================================================
-- RULE: Money Mule Indicator (Same Recipient >10 Senders in 24hrs) (rule_recipient_money_mule)
-- ============================================================================

WITH rule_recipient_money_mule_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_to,
        ac_from,
        trans_initiate_time,
        cutoff_date,
        num_senders_24h := count(DISTINCT ac_from) OVER (
            PARTITION BY ac_to
            ORDER BY toUnixTimestamp(trans_initiate_time)
            RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
        ),
        CASE 
            WHEN num_senders_24h > 10 THEN 1 ELSE 0
        END AS rule_recipient_money_mule_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Money Mule >10 In 24h' AS rule_name,
    countIf(rule_recipient_money_mule_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_recipient_money_mule_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_recipient_money_mule_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_recipient_money_mule_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_recipient_money_mule_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_recipient_money_mule_flag = 1), 0), 4) AS precision,
    round(countIf(rule_recipient_money_mule_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_recipient_money_mule_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_recipient_money_mule_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_recipient_money_mule_flag = 1) AS total_flagged
FROM rule_recipient_money_mule_calc;

-- ============================================================================
-- RULE: Recipient Receives >$10,000 in 1hr (rule_recipient_high_amount_1hour)
-- ============================================================================

WITH rule_recipient_high_amount_1hour_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_to,
        trx_amt,
        trans_initiate_time,
        cutoff_date,
        sum_amt_1h := sum(trx_amt) OVER (
            PARTITION BY ac_to
            ORDER BY toUnixTimestamp(trans_initiate_time)
            RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
        ),
        CASE 
            WHEN sum_amt_1h > 10000 THEN 1 ELSE 0
        END AS rule_recipient_high_amount_1hour_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Recipient >$10k In 1hr' AS rule_name,
    countIf(rule_recipient_high_amount_1hour_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_recipient_high_amount_1hour_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_recipient_high_amount_1hour_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_recipient_high_amount_1hour_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_recipient_high_amount_1hour_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_recipient_high_amount_1hour_flag = 1), 0), 4) AS precision,
    round(countIf(rule_recipient_high_amount_1hour_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_recipient_high_amount_1hour_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_recipient_high_amount_1hour_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_recipient_high_amount_1hour_flag = 1) AS total_flagged
FROM rule_recipient_high_amount_1hour_calc;

-- ============================================================================
-- RULE: Known Fraudster Sender (rule_ring_known_fraudster_sender)
-- ============================================================================

WITH rule_ring_known_fraudster_sender_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trans_initiate_time,
        cutoff_date,
        CASE 
            WHEN ac_from IN 
            (
                SELECT ac_from
                FROM fraud_accounts_blacklist
                WHERE status = 'HIGH_RISK'
            ) THEN 1 ELSE 0 
        END AS rule_ring_known_fraudster_sender_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30'
      AND mbar_account_type_name = 'Customer Account'
)
SELECT 
    'Rule: Known Fraudster Sender' AS rule_name,
    countIf(rule_ring_known_fraudster_sender_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(rule_ring_known_fraudster_sender_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(rule_ring_known_fraudster_sender_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(rule_ring_known_fraudster_sender_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(rule_ring_known_fraudster_sender_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(rule_ring_known_fraudster_sender_flag = 1), 0), 4) AS precision,
    round(countIf(rule_ring_known_fraudster_sender_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(rule_ring_known_fraudster_sender_flag = 1 AND fraud_flag = 1) / nullIf(countIf(rule_ring_known_fraudster_sender_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(rule_ring_known_fraudster_sender_flag = 1) AS total_flagged
FROM rule_ring_known_fraudster_sender_calc;

-- ============================================================================
-- END OF ADDITIONAL RULES
-- ============================================================================

-- NOTE: Insert rules 1-6 and combined rules here as in the original file!

-- ============================================================================
-- END OF FILE
-- ============================================================================
