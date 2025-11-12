-- ============================================================================
-- FRAUD DETECTION RULES - COMPREHENSIVE ANALYSIS
-- ============================================================================
-- This file contains all fraud detection rules and their combinations
-- Date: November 12, 2025
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

-- ============================================================================
-- COMBINED RULES: Rules 1, 2, 3
-- ============================================================================

WITH rule_calculations AS (
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
        END AS rule_1,
        CASE 
            WHEN dateDiff('hour', mbar_registered_date_time, trans_initiate_time) < 1 
                AND trx_amt > 10000 
            THEN 1 ELSE 0 
        END AS rule_2,
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time) ASC 
                RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
            ) > 10 
            THEN 1 ELSE 0 
        END AS rule_3
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
),
combined_rules AS (
    SELECT 
        *,
        CASE 
            WHEN rule_1 = 1 OR rule_2 = 1 OR rule_3 = 1
            THEN 1 ELSE 0 
        END AS any_rule_triggered,
        CASE 
            WHEN rule_1 = 1 AND rule_2 = 1 AND rule_3 = 1
            THEN 1 ELSE 0 
        END AS all_rules_triggered
    FROM rule_calculations
)
SELECT 
    'Combined: ANY Rule (1-3)' AS strategy,
    countIf(any_rule_triggered = 1 AND fraud_flag = 1) AS true_positives,
    countIf(any_rule_triggered = 1 AND fraud_flag = 0) AS false_positives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0), 4) AS precision,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2 * (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) * 
              (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)) /
          nullIf((countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) + 
                 (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)), 0), 4) AS f1_score,
    countIf(any_rule_triggered = 1) AS total_flagged
FROM combined_rules;

-- ============================================================================
-- COMBINED RULES: Rules 1, 2, 3, 4
-- ============================================================================

WITH rule_calculations AS (
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
        END AS rule_1,
        CASE 
            WHEN dateDiff('hour', mbar_registered_date_time, trans_initiate_time) < 1 
                AND trx_amt > 10000 
            THEN 1 ELSE 0 
        END AS rule_2,
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time) ASC 
                RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
            ) > 10 
            THEN 1 ELSE 0 
        END AS rule_3,
        CASE 
            WHEN trx_amt >= 3 * MAX(trx_amt) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time)
                RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
            ) 
            THEN 1 ELSE 0 
        END AS rule_4
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
),
combined_rules AS (
    SELECT 
        *,
        CASE 
            WHEN rule_1 = 1 OR rule_2 = 1 OR rule_3 = 1 OR rule_4 = 1
            THEN 1 ELSE 0 
        END AS any_rule_triggered,
        CASE 
            WHEN rule_1 = 1 AND rule_2 = 1 AND rule_3 = 1 AND rule_4 = 1
            THEN 1 ELSE 0 
        END AS all_rules_triggered
    FROM rule_calculations
)
SELECT 
    'Combined: ANY Rule (1-4)' AS strategy,
    countIf(any_rule_triggered = 1 AND fraud_flag = 1) AS true_positives,
    countIf(any_rule_triggered = 1 AND fraud_flag = 0) AS false_positives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0), 4) AS precision,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2 * (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) * 
              (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)) /
          nullIf((countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) + 
                 (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)), 0), 4) AS f1_score,
    countIf(any_rule_triggered = 1) AS total_flagged
FROM combined_rules;

-- ============================================================================
-- COMBINED RULES: Rules 1, 4, 5
-- ============================================================================

WITH rule_calculations AS (
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
            WHEN dateDiff('day', toDate(mbar_registered_date_time), cutoff_date) < 30 
                AND trx_amt > 10000 
            THEN 1 ELSE 0 
        END AS rule_1,
        CASE 
            WHEN trx_amt >= 3 * MAX(trx_amt) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time)
                RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
            ) 
            THEN 1 ELSE 0 
        END AS rule_4,
        CASE 
            WHEN uniqExact(trx_channel) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time)
                RANGE BETWEEN 1800 PRECEDING AND CURRENT ROW
            ) > 1 
            THEN 1 ELSE 0 
        END AS rule_5
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
),
combined_rules AS (
    SELECT 
        *,
        CASE 
            WHEN rule_1 = 1 OR rule_4 = 1 OR rule_5 = 1
            THEN 1 ELSE 0 
        END AS any_rule_triggered,
        CASE 
            WHEN rule_1 = 1 AND rule_4 = 1 AND rule_5 = 1
            THEN 1 ELSE 0 
        END AS all_rules_triggered
    FROM rule_calculations
)
SELECT 
    'Combined: ANY Rule (1,4,5)' AS strategy,
    countIf(any_rule_triggered = 1 AND fraud_flag = 1) AS true_positives,
    countIf(any_rule_triggered = 1 AND fraud_flag = 0) AS false_positives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0), 4) AS precision,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2 * (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) * 
              (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)) /
          nullIf((countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) + 
                 (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)), 0), 4) AS f1_score,
    countIf(any_rule_triggered = 1) AS total_flagged
FROM combined_rules;

-- ============================================================================
-- COMBINED RULES: ALL 6 RULES
-- ============================================================================

WITH rule_calculations AS (
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
            WHEN dateDiff('day', toDate(mbar_registered_date_time), cutoff_date) < 30 
                AND trx_amt > 10000 
            THEN 1 ELSE 0 
        END AS rule_1,
        CASE 
            WHEN dateDiff('hour', mbar_registered_date_time, trans_initiate_time) < 1 
                AND trx_amt > 10000 
            THEN 1 ELSE 0 
        END AS rule_2,
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time) ASC 
                RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
            ) > 10 
            THEN 1 ELSE 0 
        END AS rule_3,
        CASE 
            WHEN trx_amt >= 3 * MAX(trx_amt) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time)
                RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
            ) 
            THEN 1 ELSE 0 
        END AS rule_4,
        CASE 
            WHEN uniqExact(trx_channel) OVER (
                PARTITION BY ac_from 
                ORDER BY toUnixTimestamp(trans_initiate_time)
                RANGE BETWEEN 1800 PRECEDING AND CURRENT ROW
            ) > 1 
            THEN 1 ELSE 0 
        END AS rule_5,
        CASE 
            WHEN toHour(trans_initiate_time) BETWEEN 1 AND 4 
            THEN 1 ELSE 0 
        END AS rule_6
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-30' 
        AND mbar_account_type_name = 'Customer Account'
),
combined_rules AS (
    SELECT 
        *,
        CASE 
            WHEN rule_1 = 1 OR rule_2 = 1 OR rule_3 = 1 
                OR rule_4 = 1 OR rule_5 = 1 OR rule_6 = 1 
            THEN 1 ELSE 0 
        END AS any_rule_triggered,
        CASE 
            WHEN rule_1 = 1 AND rule_2 = 1 AND rule_3 = 1 
                AND rule_4 = 1 AND rule_5 = 1 AND rule_6 = 1 
            THEN 1 ELSE 0 
        END AS all_rules_triggered,
        rule_1 + rule_2 + rule_3 + rule_4 + rule_5 + rule_6 AS rules_count
    FROM rule_calculations
)
SELECT 
    'Combined: ANY Rule (All 6)' AS strategy,
    countIf(any_rule_triggered = 1 AND fraud_flag = 1) AS true_positives,
    countIf(any_rule_triggered = 1 AND fraud_flag = 0) AS false_positives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(any_rule_triggered = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0), 4) AS precision,
    round(countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2 * (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) * 
              (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)) /
          nullIf((countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(any_rule_triggered = 1), 0)) + 
                 (countIf(any_rule_triggered = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0)), 0), 4) AS f1_score,
    countIf(any_rule_triggered = 1) AS total_flagged
FROM combined_rules;

-- ============================================================================
-- END OF FILE
-- ============================================================================