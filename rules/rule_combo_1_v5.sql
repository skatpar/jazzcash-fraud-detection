WITH eligible_accounts AS (
    SELECT 
        ac_from
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-06-01' AND '2025-06-30'
      AND mbar_account_type_name = 'Customer Account'
    GROUP BY ac_from
    HAVING max(trx_amt) < 1000
),
combo_calc AS (
    SELECT 
        trans_id,
        fraud_flag,
        ac_from,
        trx_amt,
        trans_initiate_time,
        CASE 
            WHEN (COUNT(*) OVER (
                    PARTITION BY ac_from 
                    ORDER BY toUnixTimestamp(trans_initiate_time) ASC 
                    RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
                ) > 5)
              AND (trx_amt > 3 * MAX(trx_amt) OVER (
                    PARTITION BY ac_from 
                    ORDER BY toUnixTimestamp(trans_initiate_time)
                    RANGE BETWEEN 2*2592000 PRECEDING AND 1 PRECEDING
                ))
            THEN 1 ELSE 0 
        END AS combo_flag
    FROM public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '2025-07-01' AND '2025-07-31'
      AND mbar_account_type_name = 'Customer Account'
      AND ac_from GLOBAL IN (SELECT ac_from FROM eligible_accounts)
)
SELECT 
    'COMBO 2: High Velocity OR Amount Spike (OR)' AS rule_name,
    countIf(combo_flag = 1 AND fraud_flag = 1) AS true_positives,
    countIf(combo_flag = 1 AND fraud_flag = 0) AS false_positives,
    countIf(combo_flag = 0 AND fraud_flag = 1) AS false_negatives,
    countIf(combo_flag = 0 AND fraud_flag = 0) AS true_negatives,
    round(countIf(combo_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(combo_flag = 1), 0), 4) AS precision,
    round(countIf(combo_flag = 1 AND fraud_flag = 1) * 100.0 / nullIf(countIf(fraud_flag = 1), 0), 4) AS recall,
    round(2.0 * countIf(combo_flag = 1 AND fraud_flag = 1) / nullIf(countIf(combo_flag = 1) + countIf(fraud_flag = 1), 0) * 100, 4) AS f1_score,
    countIf(combo_flag = 1) AS total_flagged
FROM combo_calc;