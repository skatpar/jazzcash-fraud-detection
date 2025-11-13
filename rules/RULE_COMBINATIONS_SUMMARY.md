# Fraud Detection Rule Combinations - Summary

## Overview
This document summarizes the rule combination analysis added to the fraud detection evaluation notebook.

## What Was Added

### 10 Strategic Rule Combinations

#### High Precision Combinations (AND Logic - Stricter)
These combinations require ALL conditions to be true, resulting in higher precision but potentially lower recall:

1. **COMBO 1: New Account + High Amount + Off-Peak**
   - New account (<30 days) AND amount >$10k AND off-peak hours (1-4 AM)
   - Target: Highly suspicious new account behavior

2. **COMBO 3: Multiple Channels + High Amount**
   - Multiple channels in 30 min AND amount >$5k
   - Target: Channel switching with high-value transactions

3. **COMBO 4: Very New Account + High Velocity**
   - Account <7 days AND >10 transactions in 1 hour
   - Target: Rapid exploitation of new accounts

4. **COMBO 6: Immediate Activity + High Amount + Weekend**
   - First transaction <1 hour after registration AND amount >$5k AND weekend
   - Target: Instant high-value fraud on weekends

5. **COMBO 8: PGW + New Account + High Amount + Unusual Hour**
   - Payment Gateway AND new account AND amount >$5k AND hours 2-4 AM
   - Target: High-risk PGW transactions

6. **COMBO 9: Dormant + High Amount + Round Number**
   - Dormant >30 days AND amount >$1k AND round number
   - Target: Account takeover patterns

7. **COMBO 10: New Recipient + Amount Jump**
   - First transaction to recipient AND 5x amount spike
   - Target: Behavioral anomalies

#### High Recall Combinations (OR Logic - Broader)
These combinations trigger if ANY condition is true, capturing more fraud but with lower precision:

8. **COMBO 2: High Velocity OR Amount Spike**
   - >10 transactions in 1 hour OR 3x max amount
   - Target: Broad velocity and amount anomalies

9. **COMBO 5: Round Number OR Structuring OR Off-Peak**
   - Round number OR just-below-threshold OR off-peak hours
   - Target: Multiple suspicious patterns

10. **COMBO 7: Money Mule OR High Amount to Recipient**
    - Recipient from >10 senders in 24h OR >$10k received in 1 hour
    - Target: Recipient-side fraud indicators

## Benchmark Analysis Features

### 1. **Comprehensive Comparison Table**
- Side-by-side comparison of all 24 individual rules vs 10 combinations
- Sorted by F1 score for easy identification of best performers
- Includes: precision, recall, F1 score, FPR, accuracy, and total flagged

### 2. **Performance Improvement Metrics**
- Average metrics for individual rules vs combinations
- Percentage improvement calculations
- Clear indication of which approach performs better

### 3. **Head-to-Head Comparison**
- Best individual rule vs best combination
- Metric-by-metric winner identification
- Detailed performance breakdown

### 4. **Precision-Recall Profile Analysis**
- Rules categorized into 4 profiles:
  - High Precision & High Recall
  - High Precision, Lower Recall
  - High Recall, Lower Precision
  - Low Precision & Low Recall
- Distribution analysis by rule type

### 5. **Top Performers Lists**
- Top 5 individual rules by F1 score
- Top 5 combinations by F1 score
- Best performers by precision
- Best performers by recall

### 6. **Strategic Insights**
- AND vs OR logic comparison
- Deployment recommendations
- Areas for further investigation
- Rules requiring refinement

## Output Files Generated

1. **fraud_rules_benchmark_comparison.csv**
   - Complete benchmark with all rules and combinations
   - All metrics and categorizations

2. **fraud_rule_combinations_results.csv**
   - Detailed results for all 10 combinations
   - Performance metrics and execution times

3. **fraud_rules_improvement_summary.csv**
   - Improvement analysis: combinations vs individual rules
   - Percentage gains/losses for each metric

4. **fraud_rules_top_performers.csv**
   - Top 5 individual rules + top 5 combinations
   - Easy reference for deployment decisions

## How to Use the Results

### For Production Deployment:

1. **Automatic Blocking (High Precision Required)**
   - Use combinations with precision ≥80% and FPR <5%
   - Recommended: High-precision AND combinations
   - Example: COMBO 1, COMBO 6, COMBO 8

2. **Manual Review Queue (Balanced Approach)**
   - Use combinations with F1 score ≥50
   - Balance precision and recall
   - Example: Best performing combinations overall

3. **Fraud Monitoring (High Recall Required)**
   - Use OR combinations to catch more potential fraud
   - Accept higher false positives for investigation
   - Example: COMBO 2, COMBO 5, COMBO 7

### Ensemble Strategy:
Consider implementing multiple rule tiers:
- **Tier 1 (Auto-Block)**: Precision ≥90%, FPR <2%
- **Tier 2 (High Priority Review)**: Precision ≥70%, FPR <5%
- **Tier 3 (Standard Review)**: Precision ≥50%, Recall ≥30%
- **Tier 4 (Monitoring)**: High recall rules for pattern detection

## Key Metrics Explained

- **Precision**: Of all flagged transactions, what % are actually fraud?
  - Higher = fewer false positives = less customer friction

- **Recall**: Of all fraud transactions, what % did we catch?
  - Higher = better fraud coverage = fewer missed frauds

- **F1 Score**: Harmonic mean of precision and recall
  - Balanced metric for overall performance

- **FPR (False Positive Rate)**: Of all legitimate transactions, what % did we incorrectly flag?
  - Lower = better customer experience

- **Accuracy**: Overall correctness
  - Can be misleading with imbalanced datasets

## Next Steps

1. **Run the notebook** to execute all combinations and see actual results
2. **Review the benchmark comparison** to identify top performers
3. **Select rule combinations** based on your business requirements:
   - Risk tolerance (FPR threshold)
   - Fraud coverage goals (recall target)
   - Operational capacity (number of flags to review)
4. **Validate on test data** before production deployment
5. **Monitor performance** and adjust thresholds as needed

## Notes

- All combinations use the same date range as individual rules (2025-07-01 to 2025-07-30)
- Combinations are executed as direct ClickHouse queries for efficiency
- Results may vary based on data distribution and fraud patterns in your specific timeframe
- Consider seasonal effects and evolving fraud patterns when deploying rules
