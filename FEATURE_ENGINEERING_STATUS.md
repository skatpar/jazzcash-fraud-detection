# Fraud Detection Feature Engineering - Complete Documentation

## 📁 Project Structure

```
jazzcash-fraud-detection/
├── sql/
│   └── feature_tables_ddl.sql          # All table definitions (CREATED ✅)
├── scripts/
│   ├── generate_user_features.py       # User-level features (CREATED ✅)
│   ├── generate_transaction_features.py # Transaction-level features (TODO)
│   ├── generate_combined_features.py   # Combined features (TODO)
│   └── run_feature_pipeline.py         # Master orchestration (TODO)
├── airflow/
│   └── dags/
│       └── fraud_detection_feature_generation.py
└── README.md
```

## ✅ Completed

### 1. SQL DDL File (`sql/feature_tables_ddl.sql`)

**Location:** `/root/research-dir/dev/jazzcash-fraud-detection/sql/feature_tables_ddl.sql`

**Contents:**
- **ac_from_features_local/distributed** - User-level features (38 features)
  - 3-day window: 9 features
  - 7-day window: 29 features
  
- **transaction_features_local/distributed** - Transaction-level features
  - Time-based features
  - Balance features
  - 3-day lookback features
  - Risk indicators
  - Channel/Type one-hot encoding
  
- **combined_features_local/distributed** - Combined features
  - Transaction features
  - User features  
  - MBAR account data (24 columns)
  - Fraud labels
  
- **Balanced datasets** (4 tables)
  - combined_features_balanced_1_1 (1:1 ratio)
  - combined_features_balanced_1_5 (1:5 ratio)
  - combined_features_balanced_1_10 (1:10 ratio)
  - combined_features_balanced_1_20 (1:20 ratio)

**Usage:**
```bash
clickhouse-client --multiquery < sql/feature_tables_ddl.sql
```

### 2. User Features Script (`scripts/generate_user_features.py`)

**Location:** `/root/research-dir/dev/jazzcash-fraud-detection/scripts/generate_user_features.py`

**Features Generated:**
- **3-day window (9 features):**
  - total_txns_3d, total_amount_3d, avg_amount_3d
  - median_amount_3d, max_amount_3d, min_amount_3d
  - unique_recipients_3d, unique_channels_3d, unique_types_3d

- **7-day window (29 features):**
  - Aggregates: total_txns_7d, total_amount_7d, avg_amount_7d, etc.
  - Channel features: most_used_channel_7d, channel_diversity_score_7d
  - Type features: most_used_type_7d, type_diversity_score_7d
  - Time-based: night_txns_7d, weekend_txns_7d, peak_hour_txns_7d
  - Balance: avg_start_balance_7d, balance_volatility_7d
  - Recipient: top_recipient_7d, recipient_concentration_ratio_7d
  - Behavioral: avg_time_between_txns_7d, txn_frequency_score_7d

**Usage:**
```bash
# Basic usage
python scripts/generate_user_features.py --cutoff-date 2025-07-01

# With options
python scripts/generate_user_features.py \
    --cutoff-date 2025-07-01 \
    --lookback-days 7 \
    --host localhost \
    --port 9000 \
    --database public \
    --verbose

# Dry run (preview query)
python scripts/generate_user_features.py --cutoff-date 2025-07-01 --dry-run
```

## 📋 TODO: Remaining Scripts

### 3. Transaction Features Script (TO BE CREATED)

**File:** `scripts/generate_transaction_features.py`

**Purpose:** Generate transaction-level features with historical lookback

**Features to Generate:**
- Time-based: hour_of_day, day_of_week, is_weekend, is_night, is_business_hours
- Balance: start_balance_log, balance_change, balance_change_pct
- 3-day lookback: txns_3d, total_amount_3d, avg_amount_3d, unique_recipients_3d
- Risk indicators: is_high_activity_3d, multi_channel_recent, amount_deviation_from_avg
- One-hot encoding: channel_* (5 columns), type_* (4 columns)

**Query Pattern:**
```sql
INSERT INTO transaction_features_distributed
SELECT 
    trans_id,
    ac_from,
    ac_to,
    data_date,
    trans_initiate_time,
    -- Time features
    toHour(trans_initiate_time) as hour_of_day,
    toDayOfWeek(trans_initiate_time) as day_of_week,
    -- Lookback features (self-join to get past 3 days)
    (SELECT count(*) FROM stixor_iar_distributed t2 
     WHERE t2.ac_from = t1.ac_from 
     AND t2.data_date BETWEEN t1.data_date - 3 AND t1.data_date - 1) as txns_3d,
    -- ... more features
FROM stixor_iar_distributed t1
WHERE data_date = toDate('{cutoff_date}')
```

### 4. Combined Features Script (TO BE CREATED)

**File:** `scripts/generate_combined_features.py`

**Purpose:** Join transaction + user + MBAR features + fraud labels

**Join Logic:**
```sql
INSERT INTO combined_features_distributed
SELECT 
    t.*,  -- All transaction features
    u.*,  -- All user features (prefixed with user_)
    m.*,  -- All MBAR columns (prefixed with mbar_)
    CASE 
        WHEN f.account_no IS NOT NULL THEN 1
        WHEN v.account_no IS NOT NULL THEN 1
        WHEN c.account_no IS NOT NULL THEN 1
        ELSE 0
    END AS fraud_flag
FROM transaction_features_distributed t
LEFT JOIN ac_from_features_distributed u 
    ON t.ac_from = u.ac_from 
    AND t.cutoff_date = u.cutoff_date
LEFT JOIN stixor_mbar_v m 
    ON t.ac_from = m.a_c_reference
LEFT JOIN fraud_accounts_with_types f 
    ON t.ac_from = f.account_no
LEFT JOIN victim_accounts_with_types v 
    ON t.ac_from = v.account_no
LEFT JOIN complaint_accounts_with_types c 
    ON t.ac_from = c.account_no
WHERE t.cutoff_date = toDate('{cutoff_date}')
```

### 5. Master Orchestration Script (TO BE CREATED)

**File:** `scripts/run_feature_pipeline.py`

**Purpose:** Run all feature generation steps in sequence

**Workflow:**
```python
def run_feature_pipeline(cutoff_date):
    """
    1. Create tables (if not exists)
    2. Generate user features
    3. Generate transaction features
    4. Generate combined features
    5. Create balanced datasets
    6. Verify all outputs
    """
    
    # Step 1: Ensure tables exist
    create_tables()
    
    # Step 2: Generate user features
    logger.info("Generating user features...")
    generate_user_features(cutoff_date)
    
    # Step 3: Generate transaction features
    logger.info("Generating transaction features...")
    generate_transaction_features(cutoff_date)
    
    # Step 4: Combine features
    logger.info("Generating combined features...")
    generate_combined_features(cutoff_date)
    
    # Step 5: Create balanced datasets
    logger.info("Creating balanced datasets...")
    create_balanced_datasets(cutoff_date)
    
    # Step 6: Verify
    logger.info("Verifying results...")
    verify_all_tables(cutoff_date)
```

**Usage:**
```bash
# Run complete pipeline
python scripts/run_feature_pipeline.py --cutoff-date 2025-07-01

# Run specific steps
python scripts/run_feature_pipeline.py --cutoff-date 2025-07-01 --steps user,transaction,combined

# Dry run
python scripts/run_feature_pipeline.py --cutoff-date 2025-07-01 --dry-run
```

## 🔄 Feature Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SOURCE DATA                                              │
│    - stixor_iar_distributed (transactions)                  │
│    - stixor_mbar_v (account MBAR data)                      │
│    - fraud/victim/complaint_accounts_with_types (labels)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. USER FEATURES (generate_user_features.py) ✅            │
│    - Input: stixor_iar_distributed                          │
│    - Output: ac_from_features_distributed                   │
│    - Features: 38 (3d + 7d aggregates)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TRANSACTION FEATURES (generate_transaction_features.py)  │
│    - Input: stixor_iar_distributed                          │
│    - Output: transaction_features_distributed               │
│    - Features: time, balance, lookback, risk indicators     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. COMBINED FEATURES (generate_combined_features.py)        │
│    - Join: transaction + user + MBAR + labels               │
│    - Output: combined_features_distributed                  │
│    - Features: 102 total columns                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. BALANCED DATASETS (create_balanced_datasets.py)          │
│    - Input: combined_features_distributed                   │
│    - Output: 4 balanced tables (1:1, 1:5, 1:10, 1:20)       │
│    - Method: Random undersampling                           │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Table Summary

| Table Name | Granularity | Row Count (example) | Features | Purpose |
|------------|-------------|---------------------|----------|---------|
| `ac_from_features_distributed` | User per cutoff date | 150K | 38 | User behavior aggregates |
| `transaction_features_distributed` | Per transaction | 10M | ~40 | Transaction-level features |
| `combined_features_distributed` | Per transaction | 10M | 102 | Complete feature set |
| `combined_features_balanced_1_1` | Per transaction | 278 | 102 | Training data (1:1) |
| `combined_features_balanced_1_5` | Per transaction | 834 | 102 | Training data (1:5) |

## 🚀 Quick Start Guide

### Step 1: Create Tables
```bash
cd /root/research-dir/dev/jazzcash-fraud-detection
clickhouse-client --multiquery < sql/feature_tables_ddl.sql
```

### Step 2: Generate User Features
```bash
python scripts/generate_user_features.py --cutoff-date 2025-07-01 --verbose
```

### Step 3: Generate Transaction Features (TODO - use notebook for now)
```python
# Run cells in feature_creation_clickhouse.ipynb
# Section: "Transaction-Level Feature Engineering"
```

### Step 4: Generate Combined Features (TODO - use notebook for now)
```python
# Run cells in feature_creation_clickhouse.ipynb  
# Section: "Combined Features"
```

### Step 5: Create Balanced Datasets (TODO - use notebook for now)
```python
# Run cells in feature_creation_clickhouse.ipynb
# Section: "Class Imbalance Handling"
```

## 📝 Feature Descriptions

### User-Level Features (38 total)

#### 3-Day Window (9 features)
- `total_txns_3d` - Total transaction count
- `total_amount_3d` - Sum of transaction amounts
- `avg_amount_3d` - Average transaction amount
- `median_amount_3d` - Median transaction amount
- `max_amount_3d` - Maximum transaction amount
- `min_amount_3d` - Minimum transaction amount
- `unique_recipients_3d` - Count of distinct recipients
- `unique_channels_3d` - Count of distinct channels used
- `unique_types_3d` - Count of distinct transaction types

#### 7-Day Window (29 features)
**Aggregates (9):**
- `total_txns_7d`, `total_amount_7d`, `avg_amount_7d`
- `median_amount_7d`, `max_amount_7d`, `min_amount_7d`
- `unique_recipients_7d`, `unique_channels_7d`, `unique_types_7d`

**Channel Features (3):**
- `most_used_channel_7d` - Most frequent channel
- `last_used_channel` - Channel of most recent transaction
- `channel_diversity_score_7d` - Diversity metric (1 - max/total)

**Type Features (3):**
- `most_used_type_7d` - Most frequent transaction type
- `last_used_type` - Type of most recent transaction
- `type_diversity_score_7d` - Diversity metric

**Time-Based (4):**
- `night_txns_7d` - Transactions during 2-6 AM
- `weekend_txns_7d` - Weekend transactions
- `peak_hour_txns_7d` - Transactions during 9 AM - 5 PM
- `off_peak_hour_txns_7d` - Transactions outside peak hours

**Balance (5):**
- `avg_start_balance_7d` - Average starting balance
- `avg_end_balance_7d` - Average ending balance
- `min_balance_7d` - Minimum balance
- `max_balance_7d` - Maximum balance
- `balance_volatility_7d` - Standard deviation

**Recipient (4):**
- `top_recipient_7d` - Most frequent recipient
- `avg_amount_per_recipient_7d` - Average per recipient
- `max_amount_to_single_recipient_7d` - Max to any recipient
- `recipient_concentration_ratio_7d` - Concentration metric

**Behavioral (5):**
- `avg_time_between_txns_7d` - Average hours between transactions
- `txn_frequency_score_7d` - Average daily frequency
- `first_txn_time` - First transaction timestamp
- `last_txn_time` - Last transaction timestamp
- `days_since_last_txn` - Days since last transaction

### Transaction-Level Features (~40 total)

#### Time-Based (6)
- `hour_of_day`, `day_of_week`, `is_weekend`
- `is_night`, `is_business_hours`, `is_unusual_hour`

#### Balance (3)
- `start_balance_log`, `balance_change`, `balance_change_pct`

#### 3-Day Lookback (10)
- `txns_3d`, `total_amount_3d`, `avg_amount_3d`
- `max_amount_3d`, `min_amount_3d`
- `unique_recipients_3d`, `unique_channels_3d`, `unique_types_3d`
- `night_txns_3d`, `weekend_txns_3d`

#### Risk Indicators (4)
- `is_high_activity_3d` - >10 transactions in 3d
- `multi_channel_recent` - Multiple channels in 3d
- `amount_deviation_from_avg` - Deviation from 3d average
- `night_weekend_combo` - Both night and weekend

#### Channel One-Hot (5)
- `channel_new_jc_app`, `channel_ussd`, `channel_ussd_api`
- `channel_payment_gateway`, `channel_mobile_app`

#### Type One-Hot (4)
- `type_transfer_c2c`, `type_transfer_c2b`
- `type_bill_payment`, `type_mobile_load`

### MBAR Features (24 columns)
Account information from MBAR system:
- Account details: reference, status, level, type
- Demographics: region, city, year of birth, place of birth
- Limits: credit/debit limits
- Dates: registration, dormant, reactivation
- Other: filer status, trust level, MPIN status

### Target Variable
- `fraud_flag` - 1 if fraud, 0 if legitimate

## 🎯 Next Steps

1. **Complete Transaction Features Script**
   - Create `generate_transaction_features.py`
   - Implement transaction-level feature generation
   - Add lookback logic for 3-day features

2. **Complete Combined Features Script**
   - Create `generate_combined_features.py`
   - Implement join logic for all feature sources
   - Add fraud label assignment

3. **Create Master Pipeline Script**
   - Create `run_feature_pipeline.py`
   - Orchestrate all steps in sequence
   - Add error handling and logging

4. **Test End-to-End**
   - Run complete pipeline for test date
   - Verify all outputs
   - Check feature quality

5. **Integrate with Airflow**
   - Update DAG to use new scripts
   - Test automated daily execution
   - Monitor performance

## 📚 References

- Feature Creation Notebook: `scripts/feature_creation_clickhouse.ipynb`
- DDL File: `sql/feature_tables_ddl.sql`
- User Features Script: `scripts/generate_user_features.py`
- Airflow DAG: `airflow/dags/fraud_detection_feature_generation.py`
