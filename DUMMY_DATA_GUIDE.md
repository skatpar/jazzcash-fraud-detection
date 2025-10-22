# Dummy Data Guide

This guide explains how to use the fraud detection system **without access to the PostgreSQL database** by generating realistic dummy data.

## Quick Start (2 Minutes)

### Linux/Mac

```bash
./run_with_dummy_data.sh
```

### Windows

```cmd
run_with_dummy_data.bat
```

This single command will:
1. Generate 100K user accounts
2. Generate 400K transactions (June-July 2025)
3. Generate ~640 fraud cases (0.16% fraud rate)
4. Run the complete fraud detection pipeline
5. Train XGBoost model
6. Generate evaluation plots

**Expected Runtime**: 20-25 minutes

## Manual Step-by-Step Process

If you prefer to run each step manually:

### Step 1: Generate Dummy Data

```bash
# Generate with default settings (100K users, 400K transactions)
python utils/generate_dummy_data.py

# Or customize
python utils/generate_dummy_data.py \
    --num-users 150000 \
    --num-transactions 600000 \
    --fraud-rate 0.002
```

**Parameters:**
- `--num-users`: Number of user accounts (default: 100,000)
- `--num-transactions`: Number of transactions (default: 400,000)
- `--fraud-rate`: Fraud rate as decimal (default: 0.0016 = 0.16%)
- `--output-dir`: Output directory (default: data/dummy)

**Output Files:**
- `data/dummy/mbar_dummy.parquet` - User accounts (MBAR table)
- `data/dummy/iar_dummy.parquet` - Transactions (IAR table)
- `data/dummy/fraud_dummy.parquet` - Fraud cases
- `data/dummy/*_sample.csv` - Sample CSV files for inspection
- `data/dummy/data_stats.txt` - Generation statistics

### Step 2: Prepare Master Dataset

```bash
python utils/prepare_dummy_dataset.py
```

This combines IAR and MBAR data (similar to the extraction pipeline) and adds fraud labels.

**Output:**
- `data/processed/master_dataset.parquet` - Combined dataset ready for pipeline
- `data/processed/dataset_stats.txt` - Dataset statistics

### Step 3: Run Pipeline

```bash
# Run feature engineering and model training (skip data extraction)
python pipelines/main_pipeline.py --skip-data --model xgboost
```

Or run individual stages:

```bash
# Clean data
python pipelines/data_pipeline.py --skip-extraction

# Engineer features
python pipelines/feature_pipeline.py

# Train model
python pipelines/model_pipeline.py --model xgboost
```

## Data Characteristics

### User Accounts (MBAR) - 100K by default

**Demographics:**
- Cities: 15 major Pakistani cities (Karachi, Lahore, Islamabad, etc.)
- Provinces: Punjab, Sindh, KPK, Balochistan
- Age range: 18-70 years
- Registration dates: 2023-01-01 to 2025-05-31

**Account Types (weighted distribution):**
- L0 - Unverified: 15%
- L1 - CNIC Verified: 30%
- L2 - Biometric: 25%
- L3 - Full KYC: 20%
- Agent Account: 5%
- Merchant Account: 5%

**Account Status:**
- Active: 80%
- Dormant: 10%
- Suspended: 5%
- Closed: 5%

**Trust Levels:**
- High: 30%
- Medium: 45%
- Low: 20%
- Unknown: 5%

### Transactions (IAR) - 400K by default

**Date Range:** June 1, 2025 to July 31, 2025

**Channels (weighted):**
- NEW_JC_APP: 50%
- PAYMENT GATEWAY: 25%
- THIRD_PARTY_WEB: 10%
- USSD: 10%
- ATM: 5%

**Transaction Types (weighted):**
- Transfer(C2C): 30%
- Transfer(C2B): 20%
- Online Payment: 20%
- IBFT Outgoing Customer: 10%
- Bill Payment: 8%
- Get Loan: 5%
- Cash Withdrawal: 4%
- Mobile Topup: 3%

**Amount Distribution:**
- Log-normal distribution (mean=7.0, sigma=1.5)
- Range: PKR 100 to PKR 100,000
- ~30% are round amounts (multiples of 100)

**Status:**
- Completed: 92%
- Failed: 6%
- Pending: 2%

### Fraud Cases - ~640 (0.16% fraud rate)

- Random transactions marked as fraud
- Complaint filed 1-72 hours after transaction
- Resolved 1-30 days after complaint
- Receiver account marked as fraudster
- Sender marked as victim

## Customization Examples

### Quick Test (Small Dataset)

```bash
# Generate 10K users, 50K transactions for quick testing
python utils/generate_dummy_data.py \
    --num-users 10000 \
    --num-transactions 50000 \
    --output-dir data/dummy_small
```

**Runtime**: ~2 minutes

### Large Dataset (Closer to Production)

```bash
# Generate 200K users, 1M transactions
python utils/generate_dummy_data.py \
    --num-users 200000 \
    --num-transactions 1000000 \
    --output-dir data/dummy_large
```

**Runtime**: ~10-15 minutes

### Higher Fraud Rate (For Testing)

```bash
# Generate with 1% fraud rate (easier to detect patterns)
python utils/generate_dummy_data.py \
    --fraud-rate 0.01
```

## Inspecting Generated Data

### View Sample Data

```bash
# View sample accounts
head -n 20 data/dummy/mbar_sample.csv

# View sample transactions
head -n 20 data/dummy/iar_sample.csv

# View sample fraud cases
head -n 20 data/dummy/fraud_sample.csv
```

### View Statistics

```bash
cat data/dummy/data_stats.txt
```

### Load in Python

```python
import pandas as pd

# Load full datasets
df_mbar = pd.read_parquet('data/dummy/mbar_dummy.parquet')
df_iar = pd.read_parquet('data/dummy/iar_dummy.parquet')
df_fraud = pd.read_parquet('data/dummy/fraud_dummy.parquet')

# Inspect
print(df_mbar.info())
print(df_iar.describe())
print(df_fraud.head())
```

### Explore in Jupyter

```bash
jupyter lab notebooks/02_eda.ipynb
```

Then modify the notebook to load dummy data:

```python
# Change data loading cell to:
df_spark = spark.read.parquet('data/processed/master_dataset.parquet')
```

## Expected Pipeline Performance

### With Default Dummy Data (100K users, 400K transactions)

| Stage | Runtime | Output |
|-------|---------|--------|
| Data Generation | ~3-5 min | 400K transactions with fraud labels |
| Data Preparation | ~30 sec | Master dataset (IAR + MBAR combined) |
| Feature Engineering | ~8-10 min | 200+ features per transaction |
| Model Training | ~2-3 min | Trained XGBoost model |
| **Total** | **~15-20 min** | Complete fraud detection model |

### Expected Model Metrics

| Metric | Expected Range |
|--------|----------------|
| ROC-AUC | 0.75 - 0.85 |
| PR-AUC | 0.20 - 0.35 |
| F1-Score | 0.15 - 0.25 |
| Top 5% Precision | 0.30 - 0.50 |

**Note**: Metrics may be lower than with real data since dummy data has purely random fraud labels (no real patterns).

## Comparison: Dummy vs Real Data

| Aspect | Dummy Data | Real Data |
|--------|------------|-----------|
| Setup Time | 5 minutes | 30+ minutes (DB setup) |
| Data Access | No database needed | Requires PostgreSQL access |
| Data Volume | Configurable (10K-1M+) | Fixed (4.6M - 2.1B) |
| Fraud Patterns | Random (no real patterns) | Real fraud behaviors |
| Model Performance | Lower (60-85% AUC) | Higher (85-95% AUC) |
| Use Case | Development, testing, demos | Production, research |

## Troubleshooting

### Issue: Out of Memory During Generation

**Solution**: Generate smaller dataset

```bash
python utils/generate_dummy_data.py --num-users 50000 --num-transactions 200000
```

### Issue: Generation Takes Too Long

**Solution**: Reduce dataset size or use multiprocessing

```bash
# Quick test dataset
python utils/generate_dummy_data.py --num-users 10000 --num-transactions 50000
```

### Issue: Pipeline Fails After Data Generation

**Solution**: Ensure data preparation step completed

```bash
# Re-run preparation
python utils/prepare_dummy_dataset.py

# Check if master dataset exists
ls -lh data/processed/master_dataset.parquet/
```

### Issue: Low Model Performance

**Solution**: This is expected with dummy data (random fraud labels). To improve:

1. Increase fraud rate for easier detection:
```bash
python utils/generate_dummy_data.py --fraud-rate 0.01
```

2. Generate more data:
```bash
python utils/generate_dummy_data.py --num-transactions 1000000
```

## Advanced Usage

### Generate Multiple Datasets

```bash
# Development dataset
python utils/generate_dummy_data.py \
    --num-users 10000 --num-transactions 50000 \
    --output-dir data/dummy_dev

# Test dataset
python utils/generate_dummy_data.py \
    --num-users 50000 --num-transactions 200000 \
    --output-dir data/dummy_test

# Production-like dataset
python utils/generate_dummy_data.py \
    --num-users 150000 --num-transactions 600000 \
    --output-dir data/dummy_prod
```

### Combine Dummy Data with Real Data

If you have partial real data:

```python
import pandas as pd

# Load dummy and real data
df_dummy = pd.read_parquet('data/dummy/iar_dummy.parquet')
df_real = pd.read_parquet('data/real/iar_real.parquet')

# Combine
df_combined = pd.concat([df_real, df_dummy], ignore_index=True)

# Save
df_combined.to_parquet('data/combined/iar_combined.parquet')
```

## Next Steps

After generating and testing with dummy data:

1. **Understand the Pipeline**: Explore notebooks to understand each step
2. **Tune Features**: Modify `config/feature_config.yaml` to experiment
3. **Try Different Models**: Test Random Forest vs XGBoost
4. **Prepare for Real Data**: When DB access is available, switch to real data
5. **Deploy**: Use trained model for predictions

## Benefits of Dummy Data

✅ **No Database Required**: Test entire pipeline without DB access
✅ **Fast Setup**: Start working in minutes
✅ **Reproducible**: Same data every time (with fixed seed)
✅ **Configurable**: Adjust size and characteristics
✅ **Educational**: Learn pipeline with clean, understandable data
✅ **Development**: Test code changes quickly
✅ **Demos**: Present system without exposing real data

## Limitations

⚠️ **No Real Patterns**: Fraud labels are random, not based on actual behavior
⚠️ **Lower Accuracy**: Model performance will be lower than with real data
⚠️ **Simplified Data**: Real-world complexity and edge cases not represented
⚠️ **No Time Patterns**: Transaction patterns are simplified

## Support

For issues with dummy data generation:
1. Check error messages in terminal
2. Verify output directory is writable
3. Ensure enough disk space (need ~1-2GB for 400K transactions)
4. Check Python version (requires 3.8+)

For questions, see main [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md).

---

**Happy Testing!** 🚀
