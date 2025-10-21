# Quick Start Guide

This guide will help you get started with the JazzCash Fraud Detection system in 10 minutes.

## Prerequisites

- Python 3.8+
- PostgreSQL access credentials
- 16GB+ RAM
- Spark cluster or standalone mode

## Setup (5 minutes)

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Download JDBC Driver

```bash
mkdir -p utils
wget -P utils https://jdbc.postgresql.org/download/postgresql-42.7.1.jar
```

### 3. Configure Database

Edit `config/db_config.yaml` and update:

```yaml
postgresql:
  host: "YOUR_HOST"
  port: "5432"
  database: "db_fraud"
  user: "YOUR_USER"
  password: "YOUR_PASSWORD"
  jdbc_driver_path: "utils/postgresql-42.7.1.jar"  # Update path
```

### 4. Create Directories

```bash
mkdir -p data/{raw,processed,features,models}
mkdir -p plots/{eda,model}
mkdir -p logs
```

## Run Pipeline (5 minutes with sample data)

### Option 1: Full End-to-End Pipeline

```bash
# Run with sample data (4.6M rows)
python pipelines/main_pipeline.py --mode sample

# Expected output:
# - Cleaned data: data/processed/cleaned_dataset.parquet
# - Features: data/features/feature_dataset.parquet
# - Model: data/models/xgboost_model.pkl
# - Metrics: data/models/xgboost_metrics.csv
# - Plots: plots/model/*.png
```

### Option 2: Step-by-Step

```bash
# Step 1: Extract and clean data
python pipelines/data_pipeline.py --mode sample

# Step 2: Engineer features
python pipelines/feature_pipeline.py

# Step 3: Train model
python pipelines/model_pipeline.py --model xgboost
```

## View Results

### Check Metrics

```bash
cat data/models/xgboost_metrics.csv
```

### View Plots

Plots are saved in `plots/model/`:
- `xgboost_feature_importance.png` - Top features
- `xgboost_roc_curve.png` - ROC curve
- `xgboost_pr_curve.png` - Precision-Recall curve
- `xgboost_confusion_matrix.png` - Confusion matrix

### Interactive Analysis

```bash
jupyter lab notebooks/02_eda.ipynb
```

## Common Issues

### Issue: Out of Memory

**Solution**: Reduce executor memory or use smaller sample

```yaml
# config/spark_config.yaml
active_profile: "development"  # Uses 4g memory
```

### Issue: Connection Timeout

**Solution**: Check database credentials and network

```bash
# Test connection
psql -h YOUR_HOST -U YOUR_USER -d db_fraud
```

### Issue: Import Errors

**Solution**: Reinstall dependencies

```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

1. **Explore Data**: Open `notebooks/02_eda.ipynb` for detailed analysis
2. **Tune Model**: Try different hyperparameters in `notebooks/06_model_training.ipynb`
3. **Run on Full Data**: Change mode to `july` or `full` for production
4. **Deploy Model**: Use saved model for real-time scoring

## Architecture Overview

```
Data Extraction → Data Cleaning → Feature Engineering → Model Training → Evaluation
    (5 min)          (3 min)           (7 min)            (2 min)        (instant)

PostgreSQL DB → Parquet Files → Feature Store → XGBoost Model → Performance Metrics
```

## Performance Expectations

| Mode | Data Size | Processing Time | Model Metrics |
|------|-----------|----------------|---------------|
| Sample | 4.6M rows | ~17 minutes | ROC-AUC: 0.85-0.90 |
| July | 412M rows | ~2.5 hours | ROC-AUC: 0.88-0.92 |
| Full | 2.1B rows | ~13 hours | ROC-AUC: 0.90-0.95 |

## Support

For detailed documentation, see [README.md](README.md)

For issues, check the [Troubleshooting](README.md#troubleshooting) section.

---

**Ready to detect fraud!** 🚀
