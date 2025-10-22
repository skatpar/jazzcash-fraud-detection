# JazzCash Fraud Detection System

A comprehensive, production-ready fraud detection system for JazzCash mobile wallet transactions using PySpark for scalable data processing and machine learning.

## Table of Contents

- [Overview](#overview)
- [Quick Start with Dummy Data](#quick-start-with-dummy-data) ⭐ **NEW!**
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Running Individual Pipelines](#running-individual-pipelines)
  - [Using Jupyter Notebooks](#using-jupyter-notebooks)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements an end-to-end fraud detection system for JazzCash transactions, processing billions of transaction records and millions of customer accounts to identify fraudulent activities.

### Key Capabilities

- **Scalable Data Processing**: Handles 2.1B+ transactions using PySpark
- **Comprehensive Feature Engineering**: 200+ features including temporal, behavioral, and network patterns
- **Advanced ML Models**: XGBoost and Random Forest with class imbalance handling
- **Production-Ready**: Modular design, extensive logging, and configuration management
- **Extensive EDA**: Jupyter notebooks for exploratory data analysis and visualization

### Dataset Overview

| Table | Rows | Size | Description |
|-------|------|------|-------------|
| stixor_iar (main) | 2.1B | 2TB+ | All transactions |
| stixor_iar_jul | 412M | 400GB | July 2025 transactions |
| stixor_iar_20250701_sample | 4.6M | 5GB | Sample for development |
| stixor_mbar_v | 85M | 100GB | Customer accounts |
| fraud | 40K | <1GB | Fraud cases |

## Quick Start with Dummy Data

**Don't have database access?** No problem! Generate realistic dummy data and run the complete pipeline in minutes.

### One-Command Quick Start

```bash
# Linux/Mac
./run_with_dummy_data.sh

# Windows
run_with_dummy_data.bat
```

This will:
1. Generate 100K users and 400K transactions
2. Create realistic fraud patterns
3. Run the complete fraud detection pipeline
4. Train and evaluate an XGBoost model

**Runtime**: ~20 minutes | **No database required!**

### Manual Dummy Data Generation

```bash
# Step 1: Generate dummy data
python utils/generate_dummy_data.py \
    --num-users 100000 \
    --num-transactions 400000

# Step 2: Prepare dataset
python utils/prepare_dummy_dataset.py

# Step 3: Run pipeline
python pipelines/main_pipeline.py --skip-data --model xgboost
```

### Customization

```bash
# Quick test (10K users, 50K transactions)
python utils/generate_dummy_data.py --num-users 10000 --num-transactions 50000

# Large dataset (200K users, 1M transactions)
python utils/generate_dummy_data.py --num-users 200000 --num-transactions 1000000

# Higher fraud rate for testing
python utils/generate_dummy_data.py --fraud-rate 0.01
```

**📖 Complete Guide**: See [DUMMY_DATA_GUIDE.md](DUMMY_DATA_GUIDE.md) for detailed instructions.

## Project Structure

```
jazzcash-fraud-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/                            # Configuration files
│   ├── db_config.yaml                # Database credentials & table configs
│   ├── spark_config.yaml             # Spark session configurations
│   └── feature_config.yaml           # Feature engineering settings
├── data/                              # Data storage (gitignored)
│   ├── raw/                          # Raw extracted data
│   ├── processed/                    # Cleaned datasets
│   ├── features/                     # Feature-engineered data
│   └── models/                       # Trained models
├── src/                               # Source code
│   ├── data/                         # Data extraction & cleaning
│   │   ├── extraction.py            # PostgreSQL data extraction
│   │   ├── cleaning.py              # Data cleaning & validation
│   │   └── validation.py            # Data quality checks
│   ├── features/                     # Feature engineering
│   │   ├── transaction_features.py  # Transaction-level features
│   │   ├── temporal_features.py     # Time-based features
│   │   ├── account_features.py      # Account profile features
│   │   └── selection.py             # Feature selection utilities
│   ├── models/                       # Model training & evaluation
│   │   ├── train.py                 # Model training
│   │   └── evaluate.py              # Model evaluation
│   ├── visualization/                # Plotting utilities
│   │   └── plots.py                 # EDA and model visualization
│   └── utils/                        # Utility functions
│       ├── spark_utils.py           # Spark helper functions
│       └── logger.py                # Logging utilities
├── pipelines/                         # Pipeline orchestrators
│   ├── data_pipeline.py             # Data extraction & cleaning pipeline
│   ├── feature_pipeline.py          # Feature engineering pipeline
│   ├── model_pipeline.py            # Model training pipeline
│   └── main_pipeline.py             # End-to-end orchestrator
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_extraction.ipynb     # Data extraction tutorial
│   ├── 02_eda.ipynb                 # Exploratory data analysis
│   ├── 03_data_cleaning.ipynb       # Data cleaning examples
│   ├── 04_feature_engineering.ipynb # Feature engineering guide
│   ├── 05_feature_selection.ipynb   # Feature selection analysis
│   ├── 06_model_training.ipynb      # Model training examples
│   └── 07_model_evaluation.ipynb    # Model evaluation & interpretation
├── tests/                             # Unit tests
└── logs/                              # Log files
```

## Features

### Data Processing
- Parallel data extraction from PostgreSQL using predicates
- Intelligent join strategies (broadcast for small tables)
- Outlier detection (IQR and Z-score methods)
- Missing value imputation strategies
- Data quality validation

### Feature Engineering

**Transaction Features (50+)**
- Basic: log amounts, ratios, round numbers
- Aggregations: counts, sums, averages over time windows (1d, 3d, 7d, 15d, 30d)
- Velocity: transaction rate changes, burst detection

**Temporal Features (10+)**
- Hour of day, day of week, weekend flags
- Business hours, odd hours, time buckets
- Peak hour indicators

**Account Features (40+)**
- Account age, customer age
- Dormancy and reactivation status
- Trust scores, MPIN status, filer status
- Geographic features (same city/region)

**Interaction Features (10+)**
- Sender-receiver pair history
- Amount comparison ratios
- First interaction flags

**Categorical Encoding**
- Frequency encoding for channels and transaction types
- Target encoding (fraud rate by category)
- Rare category detection

### Machine Learning Models

- **XGBoost**: Gradient boosting with class weight balancing
- **Random Forest**: Ensemble learning with balanced classes

### Evaluation Metrics

- ROC-AUC (overall discriminative ability)
- PR-AUC (precision-recall for imbalanced data)
- F1-Score, Precision, Recall
- Matthews Correlation Coefficient (MCC)
- Precision in top 5% scored transactions
- Confusion matrix analysis

## Prerequisites

### System Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 500GB+ for full dataset
- **OS**: Linux/MacOS (Windows via WSL)

### Software Requirements

- **Python**: 3.8 or higher
- **Apache Spark**: 3.4+
- **PostgreSQL**: 12+ (with JDBC driver)
- **Java**: 8 or 11 (for Spark)

### PostgreSQL JDBC Driver

Download the PostgreSQL JDBC driver:

```bash
mkdir -p utils
cd utils
wget https://jdbc.postgresql.org/download/postgresql-42.7.1.jar
cd ..
```

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd jazzcash-fraud-detection
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Spark Installation

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

## Configuration

### 1. Database Configuration

Edit `config/db_config.yaml`:

```yaml
postgresql:
  host: "YOUR_DB_HOST"
  port: "5432"
  database: "db_fraud"
  user: "YOUR_USERNAME"
  password: "YOUR_PASSWORD"
  jdbc_driver_path: "/path/to/postgresql-42.7.1.jar"
```

### 2. Spark Configuration

Edit `config/spark_config.yaml` to adjust resources:

```yaml
active_profile: "development"  # or "production"

profiles:
  development:
    executor_memory: "4g"
    executor_cores: 2
    executor_instances: 2
    shuffle_partitions: 20

  production:
    executor_memory: "20g"
    executor_cores: 4
    executor_instances: 20
    shuffle_partitions: 200
```

### 3. Processing Mode

Edit `config/db_config.yaml` to set processing mode:

```yaml
processing_mode:
  current: "sample"  # Options: sample, july, full
```

- **sample**: 4.6M rows (development/testing)
- **july**: 412M rows (monthly analysis)
- **full**: 2.1B rows (production)

## Usage

### Quick Start

Run the complete end-to-end pipeline:

```bash
# With sample data (development)
python pipelines/main_pipeline.py --mode sample

# With July data
python pipelines/main_pipeline.py --mode july --model xgboost

# With full dataset (requires substantial resources)
python pipelines/main_pipeline.py --mode full --model xgboost
```

### Running Individual Pipelines

#### 1. Data Extraction & Cleaning

```bash
python pipelines/data_pipeline.py --mode sample
```

**Output**: `data/processed/cleaned_dataset.parquet`

#### 2. Feature Engineering

```bash
python pipelines/feature_pipeline.py \
  --input data/processed/cleaned_dataset.parquet \
  --output data/features/feature_dataset.parquet
```

**Output**: `data/features/feature_dataset.parquet`

#### 3. Model Training

```bash
python pipelines/model_pipeline.py \
  --input data/features/feature_dataset.parquet \
  --model xgboost
```

**Output**:
- Model: `data/models/xgboost_model.pkl`
- Metrics: `data/models/xgboost_metrics.csv`
- Plots: `plots/model/`

### Advanced Usage

#### Skip Completed Stages

```bash
# Skip data extraction, use existing cleaned data
python pipelines/main_pipeline.py --skip-data --mode sample

# Skip both data and feature stages
python pipelines/main_pipeline.py --skip-data --skip-features --model xgboost
```

#### Train with Sample Size

```bash
# Train on 100K samples for faster iteration
python pipelines/main_pipeline.py --mode july --sample 100000
```

#### Compare Models

```bash
# Train Random Forest
python pipelines/model_pipeline.py --model random_forest

# Train XGBoost
python pipelines/model_pipeline.py --model xgboost

# Compare metrics
cat data/models/*_metrics.csv
```

### Using Jupyter Notebooks

Start Jupyter Lab:

```bash
jupyter lab
```

Navigate to `notebooks/` and run notebooks in order:

1. **01_data_extraction.ipynb**: Learn data extraction from PostgreSQL
2. **02_eda.ipynb**: Explore transaction patterns and fraud characteristics
3. **03_data_cleaning.ipynb**: Understand data cleaning procedures
4. **04_feature_engineering.ipynb**: Deep dive into feature creation
5. **05_feature_selection.ipynb**: Analyze feature importance and selection
6. **06_model_training.ipynb**: Train and tune models interactively
7. **07_model_evaluation.ipynb**: Comprehensive model evaluation

## Data Pipeline

### Extraction Process

The data extraction pipeline:

1. **Parallel Loading**: Uses date-based predicates for parallel reads
2. **IAR + MBAR Joins**: Enriches transactions with account information
3. **Fraud Labeling**: Adds fraud labels from fraud table
4. **Partitioned Storage**: Saves data partitioned by date and fraud status

### Cleaning Process

Data cleaning includes:

1. **Deduplication**: Removes duplicate trans_ids
2. **Missing Value Handling**:
   - Median imputation for numerical features
   - "UNKNOWN" for categorical features
   - Missing indicators for key columns
3. **Outlier Detection**:
   - IQR method (default)
   - Z-score method (optional)
   - Flagging (not removal) to preserve fraud signals
4. **Format Standardization**:
   - Date/timestamp normalization
   - String trimming and case normalization

## Feature Engineering

### Time Windows

Features are computed over multiple time windows:
- Short-term: 1 day, 3 days
- Medium-term: 7 days, 15 days
- Long-term: 30 days, 60 days, 90 days

### Aggregation Features

For each account (sender/receiver) and time window:
- Transaction count
- Total/average/max/min amounts
- Standard deviation
- Unique counterparties
- Channel diversity
- Transaction type diversity

### Example Features

```python
# Velocity features
velocity_1d_to_7d = tx_count_1d / (tx_count_7d + 1)
amount_acceleration = avg_amt_1d / (avg_amt_7d + 1)

# Burst detection
is_activity_burst = (tx_count_1d > 2 * tx_count_7d) & (tx_count_1d > 5)

# Risk indicators
is_new_account = account_age_days < 30
is_odd_hour = (hour >= 0) & (hour <= 5)
is_round_amount = (amount % 1000 == 0)
```

## Model Training

### Class Imbalance Handling

Given the severe class imbalance (~0.0016% fraud rate):

1. **Stratified Sampling**: Maintain fraud ratio in train/test splits
2. **Class Weights**: Use `scale_pos_weight` in XGBoost
3. **Oversampling**: Include all fraud cases, sample non-fraud
4. **Metric Selection**: Focus on PR-AUC over ROC-AUC

### Hyperparameter Tuning

Default XGBoost parameters:
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': auto_calculated,
    'eval_metric': 'logloss'
}
```

Tune via notebooks for better performance.

## Performance Metrics

### Expected Performance (Sample Data)

| Metric | Target | Typical |
|--------|--------|---------|
| ROC-AUC | >0.80 | 0.85-0.90 |
| PR-AUC | >0.30 | 0.35-0.45 |
| Precision @ 90% Recall | >0.15 | 0.18-0.25 |
| Top 5% Precision | >0.40 | 0.45-0.60 |

### Computational Performance

| Stage | Sample (4.6M) | July (412M) | Full (2.1B) |
|-------|---------------|-------------|-------------|
| Data Extraction | ~5 min | ~45 min | ~4 hours |
| Feature Engineering | ~10 min | ~90 min | ~8 hours |
| Model Training | ~2 min | ~15 min | ~60 min |
| **Total** | **~17 min** | **~2.5 hours** | **~13 hours** |

*Times approximate, vary with cluster resources*

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

**Solution**: Increase executor memory or reduce partitions

```yaml
# config/spark_config.yaml
executor:
  memory: "32g"  # Increase this
  memory_overhead: "2g"
```

#### 2. JDBC Connection Timeout

**Solution**: Check network/credentials, increase timeout

```yaml
# config/spark_config.yaml
network:
  timeout: "1200s"  # Increase from 600s
```

#### 3. Slow Feature Engineering

**Solution**: Enable checkpointing and increase partitions

```python
# In feature pipeline
spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")
df = df.checkpoint()  # After expensive operations
```

#### 4. Skewed Partitions

**Solution**: Repartition by high-cardinality columns

```python
df = df.repartition(200, "ac_from", "data_date")
```

### Debug Mode

Enable verbose logging:

```python
# In any pipeline
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs in `logs/` directory.

## Directory Setup

Create necessary directories:

```bash
mkdir -p data/{raw,processed,features,models}
mkdir -p plots/{eda,model}
mkdir -p logs
```

## Example Workflow

Complete example workflow:

```bash
# 1. Set up environment
source venv/bin/activate

# 2. Configure database
# Edit config/db_config.yaml with your credentials

# 3. Download JDBC driver
wget -P utils https://jdbc.postgresql.org/download/postgresql-42.7.1.jar

# 4. Run pipeline with sample data
python pipelines/main_pipeline.py --mode sample

# 5. Explore results in Jupyter
jupyter lab notebooks/07_model_evaluation.ipynb

# 6. If satisfied, run with full July data
python pipelines/main_pipeline.py --mode july --model xgboost

# 7. Compare models
python pipelines/model_pipeline.py --model random_forest
# Compare metrics
cat data/models/*_metrics.csv
```

## Output Files

After successful run:

```
data/
├── processed/
│   ├── master_dataset.parquet/          # Raw combined data
│   └── cleaned_dataset.parquet/         # Cleaned data
├── features/
│   └── feature_dataset.parquet/         # Feature-engineered data
└── models/
    ├── xgboost_model.pkl                # Trained model
    ├── xgboost_metrics.csv              # Performance metrics
    └── xgboost_feature_importance.csv   # Feature importance

plots/
├── eda/
│   ├── amount_distribution.png
│   ├── fraud_rate_by_channel.png
│   └── fraud_rate_by_type.png
└── model/
    ├── xgboost_feature_importance.png
    ├── xgboost_roc_curve.png
    ├── xgboost_pr_curve.png
    └── xgboost_confusion_matrix.png
```

## Performance Optimization Tips

1. **Use Sample Data** for development and testing
2. **Enable Adaptive Query Execution** in Spark config
3. **Broadcast Small Tables** (< 100MB)
4. **Checkpoint After Expensive Operations**
5. **Partition by Date** for time-based queries
6. **Coalesce Before Writing** to reduce file count
7. **Cache Frequently Used DataFrames**

## Next Steps

After running the pipeline:

1. **Hyperparameter Tuning**: Use GridSearchCV or Bayesian optimization
2. **Feature Selection**: Use RFE or SHAP for feature pruning
3. **Ensemble Methods**: Combine XGBoost and Random Forest
4. **Deep Learning**: Try neural networks for complex patterns
5. **Real-time Scoring**: Deploy model for online fraud detection
6. **Monitoring**: Track model performance over time
7. **Retraining**: Schedule periodic retraining with new data

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Specify your license]

## Contact

[Your contact information]

---

**Note**: This system is designed for JazzCash fraud detection. Ensure you have proper authorization to access and process the data. Always follow data privacy and security best practices.
