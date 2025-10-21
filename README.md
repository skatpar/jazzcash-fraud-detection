# JazzCash Fraud Detection

A comprehensive fraud detection system for JazzCash transactions using machine learning. This project integrates multiple data sources, performs advanced feature engineering, and applies state-of-the-art feature selection techniques to build high-performance fraud detection models.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Phases](#pipeline-phases)
- [Documentation](#documentation)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Overview

This project implements a complete machine learning pipeline for fraud detection, including:

1. **Data Integration**: Merging transaction data (IAR), customer data (Mbar), and fraud labels
2. **Data Cleaning**: Handling missing values, outliers, and data quality issues
3. **Feature Engineering**: Creating 100+ features including temporal, aggregation, network, and interaction features
4. **Feature Selection**: Using correlation analysis, VIF, and multiple ML-based methods to select optimal features
5. **Model Training**: Training and evaluating multiple fraud detection models

### Key Features

- **Modular Architecture**: Clean separation of concerns with reusable components
- **Comprehensive Feature Engineering**: 16+ temporal features, 30+ aggregation features, network features, and more
- **Multi-Stage Feature Selection**: Combines correlation analysis, VIF reduction, and ensemble importance scoring
- **Configurable Pipeline**: YAML-based configuration for easy experimentation
- **Production-Ready**: Includes data validation, logging, and error handling

## Project Structure

```
jazzcash-fraud-detection/
│
├── data/
│   ├── raw/                          # Raw datasets (not in git)
│   │   ├── iar_transactions.csv
│   │   ├── mbar_customers.csv
│   │   └── fraud_labels.csv
│   ├── processed/                    # Processed datasets
│   │   ├── integrated_dataset.csv
│   │   └── cleaned_dataset.csv
│   └── features/                     # Feature-engineered datasets
│       ├── feature_engineered_dataset.csv
│       ├── selected_features.txt
│       └── feature_validation_results.csv
│
├── src/
│   ├── data_integration/
│   │   ├── merge_datasets.py        # Data integration module
│   │   └── data_cleaner.py          # Data cleaning module
│   ├── feature_engineering/
│   │   ├── feature_builder.py       # Feature engineering module
│   │   └── feature_selector.py      # Feature selection module
│   └── modeling/                     # (Future) Model training modules
│
├── notebooks/
│   ├── 01_data_integration_exploration.ipynb
│   ├── 02_data_cleaning.ipynb       # (To be created)
│   ├── 03_feature_engineering.ipynb # (To be created)
│   └── 04_feature_selection_modeling.ipynb
│
├── config/
│   └── config.yaml                   # Pipeline configuration
│
├── tests/                            # Unit tests
│
├── docs/
│   └── FRAUD_DETECTION_PLAN.md      # Comprehensive strategic plan
│
├── main_pipeline.py                  # Main pipeline orchestrator
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

## Datasets

The project uses three primary datasets:

### 1. IAR Transactions
- **Fields**: `tid`, `time`, `sender`, `receiver`, `channel`, `type`, `amount`, `balance`
- **Description**: Core transactional data with details about each transaction
- **Granularity**: Transaction-level

### 2. Mbar (Customer Master)
- **Fields**: `registration_date`, `channel`, `age`, `location`, `account_type`, `limits`, `kyc_status`
- **Description**: Customer demographics and account characteristics
- **Granularity**: Customer-level

### 3. Fraud Dataset
- **Fields**: Contains only fraud transactions (subset of IAR)
- **Description**: Ground truth labels for supervised learning
- **Granularity**: Transaction-level (fraud cases only)

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/skatpar/jazzcash-fraud-detection.git
cd jazzcash-fraud-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your data files in the `data/raw/` directory:
   - `iar_transactions.csv`
   - `mbar_customers.csv`
   - `fraud_labels.csv`

## Quick Start

### Option 1: Run Complete Pipeline

Run the entire pipeline with default configuration:

```bash
python main_pipeline.py --phase all
```

### Option 2: Run Individual Phases

Run specific phases of the pipeline:

```bash
# Data integration only
python main_pipeline.py --phase integration

# Data cleaning only
python main_pipeline.py --phase cleaning

# Feature engineering only
python main_pipeline.py --phase feature_engineering

# Feature selection only
python main_pipeline.py --phase feature_selection
```

### Option 3: Use Individual Modules

```python
from src.data_integration.merge_datasets import DataIntegrator
from src.data_integration.data_cleaner import DataCleaner
from src.feature_engineering.feature_builder import FeatureEngineer
from src.feature_engineering.feature_selector import FeatureSelector

# Data Integration
integrator = DataIntegrator()
integrated_df = integrator.integrate_all(
    iar_path='data/raw/iar_transactions.csv',
    mbar_path='data/raw/mbar_customers.csv',
    fraud_path='data/raw/fraud_labels.csv',
    output_path='data/processed/integrated_dataset.csv'
)

# Data Cleaning
cleaner = DataCleaner()
clean_df = cleaner.clean_all(integrated_df)

# Feature Engineering
engineer = FeatureEngineer()
feature_df = engineer.engineer_all_features(clean_df, include_aggregations=True)

# Feature Selection
selector = FeatureSelector(target_col='is_fraud')
selected_features = selector.select_features_multistage(feature_df, n_final_features=50)
```

### Option 4: Interactive Notebooks

Explore the data and pipeline interactively:

```bash
jupyter notebook
# Open notebooks/01_data_integration_exploration.ipynb
```

## Pipeline Phases

### Phase 1: Data Integration

**Purpose**: Combine three datasets into a single, unified dataset

**Process**:
1. Load IAR transactions, Mbar customers, and fraud labels
2. Standardize schemas and field names
3. Merge fraud labels with transactions (left join)
4. Enrich with sender customer data (left join)
5. Enrich with receiver customer data (left join)
6. Validate integration quality

**Output**: `data/processed/integrated_dataset.csv`

**Key Metrics**:
- Join coverage: % of transactions matched with customer data
- Fraud rate: % of transactions labeled as fraud
- Data completeness: % of non-null values

### Phase 2: Data Cleaning

**Purpose**: Handle missing values, outliers, and data quality issues

**Process**:
1. Generate missing value profile
2. Impute missing values using feature-specific strategies:
   - Numerical: median, forward fill, grouped imputation
   - Categorical: mode, custom values (e.g., 'UNKNOWN', 'NOT_VERIFIED')
3. Detect and flag outliers (IQR method, Isolation Forest)
4. Standardize data formats and categorical values
5. Apply data quality rules (hard rules drop rows, soft rules flag issues)

**Output**: `data/processed/cleaned_dataset.csv`

**Key Techniques**:
- Missing value indicators for important features
- Multivariate anomaly detection using Isolation Forest
- Temporal consistency validation

### Phase 3: Feature Engineering

**Purpose**: Create rich features to capture fraud patterns

**Process**:
1. **Temporal Features** (16+): hour, day_of_week, is_weekend, is_night, cyclical encoding
2. **Transaction Features** (10+): amount_log, amount_to_balance_ratio, tenure features
3. **Aggregation Features** (30+): Rolling windows (1d, 7d, 30d) for:
   - Transaction count, amount sum/avg/max/std
   - Unique receivers/senders
   - Velocity features (spike detection)
4. **Network Features** (8+): sender-receiver relationships, degree centrality
5. **Categorical Encoding**: frequency encoding, target encoding (with CV)
6. **Interaction Features** (6+): channel×type, age×amount, etc.

**Output**: `data/features/feature_engineered_dataset.csv`

**Feature Count**: 100+ features

### Phase 4: Feature Selection

**Purpose**: Select optimal subset of features to maximize model performance

**Process**:
1. **Correlation Analysis**: Identify and remove highly correlated features (|r| > 0.95)
2. **VIF Analysis**: Iteratively remove features with VIF > 10 to handle multicollinearity
3. **Feature Importance**: Calculate using multiple methods:
   - Mutual Information
   - Random Forest importance
   - LASSO (L1 regularization)
   - Recursive Feature Elimination (RFE)
4. **Ensemble Scoring**: Combine importance scores from multiple methods
5. **Validation**: Cross-validate different feature sets to select optimal size

**Output**:
- `data/features/selected_features.txt` (final feature list)
- `data/features/feature_validation_results.csv` (performance comparison)

**Recommended Feature Count**: 30-50 features

## Documentation

### Strategic Plan

See [FRAUD_DETECTION_PLAN.md](FRAUD_DETECTION_PLAN.md) for:
- Detailed data integration strategy
- Comprehensive data cleaning approach
- Complete feature engineering specifications
- Feature selection methodologies
- Implementation roadmap
- Success metrics and targets

### Module Documentation

Each Python module contains detailed docstrings:
- `src/data_integration/merge_datasets.py`: Data integration
- `src/data_integration/data_cleaner.py`: Data cleaning
- `src/feature_engineering/feature_builder.py`: Feature engineering
- `src/feature_engineering/feature_selector.py`: Feature selection

## Configuration

Edit `config/config.yaml` to customize the pipeline:

```yaml
# Example: Change aggregation windows
feature_engineering:
  aggregation:
    enabled: true
    windows: [1, 7, 14, 30]  # Add 14-day window

# Example: Change number of selected features
feature_selection:
  final:
    n_features: 30  # Select top 30 instead of 50

# Example: Change VIF threshold
feature_selection:
  vif:
    threshold: 5.0  # Stricter multicollinearity control
```

## Expected Results

Based on the strategic plan, you can expect:

### Data Quality
- Join coverage: > 95%
- Missing value rate: < 5% (after imputation)
- Outlier rate: ~2-5% (flagged, not removed)

### Features
- Initial features after engineering: 100-150
- Features after VIF reduction: 60-80
- Final selected features: 30-50
- All features VIF < 10

### Model Performance (Targets)
- AUC-ROC: > 0.85
- Precision-Recall AUC: > 0.70
- Precision @ 90% Recall: > 0.30
- False Positive Rate @ 80% Recall: < 5%

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` when running pipeline
- **Solution**: Ensure data files are in `data/raw/` directory with correct names

**Issue**: Memory error during aggregation features
- **Solution**: Disable aggregations or reduce windows in `config/config.yaml`:
  ```yaml
  feature_engineering:
    aggregation:
      enabled: false
  ```

**Issue**: Long runtime for feature selection
- **Solution**: Reduce number of features or use faster methods in `feature_selector.py`

## Next Steps

After completing the feature engineering and selection pipeline:

1. **Model Development**: Train fraud detection models using selected features
2. **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV or Optuna
3. **Model Evaluation**: Evaluate on hold-out test set with temporal validation
4. **Model Interpretation**: Use SHAP values to explain predictions
5. **Deployment**: Package model for real-time inference

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is developed for JazzCash fraud detection purposes.

## Contact

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Last Updated**: 2025-10-21
**Version**: 1.0
**Status**: Active Development
