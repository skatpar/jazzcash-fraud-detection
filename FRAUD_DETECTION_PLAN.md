# JazzCash Fraud Detection - Strategic Data Integration and Feature Engineering Plan

## Executive Summary

This document outlines a comprehensive strategy for building a fraud detection system for JazzCash by integrating multiple data sources, cleaning the data, and engineering optimal features for machine learning models.

---

## 1. DATA INTEGRATION STRATEGY

### 1.1 Dataset Overview

#### Dataset A: IAR Transactions
- **Fields**: `tid`, `time`, `sender`, `receiver`, `channel`, `type`, `amount`, `balance`
- **Granularity**: Transaction-level
- **Primary Key**: `tid` (Transaction ID)
- **Purpose**: Core transactional behavior and patterns

#### Dataset B: Mbar (Customer Master)
- **Fields**: `registration_date`, `channel`, `age`, `location`, `account_type`, `limits`, `kyc_status`
- **Granularity**: Customer-level
- **Primary Key**: Customer identifier (inferred: `sender`/`receiver` from IAR)
- **Purpose**: Customer demographics and account characteristics

#### Dataset C: Fraud Labels
- **Fields**: Contains only fraud transactions (subset of IAR)
- **Primary Key**: `tid` (Transaction ID)
- **Purpose**: Ground truth labels for supervised learning

### 1.2 Data Integration Approach

#### Phase 1: Schema Alignment and Key Mapping

**Step 1.1: Identify Join Keys**
```
IAR (tid) ←→ Fraud (tid)                    [1:1 relationship]
IAR (sender) ←→ Mbar (customer_id)           [Many:1 relationship]
IAR (receiver) ←→ Mbar (customer_id)         [Many:1 relationship]
```

**Step 1.2: Standardize Field Names**
- Ensure consistent naming conventions across datasets
- Map equivalent fields (e.g., `tid` vs `transaction_id`)
- Document data dictionaries for each source

#### Phase 2: Multi-Level Integration Strategy

**Strategy 2.1: Transaction-Fraud Merge (Base Layer)**
```python
# Left join IAR with Fraud labels
base_data = IAR.merge(Fraud, on='tid', how='left', indicator=True)
base_data['is_fraud'] = (base_data['_merge'] == 'both').astype(int)
```
- **Result**: All transactions with fraud label (0/1)
- **Rows**: All IAR transactions
- **Key Addition**: `is_fraud` binary target variable

**Strategy 2.2: Sender Enrichment (Customer Features)**
```python
# Add sender (customer) information
enriched_data = base_data.merge(
    Mbar,
    left_on='sender',
    right_on='customer_id',
    how='left',
    suffixes=('', '_sender')
)
```
- **Result**: Each transaction enriched with sender's profile
- **Added Features**: sender demographics, account type, KYC status, limits

**Strategy 2.3: Receiver Enrichment (Network Features)**
```python
# Add receiver information for network analysis
final_data = enriched_data.merge(
    Mbar,
    left_on='receiver',
    right_on='customer_id',
    how='left',
    suffixes=('_sender', '_receiver')
)
```
- **Result**: Complete view of both transaction parties
- **Added Features**: receiver demographics and characteristics
- **Use Case**: Network analysis, peer group patterns

#### Phase 3: Data Integrity Checks

**Check 3.1: Join Quality Metrics**
- Transaction coverage: % of IAR transactions successfully joined
- Customer match rate: % of senders/receivers found in Mbar
- Null analysis: Distribution of missing joins by channel/type

**Check 3.2: Cardinality Validation**
- Verify no duplicate transactions after merge
- Check for unexpected many-to-many relationships
- Validate primary key uniqueness: `assert final_data['tid'].is_unique`

**Check 3.3: Temporal Alignment**
```python
# Ensure transaction time >= sender registration_date
temporal_check = final_data[
    final_data['time'] < final_data['registration_date_sender']
]
# Flag: transactions before sender account creation (data quality issue)
```

---

## 2. DATA CLEANING STRATEGY

### 2.1 Missing Value Analysis and Treatment

#### Step 1: Missing Data Profiling
```python
# Generate missing value report
missing_profile = {
    'column': [],
    'missing_count': [],
    'missing_pct': [],
    'data_type': [],
    'imputation_strategy': []
}
```

#### Step 2: Feature-Specific Imputation Strategies

**Numerical Features**

| Feature | Imputation Method | Rationale |
|---------|-------------------|-----------|
| `amount` | **No imputation** (critical field) | Missing amounts indicate data quality issues; drop rows |
| `balance` | **Forward fill** by customer → median | Use last known balance, fallback to customer median |
| `age` | **Median** by location | Age correlates with geography |
| `limits` | **Median** by account_type | Transaction limits depend on account tier |

**Categorical Features**

| Feature | Imputation Method | Rationale |
|---------|-------------------|-----------|
| `channel` | **Mode** (most frequent) | Represent typical transaction channel |
| `type` | **'UNKNOWN'** category | Preserve missingness as signal |
| `location` | **'UNSPECIFIED'** | Geographic missing may indicate fraud |
| `kyc_status` | **'NOT_VERIFIED'** | Conservative assumption for missing KYC |
| `account_type` | **Mode** by channel | Account type correlates with channel usage |

**Temporal Features**

| Feature | Imputation Method | Rationale |
|---------|-------------------|-----------|
| `registration_date` | **Drop rows** or **median** registration date | Critical for tenure calculation |
| `time` | **No imputation** | Transaction timestamp is mandatory |

#### Step 3: Missing Value Creation of Indicator Features
```python
# Create binary flags for important missing values
final_data['missing_balance'] = final_data['balance'].isna().astype(int)
final_data['missing_location'] = final_data['location'].isna().astype(int)
final_data['missing_kyc'] = final_data['kyc_status'].isna().astype(int)
```
**Rationale**: Missingness itself can be predictive of fraud

### 2.2 Outlier Detection and Treatment

#### Method 1: Statistical Outliers (Numerical Features)

**IQR Method for Transaction Amounts**
```python
Q1 = final_data['amount'].quantile(0.25)
Q3 = final_data['amount'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_upper = Q3 + 3 * IQR  # 3 IQRs for extreme values
outlier_threshold_lower = Q1 - 3 * IQR
```

**Treatment Strategy**:
- **DO NOT DROP** outliers in fraud detection (high amounts may be fraud)
- **Flag** outliers as features: `is_amount_outlier`
- **Cap** at 99.9th percentile for modeling stability (optional)

**Domain-Specific Outlier Rules**
- Age: Flag if < 13 or > 100
- Balance: Flag if negative (overdraft) or > 99.9th percentile
- Transaction frequency: Flag customers with > 50 transactions/day

#### Method 2: Isolation Forest (Multivariate Outliers)

```python
from sklearn.ensemble import IsolationForest

# Apply to numerical features for anomaly scoring
iso_forest = IsolationForest(contamination=0.05, random_state=42)
final_data['anomaly_score'] = iso_forest.fit_predict(
    final_data[['amount', 'balance', 'age', 'hour_of_day']]
)
```
**Use Case**: Identify unusual combinations of features (not just univariate outliers)

### 2.3 Data Standardization

#### Schema Standardization
- **Date formats**: Convert all dates to `datetime64` format
- **Amount precision**: Standardize to 2 decimal places
- **String normalization**: Lowercase, trim whitespace, remove special characters
- **Categorical encoding**: Standardize category labels (e.g., 'Mobile' vs 'MOBILE' vs 'mobile')

#### Value Standardization
```python
# Channel standardization
channel_mapping = {
    'mobile': 'MOBILE',
    'Mobile': 'MOBILE',
    'web': 'WEB',
    'Web': 'WEB',
    'agent': 'AGENT',
    # ... etc
}
final_data['channel'] = final_data['channel'].map(channel_mapping)
```

### 2.4 Data Quality Rules

**Hard Rules (Drop Records)**
1. Missing `tid`, `time`, or `amount` → Drop
2. Duplicate `tid` → Keep first occurrence, drop duplicates
3. `amount` <= 0 → Drop (invalid transactions)
4. `time` < '2020-01-01' → Drop (suspiciously old data)

**Soft Rules (Flag for Review)**
1. Transaction before sender registration → Flag `invalid_temporal_order`
2. Amount > 99.9th percentile → Flag `high_value_transaction`
3. Missing receiver in Mbar → Flag `unregistered_receiver`

---

## 3. FEATURE ENGINEERING STRATEGY

### 3.1 Transaction-Level Features (Direct Derivation)

#### Temporal Features
```python
# Extract time components
final_data['hour'] = final_data['time'].dt.hour
final_data['day_of_week'] = final_data['time'].dt.dayofweek
final_data['is_weekend'] = final_data['day_of_week'].isin([5, 6]).astype(int)
final_data['is_night'] = final_data['hour'].isin(range(0, 6)).astype(int)
final_data['month'] = final_data['time'].dt.month
final_data['day'] = final_data['time'].dt.day
```

**Rationale**: Fraud patterns often vary by time (e.g., night transactions, weekends)

#### Amount-Based Features
```python
# Transaction amount features
final_data['amount_log'] = np.log1p(final_data['amount'])
final_data['amount_round'] = (final_data['amount'] % 100 == 0).astype(int)
final_data['amount_to_balance_ratio'] = final_data['amount'] / (final_data['balance'] + 1)
```

**Rationale**:
- Log transform handles skewness
- Round amounts may indicate automated fraud
- Amount-to-balance ratio flags risky transactions

#### Customer Tenure Features
```python
# Account age at transaction time
final_data['sender_tenure_days'] = (
    final_data['time'] - final_data['registration_date_sender']
).dt.days
final_data['receiver_tenure_days'] = (
    final_data['time'] - final_data['registration_date_receiver']
).dt.days
final_data['is_new_sender'] = (final_data['sender_tenure_days'] < 30).astype(int)
```

**Rationale**: New accounts are higher fraud risk

### 3.2 Aggregation Features (Customer Behavior)

#### Sender Aggregations (Rolling Windows)
```python
# 1-day, 7-day, 30-day windows
for window in [1, 7, 30]:
    # Transaction count
    final_data[f'sender_txn_count_{window}d'] = final_data.groupby('sender')[
        'time'
    ].transform(lambda x: x.rolling(f'{window}D').count())

    # Total amount sent
    final_data[f'sender_amount_sum_{window}d'] = final_data.groupby('sender')[
        'amount'
    ].transform(lambda x: x.rolling(f'{window}D').sum())

    # Average transaction amount
    final_data[f'sender_amount_avg_{window}d'] = final_data.groupby('sender')[
        'amount'
    ].transform(lambda x: x.rolling(f'{window}D').mean())

    # Unique receivers
    final_data[f'sender_unique_receivers_{window}d'] = final_data.groupby('sender')[
        'receiver'
    ].transform(lambda x: x.rolling(f'{window}D').nunique())
```

#### Receiver Aggregations (Incoming Pattern)
```python
# Identify frequently used receivers (potential mule accounts)
for window in [1, 7, 30]:
    final_data[f'receiver_txn_count_{window}d'] = final_data.groupby('receiver')[
        'time'
    ].transform(lambda x: x.rolling(f'{window}D').count())

    final_data[f'receiver_unique_senders_{window}d'] = final_data.groupby('receiver')[
        'sender'
    ].transform(lambda x: x.rolling(f'{window}D').nunique())
```

#### Velocity Features (Change Detection)
```python
# Sudden spikes in activity
final_data['amount_spike'] = (
    final_data['sender_amount_sum_1d'] /
    (final_data['sender_amount_avg_30d'] * 30 + 1)
)
final_data['txn_frequency_spike'] = (
    final_data['sender_txn_count_1d'] /
    (final_data['sender_txn_count_30d'] / 30 + 1)
)
```

**Rationale**: Sudden changes in behavior indicate account takeover or fraud

### 3.3 Categorical Encoding

#### Frequency Encoding (High Cardinality)
```python
# Encode by fraud rate for each category
def frequency_encode(df, column):
    freq = df[column].value_counts(normalize=True)
    return df[column].map(freq)

final_data['channel_freq'] = frequency_encode(final_data, 'channel')
final_data['type_freq'] = frequency_encode(final_data, 'type')
final_data['location_freq'] = frequency_encode(final_data, 'location')
```

#### Target Encoding (Mean Encoding)
```python
# Encode by fraud rate for each category (use cross-validation to prevent leakage)
def target_encode(train, test, column, target, smoothing=10):
    # Global mean
    global_mean = train[target].mean()

    # Category means
    agg = train.groupby(column)[target].agg(['mean', 'count'])
    smoothed_mean = (agg['mean'] * agg['count'] + global_mean * smoothing) / (
        agg['count'] + smoothing
    )

    # Apply to test
    return test[column].map(smoothed_mean).fillna(global_mean)

# Apply during model training (with proper CV)
final_data['channel_fraud_rate'] = target_encode(train, test, 'channel', 'is_fraud')
final_data['type_fraud_rate'] = target_encode(train, test, 'type', 'is_fraud')
```

**Rationale**: Certain channels/types have higher fraud rates

#### One-Hot Encoding (Low Cardinality)
```python
# For features with < 10 unique values
final_data = pd.get_dummies(
    final_data,
    columns=['account_type_sender', 'kyc_status_sender'],
    drop_first=True  # Avoid multicollinearity
)
```

### 3.4 Network Features

#### Sender-Receiver Relationship
```python
# Historical relationship strength
final_data['sender_receiver_prev_txns'] = final_data.groupby(
    ['sender', 'receiver']
).cumcount()

final_data['is_first_time_receiver'] = (
    final_data['sender_receiver_prev_txns'] == 0
).astype(int)
```

**Rationale**: First-time interactions are riskier

#### Graph-Based Features (Advanced)
```python
# Degree centrality: How connected is the sender?
sender_degree = final_data.groupby('sender')['receiver'].nunique()
receiver_degree = final_data.groupby('receiver')['sender'].nunique()

final_data['sender_degree'] = final_data['sender'].map(sender_degree)
final_data['receiver_degree'] = final_data['receiver'].map(receiver_degree)
```

### 3.5 Interaction Features

```python
# Channel × Type fraud risk
final_data['channel_type_interaction'] = (
    final_data['channel'].astype(str) + '_' + final_data['type'].astype(str)
)

# Amount × Channel risk
final_data['high_amount_mobile'] = (
    (final_data['amount'] > final_data['amount'].quantile(0.9)) &
    (final_data['channel'] == 'MOBILE')
).astype(int)

# Age × Amount risk (unusual for demographics)
final_data['age_amount_interaction'] = final_data['age'] * final_data['amount_log']
```

---

## 4. FEATURE IMPORTANCE AND SELECTION

### 4.1 Correlation Analysis

#### Pearson Correlation (Linear Relationships)
```python
# Compute correlation matrix for numerical features
numerical_features = final_data.select_dtypes(include=[np.number]).columns
correlation_matrix = final_data[numerical_features].corr()

# Identify highly correlated features (|r| > 0.9)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))
```

**Decision Rule**: If two features have |r| > 0.9, drop the one with lower correlation to target (`is_fraud`)

#### Point-Biserial Correlation (Feature vs Target)
```python
# Correlation of each feature with fraud label
feature_target_corr = final_data[numerical_features].corrwith(
    final_data['is_fraud']
).abs().sort_values(ascending=False)
```

### 4.2 Multicollinearity Detection (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    vif_data['VIF'] = [
        variance_inflation_factor(df[features].values, i)
        for i in range(len(features))
    ]
    return vif_data.sort_values('VIF', ascending=False)

vif_results = calculate_vif(final_data, numerical_features)
```

**Decision Rule**:
- VIF > 10: High multicollinearity → Drop feature
- VIF 5-10: Moderate → Monitor
- VIF < 5: Acceptable

**Iterative Process**:
1. Calculate VIF
2. Drop feature with highest VIF > 10
3. Recalculate VIF
4. Repeat until all VIF < 10

### 4.3 Feature Selection Methods

#### Method 1: Filter Methods (Univariate)

**Chi-Square Test (Categorical Features)**
```python
from sklearn.feature_selection import chi2, SelectKBest

# For categorical features
chi2_selector = SelectKBest(chi2, k=20)
chi2_selector.fit(X_categorical, y)
chi2_scores = pd.DataFrame({
    'feature': X_categorical.columns,
    'chi2_score': chi2_selector.scores_
}).sort_values('chi2_score', ascending=False)
```

**Mutual Information (All Features)**
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)
```

#### Method 2: Embedded Methods (Model-Based)

**Random Forest Feature Importance**
```python
from sklearn.ensemble import RandomForestClassifier

# Train RF model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Extract feature importance
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
```

**XGBoost Feature Importance (Gain, Cover, Frequency)**
```python
import xgboost as xgb

# Train XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Get importance scores
feature_importance_xgb = pd.DataFrame({
    'feature': X.columns,
    'gain': xgb_model.get_booster().get_score(importance_type='gain'),
    'cover': xgb_model.get_booster().get_score(importance_type='cover'),
    'frequency': xgb_model.get_booster().get_score(importance_type='weight')
})
```

**LASSO (L1 Regularization)**
```python
from sklearn.linear_model import LogisticRegression

# L1 regularization for feature selection
lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
lasso.fit(X_train, y_train)

# Features with non-zero coefficients
selected_features_lasso = X.columns[lasso.coef_[0] != 0]
```

#### Method 3: Wrapper Methods (Iterative Selection)

**Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier

# Use gradient boosting as base estimator
estimator = GradientBoostingClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator, n_features_to_select=50, step=5)
rfe.fit(X_train, y_train)

# Get selected features
selected_features_rfe = X.columns[rfe.support_]
```

**Forward/Backward Selection (Custom)**
```python
from sklearn.model_selection import cross_val_score

def forward_selection(X, y, feature_list, n_features=20):
    selected = []
    remaining = list(feature_list)

    for i in range(n_features):
        best_score = 0
        best_feature = None

        for feature in remaining:
            features = selected + [feature]
            score = cross_val_score(
                RandomForestClassifier(random_state=42),
                X[features], y, cv=5, scoring='roc_auc'
            ).mean()

            if score > best_score:
                best_score = score
                best_feature = feature

        selected.append(best_feature)
        remaining.remove(best_feature)
        print(f"Added {best_feature}, AUC: {best_score:.4f}")

    return selected
```

### 4.4 Feature Selection Strategy (Ensemble Approach)

**Step 1: Create Feature Importance Ranking from Multiple Methods**
```python
# Normalize scores to 0-1 range
def normalize_scores(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

# Combine rankings
combined_importance = pd.DataFrame({'feature': X.columns})
combined_importance['rf_importance'] = normalize_scores(feature_importance_rf['importance'])
combined_importance['xgb_importance'] = normalize_scores(feature_importance_xgb['gain'])
combined_importance['mi_score'] = normalize_scores(mi_scores)
combined_importance['target_corr'] = normalize_scores(feature_target_corr)

# Average rank
combined_importance['avg_importance'] = combined_importance[
    ['rf_importance', 'xgb_importance', 'mi_score', 'target_corr']
].mean(axis=1)
```

**Step 2: Apply Multi-Stage Filter**
1. **Stage 1 - Variance Filter**: Remove features with near-zero variance
2. **Stage 2 - Correlation Filter**: Remove highly correlated features (keep one with higher importance)
3. **Stage 3 - VIF Filter**: Remove features with VIF > 10
4. **Stage 4 - Importance Filter**: Keep top N features by combined importance score
5. **Stage 5 - Model Validation**: Final selection based on CV performance

**Step 3: Validation**
```python
# Compare model performance with different feature sets
feature_sets = {
    'top_20': top_20_features,
    'top_50': top_50_features,
    'top_100': top_100_features,
    'all_filtered': all_features_after_vif
}

for name, features in feature_sets.items():
    model = XGBClassifier(random_state=42)
    scores = cross_val_score(model, X_train[features], y_train, cv=5, scoring='roc_auc')
    print(f"{name}: AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 5. IMPLEMENTATION ROADMAP

### Phase 1: Data Integration (Week 1)
- [ ] Load and profile all three datasets
- [ ] Standardize schemas and identify join keys
- [ ] Perform left joins: IAR → Fraud → Mbar (sender) → Mbar (receiver)
- [ ] Validate join quality and cardinality
- [ ] Create integrated dataset checkpoint

### Phase 2: Data Cleaning (Week 1-2)
- [ ] Generate missing value profile
- [ ] Implement imputation strategies
- [ ] Detect and flag outliers (do not drop)
- [ ] Standardize data formats and values
- [ ] Apply data quality rules
- [ ] Create clean dataset checkpoint

### Phase 3: Feature Engineering (Week 2-3)
- [ ] Create temporal features (hour, day, weekend, etc.)
- [ ] Develop transaction-level features (amount ratios, logs)
- [ ] Build aggregation features (rolling windows: 1d, 7d, 30d)
- [ ] Implement categorical encoding (frequency, target, one-hot)
- [ ] Generate network features (sender-receiver relationships)
- [ ] Create interaction features
- [ ] Create feature-engineered dataset checkpoint

### Phase 4: Feature Selection (Week 3-4)
- [ ] Compute correlation matrix and identify high correlations
- [ ] Calculate VIF and remove multicollinear features
- [ ] Run multiple feature importance methods (RF, XGBoost, MI)
- [ ] Combine importance scores using ensemble approach
- [ ] Apply multi-stage filtering
- [ ] Validate feature sets using cross-validation
- [ ] Document final feature set and rationale

### Phase 5: Model Development (Week 4-5)
- [ ] Split data: Train (60%), Validation (20%), Test (20%)
- [ ] Train baseline models (Logistic Regression, Random Forest)
- [ ] Train advanced models (XGBoost, LightGBM, CatBoost)
- [ ] Hyperparameter tuning using validation set
- [ ] Evaluate on test set (AUC-ROC, Precision-Recall, F1)
- [ ] Analyze feature importance in final model

### Phase 6: Model Validation and Deployment (Week 5-6)
- [ ] Perform temporal validation (train on old, test on recent)
- [ ] Analyze false positives and false negatives
- [ ] Create model documentation and deployment package
- [ ] Set up monitoring for model performance

---

## 6. KEY CONSIDERATIONS FOR FRAUD DETECTION

### 6.1 Class Imbalance
- Fraud is typically < 1% of transactions
- **Strategies**:
  - SMOTE or ADASYN for oversampling minority class
  - Class weights in model training
  - Evaluation metrics: Precision-Recall AUC (not just ROC-AUC)

### 6.2 Temporal Leakage Prevention
- **Never** use future information in features
- Aggregations must use only historical data (up to transaction time)
- Use time-based splits for train/test (not random splits)

### 6.3 Feature Interpretability
- Fraud models often require explainability for compliance
- Prioritize interpretable features when performance is similar
- Use SHAP values to explain predictions

### 6.4 Real-Time Inference Constraints
- Aggregation features may be expensive to compute in real-time
- Design features for efficient computation and caching
- Consider feature store for pre-computed aggregations

---

## 7. SUCCESS METRICS

### Data Quality Metrics
- Join coverage: > 95% of transactions matched with customer data
- Missing value rate: < 5% after imputation
- Outlier rate: < 2% (flagged, not removed)

### Feature Engineering Metrics
- Feature count: 50-150 features after selection
- VIF: All features < 10
- Feature importance: Top 20 features explain > 80% of model importance

### Model Performance Metrics (Minimum Targets)
- AUC-ROC: > 0.85
- Precision-Recall AUC: > 0.70
- Precision @ 90% Recall: > 0.30
- False Positive Rate @ 80% Recall: < 5%

---

## 8. TOOLING AND LIBRARIES

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Feature engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import TargetEncoder

# Feature selection
from sklearn.feature_selection import (
    mutual_info_classif, chi2, SelectKBest, RFE
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Modeling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Evaluation
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Explainability
import shap
```

---

## 9. DELIVERABLES

1. **Integrated Dataset**: Single CSV/Parquet with all merged data
2. **Data Quality Report**: Missing values, outliers, join coverage
3. **Feature Engineering Pipeline**: Reproducible Python scripts
4. **Feature Importance Report**: Rankings from multiple methods
5. **Final Feature Set**: Documented list with rationale
6. **Model Performance Report**: Metrics on test set
7. **Deployment Package**: Trained model + feature engineering code

---

## 10. RISK MITIGATION

| Risk | Impact | Mitigation |
|------|--------|------------|
| Poor join coverage (< 80%) | Missing customer features | Investigate data quality, use imputation for missing joins |
| High class imbalance (fraud < 0.1%) | Model bias toward majority class | Apply SMOTE, adjust class weights, focus on Precision-Recall |
| Data leakage | Overly optimistic metrics | Strict temporal splits, careful feature engineering |
| High multicollinearity | Unstable model coefficients | VIF analysis, regularization (L1/L2) |
| Concept drift | Model degrades over time | Implement monitoring, retrain quarterly |

---

**Document Version**: 1.0
**Last Updated**: 2025-10-21
**Author**: JazzCash Fraud Detection Team
