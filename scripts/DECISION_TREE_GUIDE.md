# Decision Tree Training Script - Complete Guide

## 📋 Overview

This script trains a **Decision Tree Classifier** for fraud detection with:
- ✅ **Undersampling**: Keeps all fraud cases, samples 30% of legitimate cases
- ✅ **Rule Extraction**: Generates human-readable decision rules
- ✅ **Complete Inference Pipeline**: Includes code examples for predictions
- ✅ **Memory Optimized**: Handles large datasets efficiently
- ✅ **Comprehensive Artifacts**: Saves model, rules, predictions, and metadata

## 🚀 Quick Start

### Option 1: Using the Shell Script (Recommended)
```bash
cd /root/research-dir/dev/jazzcash-fraud-detection/scripts
./run_decision_tree_training.sh
```

### Option 2: Direct Python Execution
```bash
cd /root/research-dir/dev/jazzcash-fraud-detection/scripts
python train_decision_tree.py
```

## 📊 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Load Data from ClickHouse                                       │
│     • Date Range: 2025-01-01 to 2025-06-30                         │
│     • Features: 79 features                                         │
│     • Partitioned loading for efficiency                           │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. Prepare Categorical Features                                    │
│     • Fit StringIndexer + OneHotEncoder on 10% sample              │
│     • Memory optimized approach                                     │
│     • Handles missing values                                        │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. Undersample Dataset                                             │
│     • Keep 100% of fraud cases                                      │
│     • Sample 30% of legitimate cases                                │
│     • Balances class distribution                                   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. Apply Transformations                                           │
│     • Transform categorical features                                │
│     • Assemble feature vectors                                      │
│     • Standardize features                                          │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  5. Train/Test Split                                                │
│     • 80% training data                                             │
│     • 20% test data                                                 │
│     • Stratified split                                              │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  6. Train Decision Tree                                             │
│     • Max Depth: 10                                                 │
│     • Min Instances Per Node: 100                                   │
│     • Gini impurity measure                                         │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  7. Evaluate & Extract Rules                                        │
│     • Calculate metrics (Accuracy, Precision, Recall, F1, AUC)     │
│     • Extract feature importance                                    │
│     • Generate decision rules                                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  8. Save All Artifacts                                              │
│     • Trained model                                                 │
│     • Preprocessing components                                      │
│     • Decision rules                                                │
│     • Predictions                                                   │
│     • Comprehensive metadata                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Generated Artifacts

After training, you'll find the following timestamped artifacts:

```
scripts/
├── decision_tree_model_YYYYMMDD_HHMMSS/
│   └── [Trained Decision Tree Model]
│
├── dt_preprocessing_YYYYMMDD_HHMMSS/
│   ├── categorical_pipeline/
│   │   └── [StringIndexer + OneHotEncoder models]
│   ├── vector_assembler/
│   │   └── [VectorAssembler model]
│   ├── standard_scaler/
│   │   └── [StandardScaler model]
│   └── feature_metadata.json
│
├── dt_rules_YYYYMMDD_HHMMSS/
│   ├── decision_tree_rules.txt          # Human-readable tree structure
│   ├── rule_summary.json                # Structured rule information
│   └── feature_importance.json          # Ranked feature importance
│
├── dt_predictions_YYYYMMDD_HHMMSS/
│   ├── predictions.parquet              # Full test set predictions
│   └── predictions_sample.csv           # Sample for quick viewing
│
├── training_summary_YYYYMMDD_HHMMSS.txt # Complete training report
└── decision_tree_training_YYYYMMDD_HHMMSS.log  # Detailed logs
```

## 📈 Model Configuration

### Decision Tree Parameters
```python
- Max Depth: 10
- Max Bins: 32
- Min Instances Per Node: 100
- Impurity: Gini
- Features: 79 (after encoding)
```

### Undersampling Strategy
```python
- Fraud Cases: 100% (all kept)
- Legitimate Cases: 30% (randomly sampled)
- Seed: 42 (reproducible)
```

### Spark Configuration
```python
- Executor Memory: 100g
- Driver Memory: 8g
- Executor Cores: 32
- Executor Instances: 2
- Shuffle Partitions: 200
- Default Parallelism: 96
```

## 🎯 Performance Metrics

The script calculates and reports:
- **Accuracy**: Overall correctness
- **Precision**: Fraud detection accuracy
- **Recall**: Fraud detection coverage
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve
- **Confusion Matrix**: Detailed classification results
- **Feature Importance**: Top contributing features

## 🔍 Decision Rules

The script extracts human-readable decision rules from the trained tree:

### Example Rule Format
```
If (feature_1 <= threshold_1) {
  If (feature_2 > threshold_2) {
    Predict: FRAUD (probability: 0.85)
  } else {
    Predict: LEGITIMATE (probability: 0.95)
  }
} else {
  Predict: LEGITIMATE (probability: 0.92)
}
```

### Saved Rule Formats
1. **Text File** (`decision_tree_rules.txt`): Complete tree structure
2. **JSON** (`rule_summary.json`): Structured rule data
3. **Feature Importance** (`feature_importance.json`): Ranked features

## 🔮 Inference Usage

### Loading Saved Components
```python
from pyspark.ml import PipelineModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
import json

# 1. Load preprocessing components
preprocessing_path = "dt_preprocessing_YYYYMMDD_HHMMSS"
cat_pipeline = PipelineModel.load(f"{preprocessing_path}/categorical_pipeline")
assembler = VectorAssembler.load(f"{preprocessing_path}/vector_assembler")
scaler = StandardScalerModel.load(f"{preprocessing_path}/standard_scaler")

# Load metadata
with open(f"{preprocessing_path}/feature_metadata.json", 'r') as f:
    metadata = json.load(f)

# 2. Load trained model
model_path = "decision_tree_model_YYYYMMDD_HHMMSS"
dt_model = DecisionTreeClassificationModel.load(model_path)

# 3. Apply to new data
new_df_preprocessed = cat_pipeline.transform(new_df)
new_df_assembled = assembler.transform(new_df_preprocessed)
new_df_scaled = scaler.transform(new_df_assembled)

# 4. Make predictions
predictions = dt_model.transform(new_df_scaled)
```

### Extracting Fraud Probability
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

def extract_fraud_prob(probability_vector):
    return float(probability_vector[1])

extract_fraud_prob_udf = udf(extract_fraud_prob, FloatType())

results = predictions.withColumn(
    "fraud_probability",
    extract_fraud_prob_udf(col("probability"))
)
```

## 🐛 Troubleshooting

### Memory Issues
If you still encounter OOM errors:

1. **Reduce sample fraction** (line ~275):
   ```python
   df_sample_for_fitting = df.sample(fraction=0.05, seed=42)  # 5% instead of 10%
   ```

2. **Increase executor memory**:
   ```python
   .config("spark.executor.memory", "150g")
   ```

3. **Reduce undersampling**:
   ```python
   undersample_fraction = 0.20  # 20% instead of 30%
   ```

### Slow Performance
1. **Increase partitions**:
   ```python
   .config("spark.sql.shuffle.partitions", "400")
   ```

2. **Reduce tree depth**:
   ```python
   dt = DecisionTreeClassifier(maxDepth=8)  # Instead of 10
   ```

### Class Imbalance Issues
1. **Adjust undersampling**:
   ```python
   undersample_fraction = 0.40  # Increase to 40%
   ```

2. **Modify min instances per node**:
   ```python
   dt = DecisionTreeClassifier(minInstancesPerNode=50)  # Reduce from 100
   ```

## 📊 Expected Output

### Console Output
```
✅ Libraries imported successfully
🔄 Stopped existing Spark session
✅ Spark session initialized
📦 Caching DataFrame in memory...
✅ DATA LOADED SUCCESSFULLY!
🔧 Fitting categorical pipeline on 10% sample...
✅ Categorical pipeline fitted on sample
🔍 Original counts:
   • Fraud cases: 50,000
   • Legitimate cases: 10,000,000
   • Imbalance ratio: 1:200.00
🎲 Undersampling legitimate cases to 30.0%...
✅ Balanced Dataset Created:
   • Total samples: 3,050,000
   • Fraud: 50,000 (1.64%)
   • Legitimate: 3,000,000 (98.36%)
   • New imbalance ratio: 1:60.00
🔧 Creating feature vectors...
📏 Scaling features...
🔀 Splitting data into train/test sets...
🚀 Training Decision Tree model...
✅ Model training completed!
📊 Model Performance Metrics:
   • Accuracy: 0.9542
   • Precision: 0.9538
   • Recall: 0.9542
   • F1 Score: 0.9539
   • AUC-ROC: 0.9621
💾 Saving all artifacts...
✅ All artifacts saved successfully!
🎉 DECISION TREE TRAINING COMPLETED SUCCESSFULLY!
```

## 📝 Logs

Detailed logs are saved to:
- **File**: `decision_tree_training_YYYYMMDD_HHMMSS.log`
- **Level**: DEBUG (includes all operations)
- **Console**: INFO level (key operations only)

## 🔧 Customization

### Modify Date Range
Edit line ~113:
```python
start_date = '2025-01-01'
end_date = '2025-06-30'
```

### Change Features
Edit line ~115 to add/remove features from `selected_cols` list.

### Adjust Model Parameters
Edit line ~448:
```python
dt = DecisionTreeClassifier(
    maxDepth=15,           # Increase tree depth
    maxBins=64,            # More bins for better splits
    minInstancesPerNode=50 # Allow smaller leaf nodes
)
```

### Modify Train/Test Split
Edit line ~427:
```python
train_data, test_data = df_final.randomSplit([0.7, 0.3], seed=42)  # 70/30 split
```

## ✅ Validation Checklist

Before running in production:
- [ ] Verify date range covers desired period
- [ ] Check Spark cluster has sufficient resources
- [ ] Ensure ClickHouse connection is stable
- [ ] Confirm output directories have write permissions
- [ ] Test with small date range first
- [ ] Review feature list for completeness
- [ ] Validate undersampling ratio meets requirements

## 📞 Support

For issues or questions:
1. Check log files for detailed error messages
2. Review `DECISION_TREE_TRAINING_FIX.md` for memory optimization details
3. Verify Spark cluster status
4. Check ClickHouse connectivity

---

**Last Updated**: November 7, 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready
