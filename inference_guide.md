# Fraud Detection Model Inference

This directory contains scripts for running fraud detection predictions on new data using the trained model.

## Files

- `fraud_inference.py` - Main inference engine with flexible data loading
- `inference_examples.py` - Usage examples and demonstrations
- `spark_lr_model_june_v2/` - Trained logistic regression model
- `preprocessing_components_june/` - Saved preprocessing components

## Quick Start

### 1. Command Line Usage

#### Predict on CSV file:
```bash
python fraud_inference.py --input_type csv --input_path /path/to/test_data.csv --output_path predictions.csv
```

#### Predict on ClickHouse data:
```bash
python fraud_inference.py --input_type clickhouse --start_date 2025-07-01 --end_date 2025-07-31 --output_path predictions.csv
```

#### Interactive mode (for notebooks):
```bash
python fraud_inference.py --interactive
```

### 2. Python Script Usage

```python
from fraud_inference import FraudInferenceEngine

# Initialize inference engine
engine = FraudInferenceEngine()

# Option 1: CSV file inference
predictions = engine.run_inference(
    input_type='csv',
    input_path='/path/to/test_data.csv',
    output_path='predictions.csv'
)

# Option 2: ClickHouse inference
predictions = engine.run_inference(
    input_type='clickhouse',
    start_date='2025-07-01',
    end_date='2025-07-31',
    output_path='predictions.csv',
    limit=100000
)
```

### 3. Jupyter Notebook Usage

```python
# Load the inference engine
from fraud_inference import FraudInferenceEngine
engine = FraudInferenceEngine()
engine.load_model_components()

# Load your test data
test_data = engine.load_data_from_csv('/path/to/test_data.csv')
# OR
test_data = engine.load_data_from_clickhouse('2025-07-01', '2025-07-31')

# Preprocess and predict
preprocessed_data = engine.preprocess_data(test_data)
predictions = engine.make_predictions(preprocessed_data, test_data)

# View results
predictions.select("prediction", "fraud_probability").show()
```

## Input Data Requirements

Your test data must contain the following columns (same as training data):

### Required Columns:
- `cutoff_date` - Date of the transaction
- `trx_channel` - Transaction channel
- `trx_type` - Transaction type  
- `start_balance` - Account balance before transaction
- `trx_amt` - Transaction amount
- `mbar_registered_channel` - MBAR registration channel
- `mbar_a_c_status` - Account status
- `mbar_a_c_level` - Account level
- `mbar_account_type_name` - Account type name
- `hour_of_day` - Hour of transaction (0-23)
- `day_of_week` - Day of week (0-6)
- `is_weekend` - Weekend flag (0/1)
- `is_night` - Night time flag (0/1)
- `is_business_hours` - Business hours flag (0/1)
- `is_unusual_hour` - Unusual hour flag (0/1)
- `night_weekend_combo` - Night+weekend combination
- `start_balance_log` - Log of start balance
- And many more feature columns... (see feature_metadata.json for complete list)

### CSV File Format:
- Header row required
- Comma-separated values
- Missing values should be empty or NULL
- Categorical columns will be automatically encoded

## Output Format

The predictions will include:
- `prediction` - Binary prediction (0=Legitimate, 1=Fraud)
- `fraud_probability` - Probability of fraud (0.0 to 1.0)
- `probability` - Full probability vector [prob_legit, prob_fraud]
- `rawPrediction` - Raw model scores
- All original columns from input data

## Examples and Testing

Run the examples script to see demonstrations:
```bash
python inference_examples.py
```

This will show examples of:
1. CSV file inference
2. ClickHouse database inference
3. Interactive inference
4. Batch scoring

## Troubleshooting

### Common Issues:

1. **Missing columns error**: Ensure your test data has all required feature columns
2. **Spark connection error**: Check if Spark cluster is running
3. **ClickHouse connection error**: Verify database credentials and connectivity
4. **Memory errors**: Reduce the `limit` parameter for large datasets

### Check Model Components:
```python
# Verify model components are available
import os
model_path = "spark_lr_model_june_v2"
preprocessing_path = "preprocessing_components_june"

print("Model exists:", os.path.exists(model_path))
print("Preprocessing exists:", os.path.exists(preprocessing_path))

# Check feature metadata
import json
with open(f"{preprocessing_path}/feature_metadata.json", 'r') as f:
    metadata = json.load(f)
print("Features:", metadata['num_features'])
print("Training date:", metadata['training_date'])
```

## Performance Tips

1. **For large datasets**: Use ClickHouse input with appropriate `limit`
2. **For CSV files**: Ensure files are not too large (< 1GB recommended)
3. **Batch processing**: Process data in chunks for very large datasets
4. **Caching**: DataFrame caching is enabled automatically for better performance

## Contact

For questions or issues, refer to the main project documentation or training scripts.