# Fraud Detection Inference API

Flask REST API for running fraud detection inference on transactions using trained ML models. Supports multiple models with dynamic loading.

## Features

- **Single Transaction Prediction**: Get fraud probability for a single transaction
- **Batch Prediction**: Process multiple transactions in one request
- **Multi-Model Support**: Use different models by specifying `model_id`
- **Model Management**: List, load, and unload models dynamically
- **Risk Level Classification**: Automatic risk categorization (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL)
- **Model Information**: Endpoint to inspect loaded model details

## Prerequisites

- Python 3.8+
- PySpark 3.5+
- ClickHouse database with transaction features
- Trained ML pipeline models

## Installation

```bash
# Navigate to the API directory
cd /root/research-dir/dev/jazzcash-fraud-detection/api

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit the `CONFIG` dictionary in `fraud_inference_api.py` to match your environment:

```python
CONFIG = {
    "clickhouse": {
        "host": "localhost",
        "port": 9000,
        "http_port": 8123,
        "database": "public",
        "user": "default",
        "password": "your_password"
    },
    "models_dir": "/path/to/models",
    "default_model": "gbt_pipeline_model_fraud_scenario_v5",
    "feature_table": "stixor_fraud_features_distributed"
}
```

## Running the API

```bash
# Activate the conda environment
conda activate fraud-spark

# Run the API
python fraud_inference_api.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "spark_initialized": true,
    "models_loaded": ["gbt_pipeline_model_fraud_scenario_v5"],
    "default_model": "gbt_pipeline_model_fraud_scenario_v5"
}
```

### List Available Models

```
GET /models
```

**Response:**
```json
{
    "success": true,
    "data": {
        "available_models": [
            "decision_tree_pipeline_model",
            "gbt_pipeline_model_fraud_scenario_v5",
            "random_forest_pipeline_model_v2"
        ],
        "default_model": "gbt_pipeline_model_fraud_scenario_v5",
        "loaded_models": ["gbt_pipeline_model_fraud_scenario_v5"],
        "models_directory": "/path/to/models"
    }
}
```

### Preload a Model

```
POST /models/<model_id>/load
```

**Response:**
```json
{
    "success": true,
    "message": "Model gbt_pipeline_model_fraud_scenario_v6 loaded successfully",
    "loaded_models": ["gbt_pipeline_model_fraud_scenario_v5", "gbt_pipeline_model_fraud_scenario_v6"]
}
```

### Single Transaction Prediction

```
POST /predict
Content-Type: application/json

{
    "transaction_id": "TRX123456789",
    "model_id": "gbt_pipeline_model_fraud_scenario_v5"  // optional
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "transaction_id": "TRX123456789",
        "fraud_probability": 0.847523,
        "prediction": 1,
        "risk_level": "CRITICAL",
        "actual_fraud_flag": 1,
        "model_used": "gbt_pipeline_model_fraud_scenario_v5"
    }
}
```

### Batch Prediction

```
POST /predict/batch
Content-Type: application/json

{
    "transaction_ids": ["TRX123", "TRX456", "TRX789"],
    "model_id": "gbt_pipeline_model_fraud_scenario_v5"  // optional
}
```

**Response:**
```json
{
    "success": true,
    "data": [
        {
            "transaction_id": "TRX123",
            "fraud_probability": 0.123456,
            "prediction": 0,
            "risk_level": "MINIMAL",
            "actual_fraud_flag": 0
        },
        {
            "transaction_id": "TRX456",
            "fraud_probability": 0.756789,
            "prediction": 1,
            "risk_level": "HIGH",
            "actual_fraud_flag": 1
        }
    ],
    "not_found": ["TRX789"],
    "model_used": "gbt_pipeline_model_fraud_scenario_v5"
}
```

### Model Information

```
GET /models/<model_id>/info
```

**Response:**
```json
{
    "success": true,
    "data": {
        "model_id": "gbt_pipeline_model_fraud_scenario_v5",
        "model_type": "GBTClassificationModel",
        "model_path": "/path/to/model",
        "num_stages": 4,
        "num_trees": 200,
        "num_features": 14,
        "pipeline_stages": [
            "StringIndexerModel",
            "StringIndexerModel", 
            "VectorAssembler",
            "GBTClassificationModel"
        ]
    }
}
```

### Unload a Model

```
POST /models/<model_id>/unload
```

**Response:**
```json
{
    "success": true,
    "message": "Model gbt_pipeline_model_fraud_scenario_v5 unloaded",
    "loaded_models": []
}
```

## Risk Level Classification

| Probability Range | Risk Level |
|-------------------|------------|
| 0.80 - 1.00       | CRITICAL   |
| 0.60 - 0.79       | HIGH       |
| 0.40 - 0.59       | MEDIUM     |
| 0.20 - 0.39       | LOW        |
| 0.00 - 0.19       | MINIMAL    |

## Error Handling

All endpoints return consistent error responses:

```json
{
    "success": false,
    "error": "Error message description"
}
```

HTTP Status Codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (transaction or model not found)
- `500`: Internal Server Error

## Example Usage with curl

```bash
# Health check
curl http://localhost:5000/health

# List available models
curl http://localhost:5000/models

# Preload a specific model
curl -X POST http://localhost:5000/models/gbt_pipeline_model_fraud_scenario_v6/load

# Single prediction with default model
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "your_transaction_id"}'

# Single prediction with specific model
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "your_transaction_id", "model_id": "gbt_pipeline_model_fraud_scenario_v6"}'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transaction_ids": ["trx1", "trx2", "trx3"], "model_id": "gbt_pipeline_model_fraud_scenario_v5"}'

# Model info
curl http://localhost:5000/models/gbt_pipeline_model_fraud_scenario_v5/info

# Unload model
curl -X POST http://localhost:5000/models/gbt_pipeline_model_fraud_scenario_v5/unload
```

## Example Usage with Python

```python
import requests

# List available models
response = requests.get('http://localhost:5000/models')
print(response.json())

# Single prediction with specific model
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'transaction_id': 'your_transaction_id',
        'model_id': 'gbt_pipeline_model_fraud_scenario_v5'  # optional
    }
)
result = response.json()
print(f"Fraud probability: {result['data']['fraud_probability']}")
print(f"Risk level: {result['data']['risk_level']}")
print(f"Model used: {result['data']['model_used']}")
```

## Notes

- The API initializes Spark and loads the default model on startup
- Models are cached in memory after first load for faster inference
- Batch predictions are limited to 100 transactions per request
- The `actual_fraud_flag` field returns the ground truth from the database (if available)
- Use `/models/<model_id>/unload` to free memory if needed
