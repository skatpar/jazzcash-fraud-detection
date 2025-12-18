#!/usr/bin/env python3
"""
Fraud Detection Inference API
=============================
Flask API for running fraud inference on transactions using trained ML models.

Takes a transaction ID and model ID as input and returns the fraud probability score.

Author: AI Team
Date: December 2025
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

# Configuration
CONFIG = {
    "clickhouse": {
        "host": "localhost",
        "port": 9000,
        "http_port": 8123,
        "database": "public",
        "user": "default",
        "password": "DfsTeChB1"
    },
    "models_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models",
    "default_model": "gbt_pipeline_model_fraud_scenario_v5",
    "feature_table": "stixor_fraud_features_distributed",
    "selected_features": [
        'trans_id', 'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 
        'start_balance', 'trx_amt',
        'hour_of_day', 'is_weekend',
        'txn_txns_3d', 'txn_total_amount_3d', 'txn_avg_amount_3d',
        'txn_max_amount_3d', 'txn_min_amount_3d', 'txn_unique_types_3d', 
        'txn_multi_channel_recent', 'txn_amount_deviation_from_avg'
    ]
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('FraudInferenceAPI')

# Initialize Flask app
app = Flask(__name__)

# Global variables for Spark and model cache
spark: Optional[SparkSession] = None
model_cache: Dict[str, PipelineModel] = {}  # Cache loaded models by model_id


def initialize_spark() -> SparkSession:
    """Initialize Spark session with ClickHouse catalog."""
    global spark
    
    if spark is not None:
        return spark
    
    logger.info("Initializing Spark session...")
    
    packages = [
        "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
        "com.clickhouse:clickhouse-client:0.9.4",
        "com.clickhouse:clickhouse-http-client:0.9.4",
        "org.apache.httpcomponents.client5:httpclient5:5.2.1"
    ]
    
    spark = (SparkSession.builder
        .appName("fraud-inference-api")
        .master("local[*]")
        .config("spark.jars.packages", ",".join(packages))
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "10")
        .getOrCreate()
    )
    
    # Configure ClickHouse catalog
    ch = CONFIG['clickhouse']
    spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
    spark.conf.set("spark.sql.catalog.clickhouse.host", ch['host'])
    spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
    spark.conf.set("spark.sql.catalog.clickhouse.http_port", str(ch['http_port']))
    spark.conf.set("spark.sql.catalog.clickhouse.user", ch['user'])
    spark.conf.set("spark.sql.catalog.clickhouse.password", ch['password'])
    spark.conf.set("spark.sql.catalog.clickhouse.database", ch['database'])
    
    logger.info(f"✅ Spark initialized (Version: {spark.version})")
    
    return spark


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available models in the models directory.
    
    Returns:
        Dictionary of model_id -> model info
    """
    models_dir = Path(CONFIG['models_dir'])
    available_models = {}
    
    for item in models_dir.iterdir():
        if item.is_dir():
            metadata_path = item / "metadata" / "part-00000"
            if metadata_path.exists():
                # This looks like a valid Spark ML model
                available_models[item.name] = {
                    'model_id': item.name,
                    'path': str(item),
                    'has_metadata': True
                }
    
    return available_models


def load_model(model_id: str) -> Optional[PipelineModel]:
    """
    Load a model by its ID. Uses cache if already loaded.
    
    Args:
        model_id: The model identifier (directory name in models folder)
        
    Returns:
        PipelineModel or None if not found
    """
    global model_cache
    
    # Check cache first
    if model_id in model_cache:
        logger.info(f"Using cached model: {model_id}")
        return model_cache[model_id]
    
    # Construct model path
    model_path = os.path.join(CONFIG['models_dir'], model_id)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_id} at {model_path}")
        return None
    
    # Check if it's a valid Spark ML model
    metadata_path = os.path.join(model_path, "metadata", "part-00000")
    if not os.path.exists(metadata_path):
        logger.error(f"Invalid model structure: {model_id} (missing metadata)")
        return None
    
    logger.info(f"Loading model: {model_id} from {model_path}")
    
    try:
        model = PipelineModel.load(model_path)
        model_cache[model_id] = model
        logger.info(f"✅ Model loaded and cached: {model_id}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}")
        return None


def get_transaction_data(transaction_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch transaction data from ClickHouse by transaction ID.
    
    Args:
        transaction_id: The unique transaction identifier
        
    Returns:
        Dictionary with transaction data or None if not found
    """
    ch = CONFIG['clickhouse']
    features = ', '.join(CONFIG['selected_features'])
    
    query = f"""
        SELECT {features}
        FROM clickhouse.{ch['database']}.{CONFIG['feature_table']}
        WHERE trans_id = '{transaction_id}'
        LIMIT 1
    """
    
    logger.info(f"Fetching transaction: {transaction_id}")
    
    df = spark.sql(query)
    
    if df.count() == 0:
        logger.warning(f"Transaction not found: {transaction_id}")
        return None
    
    return df


def run_inference(df, model: PipelineModel) -> Dict[str, Any]:
    """
    Run inference on the transaction data using specified model.
    
    Args:
        df: Spark DataFrame with transaction features
        model: PipelineModel to use for prediction
        
    Returns:
        Dictionary with prediction results
    """
    # Run prediction through the pipeline
    predictions = model.transform(df)
    
    # Extract results
    result = predictions.select(
        'trans_id', 
        'prediction', 
        'probability',
        'fraud_flag'
    ).collect()[0]
    
    # Extract probability for fraud class (class 1)
    fraud_probability = float(result['probability'][1])
    prediction = int(result['prediction'])
    actual_fraud_flag = int(result['fraud_flag']) if result['fraud_flag'] is not None else None
    
    return {
        'transaction_id': result['trans_id'],
        'fraud_probability': round(fraud_probability, 6),
        'prediction': prediction,
        'risk_level': get_risk_level(fraud_probability),
        'actual_fraud_flag': actual_fraud_flag
    }


def get_risk_level(probability: float) -> str:
    """
    Classify fraud probability into risk levels.
    
    Args:
        probability: Fraud probability score (0-1)
        
    Returns:
        Risk level string
    """
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'spark_initialized': spark is not None,
        'models_loaded': list(model_cache.keys()),
        'default_model': CONFIG['default_model']
    })


@app.route('/models', methods=['GET'])
def list_models():
    """
    List all available models.
    
    Response:
        {
            "success": true,
            "data": {
                "available_models": [...],
                "default_model": "string",
                "loaded_models": [...]
            }
        }
    """
    try:
        available = get_available_models()
        
        return jsonify({
            'success': True,
            'data': {
                'available_models': list(available.keys()),
                'default_model': CONFIG['default_model'],
                'loaded_models': list(model_cache.keys()),
                'models_directory': CONFIG['models_dir']
            }
        })
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models/<model_id>/load', methods=['POST'])
def preload_model(model_id: str):
    """
    Preload a model into cache.
    
    Args:
        model_id: The model identifier
        
    Response:
        {
            "success": true,
            "message": "Model loaded successfully"
        }
    """
    try:
        model = load_model(model_id)
        
        if model is None:
            return jsonify({
                'success': False,
                'error': f'Model not found or invalid: {model_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'message': f'Model {model_id} loaded successfully',
            'loaded_models': list(model_cache.keys())
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud probability for a transaction.
    
    Request body:
        {
            "transaction_id": "string",
            "model_id": "string" (optional, defaults to default model)
        }
    
    Response:
        {
            "success": true,
            "data": {
                "transaction_id": "string",
                "fraud_probability": float,
                "prediction": int (0 or 1),
                "risk_level": "string",
                "actual_fraud_flag": int or null,
                "model_used": "string"
            }
        }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'transaction_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: transaction_id'
            }), 400
        
        transaction_id = data['transaction_id']
        model_id = data.get('model_id', CONFIG['default_model'])
        
        # Validate transaction_id
        if not transaction_id or not isinstance(transaction_id, str):
            return jsonify({
                'success': False,
                'error': 'Invalid transaction_id: must be a non-empty string'
            }), 400
        
        # Load model
        model = load_model(model_id)
        if model is None:
            return jsonify({
                'success': False,
                'error': f'Model not found or invalid: {model_id}'
            }), 404
        
        # Fetch transaction data
        df = get_transaction_data(transaction_id)
        
        if df is None:
            return jsonify({
                'success': False,
                'error': f'Transaction not found: {transaction_id}'
            }), 404
        
        # Run inference
        result = run_inference(df, model)
        result['model_used'] = model_id
        
        logger.info(f"Prediction for {transaction_id} using {model_id}: probability={result['fraud_probability']}, risk={result['risk_level']}")
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict fraud probability for multiple transactions.
    
    Request body:
        {
            "transaction_ids": ["string", "string", ...],
            "model_id": "string" (optional, defaults to default model)
        }
    
    Response:
        {
            "success": true,
            "data": [
                {
                    "transaction_id": "string",
                    "fraud_probability": float,
                    "prediction": int,
                    "risk_level": "string",
                    "actual_fraud_flag": int or null
                },
                ...
            ],
            "not_found": ["string", ...],
            "model_used": "string"
        }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'transaction_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: transaction_ids'
            }), 400
        
        transaction_ids = data['transaction_ids']
        model_id = data.get('model_id', CONFIG['default_model'])
        
        if not isinstance(transaction_ids, list) or len(transaction_ids) == 0:
            return jsonify({
                'success': False,
                'error': 'Invalid transaction_ids: must be a non-empty list'
            }), 400
        
        # Limit batch size
        max_batch_size = 100
        if len(transaction_ids) > max_batch_size:
            return jsonify({
                'success': False,
                'error': f'Batch size exceeds maximum of {max_batch_size}'
            }), 400
        
        # Load model
        model = load_model(model_id)
        if model is None:
            return jsonify({
                'success': False,
                'error': f'Model not found or invalid: {model_id}'
            }), 404
        
        # Fetch all transactions in one query
        ch = CONFIG['clickhouse']
        features = ', '.join(CONFIG['selected_features'])
        ids_str = "', '".join(transaction_ids)
        
        query = f"""
            SELECT {features}
            FROM clickhouse.{ch['database']}.{CONFIG['feature_table']}
            WHERE trans_id IN ('{ids_str}')
        """
        
        df = spark.sql(query)
        
        # Get found transaction IDs
        found_ids = [row['trans_id'] for row in df.select('trans_id').collect()]
        not_found = [tid for tid in transaction_ids if tid not in found_ids]
        
        if df.count() == 0:
            return jsonify({
                'success': True,
                'data': [],
                'not_found': not_found,
                'model_used': model_id
            })
        
        # Run batch inference
        predictions = model.transform(df)
        
        # Extract results
        results = []
        for row in predictions.select('trans_id', 'prediction', 'probability', 'fraud_flag').collect():
            fraud_probability = float(row['probability'][1])
            results.append({
                'transaction_id': row['trans_id'],
                'fraud_probability': round(fraud_probability, 6),
                'prediction': int(row['prediction']),
                'risk_level': get_risk_level(fraud_probability),
                'actual_fraud_flag': int(row['fraud_flag']) if row['fraud_flag'] is not None else None
            })
        
        logger.info(f"Batch prediction completed using {model_id}: {len(results)} transactions processed")
        
        return jsonify({
            'success': True,
            'data': results,
            'not_found': not_found,
            'model_used': model_id
        })
        
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models/<model_id>/info', methods=['GET'])
def model_info(model_id: str):
    """
    Get information about a specific model.
    
    Args:
        model_id: The model identifier
    """
    try:
        model = load_model(model_id)
        
        if model is None:
            return jsonify({
                'success': False,
                'error': f'Model not found or invalid: {model_id}'
            }), 404
        
        # Get model info based on type
        model_path = os.path.join(CONFIG['models_dir'], model_id)
        last_stage = model.stages[-1]
        model_type = type(last_stage).__name__
        
        info = {
            'model_id': model_id,
            'model_type': model_type,
            'model_path': model_path,
            'num_stages': len(model.stages),
            'pipeline_stages': [type(stage).__name__ for stage in model.stages]
        }
        
        # Add model-specific info
        if hasattr(last_stage, 'numTrees'):
            info['num_trees'] = last_stage.numTrees
        if hasattr(last_stage, 'numFeatures'):
            info['num_features'] = last_stage.numFeatures
        if hasattr(last_stage, 'depth'):
            info['depth'] = last_stage.depth
        if hasattr(last_stage, 'numClasses'):
            info['num_classes'] = last_stage.numClasses
        
        return jsonify({
            'success': True,
            'data': info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models/<model_id>/unload', methods=['POST'])
def unload_model(model_id: str):
    """
    Unload a model from cache to free memory.
    
    Args:
        model_id: The model identifier
    """
    try:
        if model_id in model_cache:
            del model_cache[model_id]
            logger.info(f"Model unloaded: {model_id}")
            return jsonify({
                'success': True,
                'message': f'Model {model_id} unloaded',
                'loaded_models': list(model_cache.keys())
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Model not loaded: {model_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def init_app():
    """Initialize the application (Spark + default model)."""
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION INFERENCE API")
    logger.info("=" * 60)
    
    initialize_spark()
    
    # Load default model
    default_model = CONFIG['default_model']
    logger.info(f"Loading default model: {default_model}")
    load_model(default_model)
    
    # List available models
    available = get_available_models()
    logger.info(f"Available models: {list(available.keys())}")
    
    logger.info("=" * 60)
    logger.info("API Ready to serve requests")
    logger.info("=" * 60)


if __name__ == '__main__':
    # Initialize Spark and load model on startup
    init_app()
    
    # Run Flask app with threaded=False to avoid Spark threading issues
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=False,  # Important: Spark doesn't work well with threading
        use_reloader=False  # Disable reloader to prevent multiple Spark sessions
    )
