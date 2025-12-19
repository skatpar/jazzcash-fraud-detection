#!/usr/bin/env python3
"""
Fraud Detection Inference API
=============================
Flask API for running fraud inference on transactions using trained ML models.

Takes a transaction ID and model ID as input and returns the fraud probability score.
Supports loading models from local files or MLflow runs.

Author: AI Team
Date: December 2025
"""

import os
import sys
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

# MLflow imports
import mlflow
from mlflow.tracking import MlflowClient

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
    "default_model": "gbt_pipeline_model_fraud_scenario_v2",
    "feature_table": "stixor_fraud_features_distributed",
    "selected_features": [
        'trans_id', 'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 
        'start_balance', 'trx_amt',
        'hour_of_day', 'is_weekend',
        'txn_txns_3d', 'txn_total_amount_3d', 'txn_avg_amount_3d',
        'txn_max_amount_3d', 'txn_min_amount_3d', 'txn_unique_types_3d', 
        'txn_multi_channel_recent', 'txn_amount_deviation_from_avg'
    ],
    "mlflow": {
        "tracking_uri": "http://localhost:5001",
        "experiment_name": "fraud_detection_pipeline"
    }
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
model_cache: Dict[str, PipelineModel] = {}  # Cache loaded models by model_id or run_id
mlflow_model_cache: Dict[str, Dict[str, Any]] = {}  # Cache MLflow model metadata by run_id
mlflow_client: Optional[MlflowClient] = None


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


def initialize_mlflow() -> MlflowClient:
    """Initialize MLflow client for tracking server."""
    global mlflow_client
    
    if mlflow_client is not None:
        return mlflow_client
    
    mlflow_cfg = CONFIG['mlflow']
    tracking_uri = mlflow_cfg['tracking_uri']
    
    logger.info(f"Initializing MLflow client with tracking URI: {tracking_uri}")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    
    logger.info("✅ MLflow client initialized")
    
    return mlflow_client


def get_mlflow_runs() -> Dict[str, Dict[str, Any]]:
    """
    Get all runs from the configured MLflow experiment.
    
    Returns:
        Dictionary of run_id -> run info
    """
    try:
        client = initialize_mlflow()
        mlflow_cfg = CONFIG['mlflow']
        experiment_name = mlflow_cfg['experiment_name']
        
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"MLflow experiment '{experiment_name}' not found")
            return {}
        
        # Search for runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        available_runs = {}
        for _, row in runs.iterrows():
            run_id = row['run_id']
            status = row.get('status', 'UNKNOWN')
            
            # Only include finished runs
            if status != 'FINISHED':
                continue
            
            run_name = row.get('tags.mlflow.runName', 'Unknown')
            model_type = row.get('params.model_type', 'Unknown')
            start_time = row.get('start_time')
            
            # Get metrics
            metrics = {
                'auc_roc': row.get('metrics.auc_roc', 0),
                'accuracy': row.get('metrics.accuracy', 0),
                'precision': row.get('metrics.precision', 0),
                'recall': row.get('metrics.recall', 0),
                'f1_score': row.get('metrics.f1_score', 0)
            }
            
            available_runs[run_id] = {
                'run_id': run_id,
                'run_name': run_name,
                'model_type': model_type,
                'status': status,
                'start_time': str(start_time) if start_time else None,
                'metrics': metrics,
                'source': 'mlflow'
            }
        
        return available_runs
        
    except Exception as e:
        logger.error(f"Error fetching MLflow runs: {str(e)}")
        return {}


def load_model_from_mlflow(run_id: str) -> Optional[PipelineModel]:
    """
    Load a Spark ML model from an MLflow run.
    
    Args:
        run_id: The MLflow run ID
        
    Returns:
        PipelineModel or None if not found
    """
    global model_cache, mlflow_model_cache
    
    # Check cache first (use run_id as key with prefix)
    cache_key = f"mlflow:{run_id}"
    if cache_key in model_cache:
        logger.info(f"Using cached MLflow model: {run_id}")
        return model_cache[cache_key]
    
    try:
        client = initialize_mlflow()
        
        # Get run info
        run = client.get_run(run_id)
        if not run:
            logger.error(f"MLflow run not found: {run_id}")
            return None
        
        logger.info(f"Loading model from MLflow run: {run_id}")
        
        # Download the spark_model artifact
        # The model is stored as a directory artifact
        artifact_path = "spark_model"
        
        # Create a temp directory for downloading
        temp_dir = tempfile.mkdtemp(prefix="mlflow_model_")
        
        try:
            # Download artifacts
            local_model_path = client.download_artifacts(run_id, artifact_path, dst_path=temp_dir)
            
            # The artifact might be downloaded as spark_model/<model_name>
            # Find the actual model directory
            if os.path.isdir(local_model_path):
                # Check if there's a subdirectory (the actual model)
                subdirs = [d for d in os.listdir(local_model_path) if os.path.isdir(os.path.join(local_model_path, d))]
                if subdirs:
                    # Use the first subdirectory as the model path
                    actual_model_path = os.path.join(local_model_path, subdirs[0])
                else:
                    actual_model_path = local_model_path
            else:
                actual_model_path = local_model_path
            
            # Verify it's a valid Spark ML model
            metadata_path = os.path.join(actual_model_path, "metadata", "part-00000")
            if not os.path.exists(metadata_path):
                logger.error(f"Invalid model structure in MLflow run {run_id} (missing metadata)")
                return None
            
            # Load the PipelineModel
            model = PipelineModel.load(actual_model_path)
            
            # Cache the model
            model_cache[cache_key] = model
            
            # Store metadata
            run_name = run.data.tags.get('mlflow.runName', 'Unknown')
            model_type = run.data.params.get('model_type', 'Unknown')
            mlflow_model_cache[run_id] = {
                'run_id': run_id,
                'run_name': run_name,
                'model_type': model_type,
                'loaded_at': datetime.now().isoformat(),
                'local_path': actual_model_path
            }
            
            logger.info(f"✅ MLflow model loaded and cached: {run_id} ({run_name})")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            # Cleanup temp directory on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
            
    except Exception as e:
        logger.error(f"Error loading MLflow model {run_id}: {str(e)}")
        return None


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


def load_model(model_id: str, run_id: str = None) -> Optional[PipelineModel]:
    """
    Load a model by its ID or MLflow run ID. Uses cache if already loaded.
    
    Args:
        model_id: The model identifier (directory name in models folder)
        run_id: Optional MLflow run ID to load model from MLflow instead
        
    Returns:
        PipelineModel or None if not found
    """
    global model_cache
    
    # If run_id is provided, load from MLflow
    if run_id:
        return load_model_from_mlflow(run_id)
    
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
        'mlflow_configured': CONFIG['mlflow']['tracking_uri'],
        'models_loaded': list(model_cache.keys()),
        'mlflow_models_loaded': list(mlflow_model_cache.keys()),
        'default_model': CONFIG['default_model']
    })


@app.route('/models', methods=['GET'])
def list_models():
    """
    List all available models (local and MLflow).
    
    Response:
        {
            "success": true,
            "data": {
                "local_models": [...],
                "mlflow_runs": [...],
                "default_model": "string",
                "loaded_models": [...]
            }
        }
    """
    try:
        # Get local models
        local_models = get_available_models()
        
        # Get MLflow runs
        mlflow_runs = get_mlflow_runs()
        
        return jsonify({
            'success': True,
            'data': {
                'local_models': list(local_models.keys()),
                'mlflow_runs': [
                    {
                        'run_id': info['run_id'],
                        'run_name': info['run_name'],
                        'model_type': info['model_type'],
                        'start_time': info['start_time'],
                        'metrics': info['metrics']
                    }
                    for info in mlflow_runs.values()
                ],
                'default_model': CONFIG['default_model'],
                'loaded_models': list(model_cache.keys()),
                'models_directory': CONFIG['models_dir'],
                'mlflow_tracking_uri': CONFIG['mlflow']['tracking_uri'],
                'mlflow_experiment': CONFIG['mlflow']['experiment_name']
            }
        })
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/mlflow/runs', methods=['GET'])
def list_mlflow_runs():
    """
    List all available MLflow runs.
    
    Response:
        {
            "success": true,
            "data": {
                "runs": [...],
                "tracking_uri": "string",
                "experiment_name": "string"
            }
        }
    """
    try:
        mlflow_runs = get_mlflow_runs()
        
        return jsonify({
            'success': True,
            'data': {
                'runs': [
                    {
                        'run_id': info['run_id'],
                        'run_name': info['run_name'],
                        'model_type': info['model_type'],
                        'start_time': info['start_time'],
                        'metrics': info['metrics']
                    }
                    for info in mlflow_runs.values()
                ],
                'tracking_uri': CONFIG['mlflow']['tracking_uri'],
                'experiment_name': CONFIG['mlflow']['experiment_name'],
                'loaded_runs': list(mlflow_model_cache.keys())
            }
        })
    except Exception as e:
        logger.error(f"Error listing MLflow runs: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/mlflow/runs/<run_id>/load', methods=['POST'])
def preload_mlflow_model(run_id: str):
    """
    Preload an MLflow model into cache.
    
    Args:
        run_id: The MLflow run ID
        
    Response:
        {
            "success": true,
            "message": "Model loaded successfully"
        }
    """
    try:
        model = load_model_from_mlflow(run_id)
        
        if model is None:
            return jsonify({
                'success': False,
                'error': f'MLflow model not found or invalid: {run_id}'
            }), 404
        
        run_info = mlflow_model_cache.get(run_id, {})
        
        return jsonify({
            'success': True,
            'message': f'MLflow model {run_id} loaded successfully',
            'run_info': {
                'run_id': run_id,
                'run_name': run_info.get('run_name', 'Unknown'),
                'model_type': run_info.get('model_type', 'Unknown')
            },
            'loaded_models': list(model_cache.keys())
        })
        
    except Exception as e:
        logger.error(f"Error loading MLflow model: {str(e)}", exc_info=True)
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
            "model_id": "string" (optional, defaults to default model),
            "run_id": "string" (optional, MLflow run ID to use instead of model_id)
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
                "model_used": "string",
                "model_source": "local" or "mlflow"
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
        run_id = data.get('run_id')  # MLflow run ID (optional)
        
        # Validate transaction_id
        if not transaction_id or not isinstance(transaction_id, str):
            return jsonify({
                'success': False,
                'error': 'Invalid transaction_id: must be a non-empty string'
            }), 400
        
        # Determine model source and load model
        model_source = "mlflow" if run_id else "local"
        model_used = run_id if run_id else model_id
        
        # Load model (from MLflow if run_id provided, else from local)
        model = load_model(model_id, run_id=run_id)
        if model is None:
            error_msg = f'MLflow model not found: {run_id}' if run_id else f'Model not found or invalid: {model_id}'
            return jsonify({
                'success': False,
                'error': error_msg
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
        result['model_used'] = model_used
        result['model_source'] = model_source
        
        # Add MLflow run info if applicable
        if run_id and run_id in mlflow_model_cache:
            run_info = mlflow_model_cache[run_id]
            result['run_name'] = run_info.get('run_name', 'Unknown')
            result['model_type'] = run_info.get('model_type', 'Unknown')
        
        logger.info(f"Prediction for {transaction_id} using {model_used} ({model_source}): probability={result['fraud_probability']}, risk={result['risk_level']}")
        
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
            "model_id": "string" (optional, defaults to default model),
            "run_id": "string" (optional, MLflow run ID to use instead of model_id)
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
            "model_used": "string",
            "model_source": "local" or "mlflow"
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
        run_id = data.get('run_id')  # MLflow run ID (optional)
        
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
        
        # Determine model source and load model
        model_source = "mlflow" if run_id else "local"
        model_used = run_id if run_id else model_id
        
        # Load model (from MLflow if run_id provided, else from local)
        model = load_model(model_id, run_id=run_id)
        if model is None:
            error_msg = f'MLflow model not found: {run_id}' if run_id else f'Model not found or invalid: {model_id}'
            return jsonify({
                'success': False,
                'error': error_msg
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
                'model_used': model_used,
                'model_source': model_source
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
        
        logger.info(f"Batch prediction completed using {model_used} ({model_source}): {len(results)} transactions processed")
        
        response_data = {
            'success': True,
            'data': results,
            'not_found': not_found,
            'model_used': model_used,
            'model_source': model_source
        }
        
        # Add MLflow run info if applicable
        if run_id and run_id in mlflow_model_cache:
            run_info = mlflow_model_cache[run_id]
            response_data['run_name'] = run_info.get('run_name', 'Unknown')
            response_data['model_type'] = run_info.get('model_type', 'Unknown')
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/mlflow/runs/<run_id>/info', methods=['GET'])
def mlflow_run_info(run_id: str):
    """
    Get information about a specific MLflow run.
    
    Args:
        run_id: The MLflow run ID
    """
    try:
        client = initialize_mlflow()
        run = client.get_run(run_id)
        
        if not run:
            return jsonify({
                'success': False,
                'error': f'MLflow run not found: {run_id}'
            }), 404
        
        # Extract run info
        info = {
            'run_id': run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000.0).isoformat() if run.info.start_time else None,
            'end_time': datetime.fromtimestamp(run.info.end_time / 1000.0).isoformat() if run.info.end_time else None,
            'artifact_uri': run.info.artifact_uri,
            'params': dict(run.data.params),
            'metrics': dict(run.data.metrics),
            'is_loaded': f"mlflow:{run_id}" in model_cache
        }
        
        return jsonify({
            'success': True,
            'data': info
        })
        
    except Exception as e:
        logger.error(f"Error getting MLflow run info: {str(e)}", exc_info=True)
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


@app.route('/mlflow/runs/<run_id>/unload', methods=['POST'])
def unload_mlflow_model(run_id: str):
    """
    Unload an MLflow model from cache to free memory.
    
    Args:
        run_id: The MLflow run ID
    """
    try:
        cache_key = f"mlflow:{run_id}"
        
        if cache_key in model_cache:
            del model_cache[cache_key]
            if run_id in mlflow_model_cache:
                del mlflow_model_cache[run_id]
            logger.info(f"MLflow model unloaded: {run_id}")
            return jsonify({
                'success': True,
                'message': f'MLflow model {run_id} unloaded',
                'loaded_models': list(model_cache.keys())
            })
        else:
            return jsonify({
                'success': False,
                'error': f'MLflow model not loaded: {run_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error unloading MLflow model: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def init_app():
    """Initialize the application (Spark + MLflow + default model)."""
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION INFERENCE API")
    logger.info("=" * 60)
    
    initialize_spark()
    
    # Initialize MLflow client
    try:
        initialize_mlflow()
        logger.info(f"MLflow tracking URI: {CONFIG['mlflow']['tracking_uri']}")
        logger.info(f"MLflow experiment: {CONFIG['mlflow']['experiment_name']}")
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {str(e)}")
        logger.warning("MLflow model loading will not be available")
    
    # Load default model
    default_model = CONFIG['default_model']
    logger.info(f"Loading default model: {default_model}")
    load_model(default_model)
    
    # List available local models
    available = get_available_models()
    logger.info(f"Available local models: {list(available.keys())}")
    
    # List available MLflow runs
    try:
        mlflow_runs = get_mlflow_runs()
        logger.info(f"Available MLflow runs: {len(mlflow_runs)}")
        for run_id, info in list(mlflow_runs.items())[:5]:  # Show first 5
            logger.info(f"  - {info['run_name']} ({info['model_type']}): {run_id[:8]}...")
    except Exception as e:
        logger.warning(f"Could not fetch MLflow runs: {str(e)}")
    
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
