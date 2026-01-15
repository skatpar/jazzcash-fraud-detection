#!/usr/bin/env python3
"""
XGBoost Fraud Detection Inference API
======================================
Flask API for running fraud inference on transactions using XGBoost models from MLflow.

Takes a transaction ID and MLflow run ID as input and returns the fraud probability score.
Loads XGBoost models, scalers, and feature lists from MLflow runs.

Usage:
    python xgboost_inference_api.py [--port PORT] [--host HOST]

API Endpoints:
    GET  /health                    - Health check
    GET  /models                    - List available MLflow runs
    GET  /models/<run_id>/info      - Get info about a specific run
    POST /models/<run_id>/load      - Preload a model into cache
    POST /models/<run_id>/unload    - Unload a model from cache
    POST /predict                   - Predict fraud probability for a transaction
    POST /predict/batch             - Predict fraud for multiple transactions

Example Request:
    POST /predict
    {
        "trans_id": "TXN123456789",
        "run_id": "abc123def456"
    }

Author: AI Team
Date: December 2025
"""

import os
import sys
import json
import pickle
import logging
import tempfile
import argparse
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Fix PySpark Python version mismatch
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession

# MLflow imports
import mlflow
from mlflow.tracking import MlflowClient

# XGBoost
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "clickhouse": {
        "host": "localhost",
        "port": 9000,
        "http_port": 8123,
        "database": "public",
        "user": "default",
        "password": "DfsTeChB1"
    },
    "mlflow": {
        "tracking_uri": "http://localhost:5001",
        "experiment_name": "fraud_detection_pipeline"
    },
    "feature_table": "stixor_fraud_features_distributed",
    # Default features (will be overridden by features.json from MLflow)
    "default_features": [
        'trans_id', 'cutoff_date', 'fraud_flag', 'trx_amt',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
        'user_total_txns_3d', 'user_total_amount_3d', 'user_avg_amount_3d',
        'user_median_amount_3d', 'user_max_amount_3d', 'user_min_amount_3d',
        'user_unique_recipients_3d', 'user_unique_channels_3d', 'user_unique_types_3d',
        'user_total_txns_7d', 'user_total_amount_7d', 'user_avg_amount_7d',
        'user_median_amount_7d', 'user_max_amount_7d', 'user_min_amount_7d',
        'user_unique_recipients_7d', 'user_unique_channels_7d', 'user_unique_types_7d',
        'user_channel_diversity_score_7d', 'user_type_diversity_score_7d',
        'user_night_txns_7d', 'user_weekend_txns_7d',
        'user_peak_hour_txns_7d', 'user_off_peak_hour_txns_7d',
        'user_avg_start_balance_7d', 'user_avg_end_balance_7d',
        'user_min_balance_7d', 'user_max_balance_7d', 'user_balance_volatility_7d',
        'user_avg_amount_per_recipient_7d', 'user_max_amount_to_single_recipient_7d',
        'user_recipient_concentration_ratio_7d', 'user_avg_time_between_txns_7d',
        'user_txn_frequency_score_7d', 'user_days_since_last_txn'
    ]
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('XGBoostInferenceAPI')

# Initialize Flask app
app = Flask(__name__)

# Global variables
spark: Optional[SparkSession] = None
mlflow_client: Optional[MlflowClient] = None

# Model cache structure: {run_id: {'model': XGBClassifier, 'scaler': StandardScaler, 'features': List[str], 'metadata': Dict}}
model_cache: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_spark() -> SparkSession:
    """Initialize Spark session with ClickHouse configuration."""
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
        .appName("xgboost-fraud-inference-api")
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
    """Initialize MLflow client."""
    global mlflow_client
    
    if mlflow_client is not None:
        return mlflow_client
    
    tracking_uri = CONFIG['mlflow']['tracking_uri']
    logger.info(f"Initializing MLflow client: {tracking_uri}")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    
    logger.info("✅ MLflow client initialized")
    return mlflow_client


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def get_available_runs() -> Dict[str, Dict[str, Any]]:
    """
    Get all XGBoost runs from the MLflow experiment.
    
    Returns:
        Dictionary of run_id -> run info
    """
    try:
        client = initialize_mlflow()
        experiment_name = CONFIG['mlflow']['experiment_name']
        
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
            start_time = row.get('start_time')
            
            # Get metrics
            metrics = {
                'auc_roc': row.get('metrics.auc_roc', 0),
                'accuracy': row.get('metrics.accuracy', 0),
                'precision_fraud': row.get('metrics.precision_fraud', 0),
                'recall_fraud': row.get('metrics.recall_fraud', 0),
                'f1_score_fraud': row.get('metrics.f1_score_fraud', 0),
                'false_positive_rate': row.get('metrics.false_positive_rate', 0),
                'false_negative_rate': row.get('metrics.false_negative_rate', 0)
            }
            
            # Get params
            params = {
                'train_start_date': row.get('params.train_start_date', 'Unknown'),
                'train_end_date': row.get('params.train_end_date', 'Unknown'),
                'test_start_date': row.get('params.test_start_date', 'Unknown'),
                'test_end_date': row.get('params.test_end_date', 'Unknown'),
                'num_features': row.get('params.num_features', 'Unknown'),
                'scale_pos_weight': row.get('params.scale_pos_weight', 'Unknown')
            }
            
            available_runs[run_id] = {
                'run_id': run_id,
                'run_name': run_name,
                'status': status,
                'start_time': str(start_time) if start_time else None,
                'metrics': metrics,
                'params': params,
                'is_loaded': run_id in model_cache
            }
        
        return available_runs
        
    except Exception as e:
        logger.error(f"Error fetching MLflow runs: {str(e)}")
        return {}


def load_model_from_mlflow(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Load XGBoost model, scaler, and features from an MLflow run.
    
    Args:
        run_id: The MLflow run ID
        
    Returns:
        Dictionary with model, scaler, features, and metadata, or None if failed
    """
    global model_cache
    
    # Check cache first
    if run_id in model_cache:
        logger.info(f"Using cached model: {run_id}")
        return model_cache[run_id]
    
    try:
        client = initialize_mlflow()
        
        # Get run info
        run = client.get_run(run_id)
        if not run:
            logger.error(f"MLflow run not found: {run_id}")
            return None
        
        logger.info(f"Loading XGBoost model from MLflow run: {run_id}")
        
        # Create temp directory for artifacts
        temp_dir = tempfile.mkdtemp(prefix="xgboost_model_")
        
        try:
            # Load the XGBoost model using mlflow.xgboost
            model_uri = f"runs:/{run_id}/xgboost_model"
            model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"✓ XGBoost model loaded from: {model_uri}")
            
            # Load scaler from artifacts
            scaler = None
            try:
                scaler_path = client.download_artifacts(run_id, "scaler.pkl", dst_path=temp_dir)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info("✓ Scaler loaded from artifacts")
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")
                # Try from artifacts subfolder
                try:
                    scaler_path = client.download_artifacts(run_id, "artifacts/scaler.pkl", dst_path=temp_dir)
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    logger.info("✓ Scaler loaded from artifacts subfolder")
                except Exception as e2:
                    logger.warning(f"Could not load scaler from artifacts: {e2}")
            
            # Load features list
            features = None
            try:
                features_path = client.download_artifacts(run_id, "features.json", dst_path=temp_dir)
                with open(features_path, 'r') as f:
                    features = json.load(f)
                logger.info(f"✓ Features loaded: {len(features)} features")
            except Exception as e:
                logger.warning(f"Could not load features.json: {e}")
                # Try from artifacts subfolder
                try:
                    features_path = client.download_artifacts(run_id, "artifacts/features.json", dst_path=temp_dir)
                    with open(features_path, 'r') as f:
                        features = json.load(f)
                    logger.info(f"✓ Features loaded from artifacts: {len(features)} features")
                except Exception as e2:
                    logger.warning(f"Could not load features from artifacts: {e2}")
                    features = CONFIG['default_features']
                    logger.info("Using default features list")
            
            # Extract run metadata
            run_name = run.data.tags.get('mlflow.runName', 'Unknown')
            metadata = {
                'run_id': run_id,
                'run_name': run_name,
                'loaded_at': datetime.now().isoformat(),
                'params': dict(run.data.params),
                'metrics': dict(run.data.metrics)
            }
            
            # Cache the model and components
            model_cache[run_id] = {
                'model': model,
                'scaler': scaler,
                'features': features,
                'metadata': metadata
            }
            
            logger.info(f"✅ Model cached successfully: {run_id} ({run_name})")
            return model_cache[run_id]
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading MLflow model {run_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_transaction_data(trans_id: str, features: List[str]) -> Optional[pd.DataFrame]:
    """
    Fetch transaction data from ClickHouse by transaction ID.
    
    Args:
        trans_id: The transaction ID
        features: List of feature columns to fetch
        
    Returns:
        DataFrame with transaction data or None if not found
    """
    initialize_spark()
    
    ch = CONFIG['clickhouse']
    
    # Add required columns if not present
    required_cols = ['trans_id', 'fraud_flag']
    all_cols = list(set(features + required_cols))
    cols_str = ', '.join(all_cols)
    
    query = f"""
        SELECT {cols_str}
        FROM clickhouse.{ch['database']}.{CONFIG['feature_table']}
        WHERE trans_id = '{trans_id}'
        LIMIT 1
    """
    
    logger.info(f"Fetching transaction: {trans_id}")
    
    try:
        df_spark = spark.sql(query)
        
        if df_spark.count() == 0:
            logger.warning(f"Transaction not found: {trans_id}")
            return None
        
        df = df_spark.toPandas()
        return df
        
    except Exception as e:
        logger.error(f"Error fetching transaction {trans_id}: {str(e)}")
        return None


def fetch_batch_transaction_data(trans_ids: List[str], features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch multiple transactions from ClickHouse.
    
    Args:
        trans_ids: List of transaction IDs
        features: List of feature columns to fetch
        
    Returns:
        Tuple of (DataFrame with found transactions, List of not found IDs)
    """
    initialize_spark()
    
    ch = CONFIG['clickhouse']
    
    # Add required columns if not present
    required_cols = ['trans_id', 'fraud_flag']
    all_cols = list(set(features + required_cols))
    cols_str = ', '.join(all_cols)
    
    # Build IN clause
    ids_str = "', '".join(trans_ids)
    
    query = f"""
        SELECT {cols_str}
        FROM clickhouse.{ch['database']}.{CONFIG['feature_table']}
        WHERE trans_id IN ('{ids_str}')
    """
    
    logger.info(f"Fetching {len(trans_ids)} transactions")
    
    try:
        df_spark = spark.sql(query)
        df = df_spark.toPandas()
        
        # Find not found IDs
        found_ids = df['trans_id'].tolist()
        not_found = [tid for tid in trans_ids if tid not in found_ids]
        
        return df, not_found
        
    except Exception as e:
        logger.error(f"Error fetching batch transactions: {str(e)}")
        return pd.DataFrame(), trans_ids


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def run_inference(df: pd.DataFrame, model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on transaction data using the loaded model.
    
    Args:
        df: DataFrame with transaction features
        model_data: Dictionary containing model, scaler, and features
        
    Returns:
        Dictionary with prediction results
    """
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    trans_id = df['trans_id'].values[0]
    actual_fraud = df['fraud_flag'].values[0] if 'fraud_flag' in df.columns else None
    
    # Prepare features
    # Filter to only use features that exist in both the model and the data
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        logger.warning(f"Missing features: {missing}")
    
    X = df[available_features].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Apply scaler if available
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"Could not apply scaler: {e}")
    
    # Get prediction and probability
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Extract fraud probability (class 1)
    fraud_probability = float(y_pred_proba[0][1])
    prediction = int(y_pred[0])
    
    return {
        'trans_id': trans_id,
        'fraud_probability': round(fraud_probability, 6),
        'fraud_score_pct': round(fraud_probability * 100, 2),
        'prediction': prediction,
        'predicted_label': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
        'risk_level': get_risk_level(fraud_probability),
        'actual_fraud_flag': int(actual_fraud) if actual_fraud is not None else None
    }


def run_batch_inference(df: pd.DataFrame, model_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run inference on multiple transactions.
    
    Args:
        df: DataFrame with transaction features
        model_data: Dictionary containing model, scaler, and features
        
    Returns:
        List of prediction results
    """
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    trans_ids = df['trans_id'].tolist()
    actual_frauds = df['fraud_flag'].tolist() if 'fraud_flag' in df.columns else [None] * len(df)
    
    # Prepare features
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Apply scaler if available
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"Could not apply scaler: {e}")
    
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Build results
    results = []
    for i in range(len(df)):
        fraud_probability = float(y_pred_proba[i][1])
        prediction = int(y_pred[i])
        
        results.append({
            'trans_id': trans_ids[i],
            'fraud_probability': round(fraud_probability, 6),
            'fraud_score_pct': round(fraud_probability * 100, 2),
            'prediction': prediction,
            'predicted_label': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'risk_level': get_risk_level(fraud_probability),
            'actual_fraud_flag': int(actual_frauds[i]) if actual_frauds[i] is not None else None
        })
    
    return results


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


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'xgboost-fraud-inference-api',
        'timestamp': datetime.now().isoformat(),
        'spark_initialized': spark is not None,
        'mlflow_tracking_uri': CONFIG['mlflow']['tracking_uri'],
        'experiment_name': CONFIG['mlflow']['experiment_name'],
        'loaded_models': list(model_cache.keys()),
        'num_cached_models': len(model_cache)
    })


@app.route('/models', methods=['GET'])
def list_models():
    """
    List all available MLflow runs with XGBoost models.
    
    Response:
        {
            "success": true,
            "data": {
                "runs": [...],
                "total_runs": int,
                "cached_runs": [...]
            }
        }
    """
    try:
        runs = get_available_runs()
        
        return jsonify({
            'success': True,
            'data': {
                'runs': list(runs.values()),
                'total_runs': len(runs),
                'cached_runs': list(model_cache.keys()),
                'mlflow_tracking_uri': CONFIG['mlflow']['tracking_uri'],
                'experiment_name': CONFIG['mlflow']['experiment_name']
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models/<run_id>/info', methods=['GET'])
def model_info(run_id: str):
    """
    Get detailed info about a specific MLflow run.
    
    Args:
        run_id: The MLflow run ID
    """
    try:
        client = initialize_mlflow()
        run = client.get_run(run_id)
        
        if not run:
            return jsonify({
                'success': False,
                'error': f'Run not found: {run_id}'
            }), 404
        
        info = {
            'run_id': run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000.0).isoformat() if run.info.start_time else None,
            'end_time': datetime.fromtimestamp(run.info.end_time / 1000.0).isoformat() if run.info.end_time else None,
            'artifact_uri': run.info.artifact_uri,
            'params': dict(run.data.params),
            'metrics': dict(run.data.metrics),
            'is_loaded': run_id in model_cache
        }
        
        # Add cached model info if loaded
        if run_id in model_cache:
            cached = model_cache[run_id]
            info['cached_info'] = {
                'loaded_at': cached['metadata']['loaded_at'],
                'num_features': len(cached['features']),
                'has_scaler': cached['scaler'] is not None
            }
        
        return jsonify({
            'success': True,
            'data': info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models/<run_id>/load', methods=['POST'])
def load_model(run_id: str):
    """
    Preload a model into cache.
    
    Args:
        run_id: The MLflow run ID
    """
    try:
        model_data = load_model_from_mlflow(run_id)
        
        if model_data is None:
            return jsonify({
                'success': False,
                'error': f'Failed to load model: {run_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'message': f'Model {run_id} loaded successfully',
            'data': {
                'run_id': run_id,
                'run_name': model_data['metadata']['run_name'],
                'num_features': len(model_data['features']),
                'has_scaler': model_data['scaler'] is not None,
                'cached_models': list(model_cache.keys())
            }
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models/<run_id>/unload', methods=['POST'])
def unload_model(run_id: str):
    """
    Unload a model from cache.
    
    Args:
        run_id: The MLflow run ID
    """
    try:
        if run_id in model_cache:
            del model_cache[run_id]
            logger.info(f"Model unloaded: {run_id}")
            return jsonify({
                'success': True,
                'message': f'Model {run_id} unloaded',
                'cached_models': list(model_cache.keys())
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Model not loaded: {run_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud probability for a single transaction.
    
    Request body:
        {
            "trans_id": "string" (required),
            "run_id": "string" (required - MLflow run ID)
        }
    
    Response:
        {
            "success": true,
            "data": {
                "trans_id": "string",
                "fraud_probability": float,
                "fraud_score_pct": float,
                "prediction": int (0 or 1),
                "predicted_label": "FRAUD" or "LEGITIMATE",
                "risk_level": "string",
                "actual_fraud_flag": int or null,
                "model_info": {...}
            }
        }
    """
    try:
        data = request.get_json()
        
        # Validate request
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        if 'trans_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: trans_id'
            }), 400
        
        if 'run_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: run_id'
            }), 400
        
        trans_id = data['trans_id']
        run_id = data['run_id']
        
        # Load model (from cache or MLflow)
        model_data = load_model_from_mlflow(run_id)
        if model_data is None:
            return jsonify({
                'success': False,
                'error': f'Failed to load model: {run_id}'
            }), 404
        
        # Fetch transaction data
        df = fetch_transaction_data(trans_id, model_data['features'])
        if df is None:
            return jsonify({
                'success': False,
                'error': f'Transaction not found: {trans_id}'
            }), 404
        
        # Run inference
        result = run_inference(df, model_data)
        
        # Add model info
        result['model_info'] = {
            'run_id': run_id,
            'run_name': model_data['metadata']['run_name'],
            'auc_roc': model_data['metadata']['metrics'].get('auc_roc', 'N/A'),
            'features': model_data['features'],
            'num_features': len(model_data['features'])
        }
        
        logger.info(f"Prediction: trans_id={trans_id}, probability={result['fraud_probability']}, risk={result['risk_level']}")
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
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
            "trans_ids": ["string", ...] (required),
            "run_id": "string" (required - MLflow run ID)
        }
    
    Response:
        {
            "success": true,
            "data": {
                "predictions": [...],
                "not_found": [...],
                "model_info": {...}
            }
        }
    """
    try:
        data = request.get_json()
        
        # Validate request
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        if 'trans_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: trans_ids'
            }), 400
        
        if 'run_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: run_id'
            }), 400
        
        trans_ids = data['trans_ids']
        run_id = data['run_id']
        
        if not isinstance(trans_ids, list) or len(trans_ids) == 0:
            return jsonify({
                'success': False,
                'error': 'trans_ids must be a non-empty list'
            }), 400
        
        # Limit batch size
        max_batch = 100
        if len(trans_ids) > max_batch:
            return jsonify({
                'success': False,
                'error': f'Batch size exceeds maximum of {max_batch}'
            }), 400
        
        # Load model
        model_data = load_model_from_mlflow(run_id)
        if model_data is None:
            return jsonify({
                'success': False,
                'error': f'Failed to load model: {run_id}'
            }), 404
        
        # Fetch transaction data
        df, not_found = fetch_batch_transaction_data(trans_ids, model_data['features'])
        
        predictions = []
        if len(df) > 0:
            predictions = run_batch_inference(df, model_data)
        
        # Sort predictions by fraud probability (descending)
        predictions = sorted(predictions, key=lambda x: x['fraud_probability'], reverse=True)
        
        logger.info(f"Batch prediction: {len(predictions)} processed, {len(not_found)} not found")
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': predictions,
                'total_processed': len(predictions),
                'not_found': not_found,
                'model_info': {
                    'run_id': run_id,
                    'run_name': model_data['metadata']['run_name'],
                    'auc_roc': model_data['metadata']['metrics'].get('auc_roc', 'N/A'),
                    'features': model_data['features'],
                    'num_features': len(model_data['features'])
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/explain', methods=['POST'])
def predict_with_explanation():
    """
    Predict fraud probability with feature contribution explanation.
    
    Request body:
        {
            "trans_id": "string" (required),
            "run_id": "string" (required)
        }
    
    Response includes top feature contributions to the prediction.
    """
    try:
        data = request.get_json()
        
        if not data or 'trans_id' not in data or 'run_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: trans_id and run_id'
            }), 400
        
        trans_id = data['trans_id']
        run_id = data['run_id']
        
        # Load model
        model_data = load_model_from_mlflow(run_id)
        if model_data is None:
            return jsonify({
                'success': False,
                'error': f'Failed to load model: {run_id}'
            }), 404
        
        # Fetch transaction data
        df = fetch_transaction_data(trans_id, model_data['features'])
        if df is None:
            return jsonify({
                'success': False,
                'error': f'Transaction not found: {trans_id}'
            }), 404
        
        model = model_data['model']
        features = model_data['features']
        scaler = model_data['scaler']
        
        # Prepare features
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].values
        X = np.nan_to_num(X, nan=0.0)
        
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except:
                pass
        
        # Get prediction
        y_pred_proba = model.predict_proba(X)
        fraud_probability = float(y_pred_proba[0][1])
        
        # Get feature importances from the model
        feature_importances = model.feature_importances_
        
        # Get feature values
        feature_values = df[available_features].iloc[0].to_dict()
        
        # Create feature contribution info
        contributions = []
        for i, feat in enumerate(available_features):
            contributions.append({
                'feature': feat,
                'value': float(feature_values.get(feat, 0)),
                'importance': float(feature_importances[i]) if i < len(feature_importances) else 0
            })
        
        # Sort by importance
        contributions = sorted(contributions, key=lambda x: x['importance'], reverse=True)
        
        result = {
            'trans_id': trans_id,
            'fraud_probability': round(fraud_probability, 6),
            'fraud_score_pct': round(fraud_probability * 100, 2),
            'prediction': 1 if fraud_probability >= 0.5 else 0,
            'risk_level': get_risk_level(fraud_probability),
            'actual_fraud_flag': int(df['fraud_flag'].values[0]) if 'fraud_flag' in df.columns else None,
            'top_contributing_features': contributions[:10],
            'model_info': {
                'run_id': run_id,
                'run_name': model_data['metadata']['run_name'],
                'features': model_data['features'],
                'num_features': len(model_data['features'])
            }
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error in prediction with explanation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def init_app():
    """Initialize the application."""
    logger.info("=" * 60)
    logger.info("XGBOOST FRAUD DETECTION INFERENCE API")
    logger.info("=" * 60)
    
    # Initialize Spark
    initialize_spark()
    
    # Initialize MLflow
    try:
        initialize_mlflow()
        logger.info(f"MLflow tracking URI: {CONFIG['mlflow']['tracking_uri']}")
        logger.info(f"MLflow experiment: {CONFIG['mlflow']['experiment_name']}")
        
        # List available runs
        runs = get_available_runs()
        logger.info(f"Available MLflow runs: {len(runs)}")
        for run_id, info in list(runs.items())[:5]:
            logger.info(f"  - {info['run_name']}: {run_id[:8]}... (AUC: {info['metrics'].get('auc_roc', 'N/A'):.4f})")
            
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {str(e)}")
    
    logger.info("=" * 60)
    logger.info("API Ready to serve requests")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='XGBoost Fraud Detection Inference API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the API on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Initialize
    init_app()
    
    # Run Flask
    logger.info(f"Starting API on {args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=False,  # Spark doesn't work well with threading
        use_reloader=False
    )


if __name__ == '__main__':
    main()
