#!/usr/bin/env python3
"""
Fraud Detection Model Inference Script

This script demonstrates how to load all saved preprocessing components
and the trained model to make predictions on new data.

Author: AI Assistant
Date: November 2025
"""

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fraud_inference')


def load_preprocessing_components(preprocessing_path):
    """
    Load all saved preprocessing components for inference
    
    Args:
        preprocessing_path: Path to the directory containing saved components
    
    Returns:
        dict: Dictionary containing all loaded components and metadata
    """
    logger.info("📦 Loading preprocessing components...")
    components = {}
    
    try:
        # Load categorical preprocessing pipeline (StringIndexers + OneHotEncoders)
        cat_pipeline_path = f"{preprocessing_path}/categorical_pipeline"
        components['categorical_pipeline'] = PipelineModel.load(cat_pipeline_path)
        logger.info(f"✅ Categorical pipeline loaded from {cat_pipeline_path}")
        
        # Load vector assembler
        assembler_path = f"{preprocessing_path}/vector_assembler"
        components['vector_assembler'] = VectorAssembler.load(assembler_path)
        logger.info(f"✅ Vector assembler loaded from {assembler_path}")
        
        # Load standard scaler
        scaler_path = f"{preprocessing_path}/standard_scaler"
        components['standard_scaler'] = StandardScalerModel.load(scaler_path)
        logger.info(f"✅ Standard scaler loaded from {scaler_path}")
        
        # Load feature metadata
        metadata_path = f"{preprocessing_path}/feature_metadata.json"
        with open(metadata_path, 'r') as f:
            components['feature_metadata'] = json.load(f)
        logger.info(f"✅ Feature metadata loaded from {metadata_path}")
        
        logger.info("🎯 All preprocessing components loaded successfully!")
        return components
        
    except Exception as e:
        logger.error(f"❌ Error loading preprocessing components: {str(e)}")
        raise


def load_trained_model(model_path):
    """
    Load the trained Logistic Regression model
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        LogisticRegressionModel: Loaded model
    """
    try:
        logger.info(f"🤖 Loading trained model from {model_path}...")
        model = LogisticRegressionModel.load(model_path)
        logger.info("✅ Trained model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise


def preprocess_data(df, components):
    """
    Apply the complete preprocessing pipeline to new data
    
    Args:
        df: Input DataFrame with raw features
        components: Dictionary containing preprocessing components
    
    Returns:
        DataFrame: Preprocessed data ready for prediction
    """
    logger.info("🔧 Applying preprocessing pipeline...")
    
    try:
        # Step 1: Apply categorical preprocessing (StringIndexer + OneHotEncoder)
        logger.info("   • Applying categorical preprocessing...")
        df_cat = components['categorical_pipeline'].transform(df)
        
        # Step 2: Assemble features into vector
        logger.info("   • Assembling feature vector...")
        df_assembled = components['vector_assembler'].transform(df_cat)
        
        # Step 3: Scale features
        logger.info("   • Scaling features...")
        df_scaled = components['standard_scaler'].transform(df_assembled)
        
        # Step 4: Select only required columns for prediction
        df_final = df_scaled.select("features")
        
        logger.info("✅ Preprocessing completed successfully!")
        return df_final
        
    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {str(e)}")
        raise


def make_predictions(df_preprocessed, model):
    """
    Make fraud predictions using the trained model
    
    Args:
        df_preprocessed: Preprocessed DataFrame with feature vectors
        model: Trained LogisticRegressionModel
    
    Returns:
        DataFrame: Predictions with probabilities
    """
    logger.info("🎯 Making fraud predictions...")
    
    try:
        # Make predictions
        predictions = model.transform(df_preprocessed)
        
        # Select relevant columns
        results = predictions.select(
            "features",
            "prediction",
            "probability",
            "rawPrediction"
        )
        
        logger.info("✅ Predictions completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error making predictions: {str(e)}")
        raise


def run_inference_example():
    """
    Complete inference pipeline example
    """
    logger.info("🚀 Starting Fraud Detection Inference Example")
    logger.info("=" * 60)
    
    # Paths to saved components
    base_path = "/root/research-dir/dev/jazzcash-fraud-detection/scripts"
    preprocessing_path = f"{base_path}/preprocessing_components_jun_v3"
    model_path = f"{base_path}/spark_lr_model_june_v3"
    
    # Initialize Spark
    logger.info("⚡ Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("fraud_detection_inference") \
        .master("spark://dfs-ai-app2:7077") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    logger.info("✅ Spark session initialized")
    
    try:
        # Load all preprocessing components
        components = load_preprocessing_components(preprocessing_path)
        
        # Load trained model
        model = load_trained_model(model_path)
        
        # Display model information
        metadata = components['feature_metadata']
        logger.info("📋 Model Information:")
        logger.info(f"   • Training Date: {metadata['training_date']}")
        logger.info(f"   • Number of Features: {metadata['num_features']}")
        logger.info(f"   • Target Column: {metadata['target_column']}")
        logger.info(f"   • String Columns: {len(metadata['string_columns'])}")
        
        logger.info("🎯 Inference pipeline ready!")
        logger.info("📝 To use with new data:")
        logger.info("   1. Load your new data into a Spark DataFrame")
        logger.info("   2. Call preprocess_data(new_df, components)")
        logger.info("   3. Call make_predictions(preprocessed_df, model)")
        logger.info("   4. Analyze the results")
        
        # Example of how to use (commented out since we don't have new data)
        """
        # Load new data (example)
        new_df = spark.read.format("jdbc")...
        
        # Apply preprocessing
        preprocessed_df = preprocess_data(new_df, components)
        
        # Make predictions
        predictions = make_predictions(preprocessed_df, model)
        
        # Show results
        predictions.show(10)
        
        # Get fraud probability distribution
        predictions.groupBy("prediction").count().show()
        """
        
    except Exception as e:
        logger.error(f"❌ Inference example failed: {str(e)}")
        raise
    finally:
        # Clean up
        spark.stop()
        logger.info("🔄 Spark session stopped")
    
    logger.info("✅ Inference example completed successfully!")


if __name__ == "__main__":
    run_inference_example()
