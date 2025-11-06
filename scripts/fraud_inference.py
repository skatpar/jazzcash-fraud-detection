#!/usr/bin/env python3
"""
Fraud Detection Model Inference Script

This script loads the trained fraud detection model and applies it to new test data.
Supports multiple data sources: CSV files, ClickHouse database, or Spark DataFrames.

Usage:
    # From CSV file
    python fraud_inference.py --input_type csv --input_path /path/to/test_data.csv

    # From ClickHouse 
    python fraud_inference.py --input_type clickhouse --start_date 2025-07-01 --end_date 2025-07-31

    # Interactive mode (loads components and returns objects for notebook use)
    python fraud_inference.py --interactive

Author: Generated for JazzCash Fraud Detection Project
Date: November 2025
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when


class FraudInferenceEngine:
    """
    Fraud Detection Inference Engine
    
    Loads trained model and preprocessing components to make predictions on new data.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 preprocessing_path: str = None,
                 spark_session: SparkSession = None):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the saved trained model
            preprocessing_path: Path to the saved preprocessing components
            spark_session: Existing Spark session (optional)
        """
        # Set default paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or f"{script_dir}/spark_lr_model_june_v2"
        self.preprocessing_path = preprocessing_path or f"{script_dir}/preprocessing_components_june"
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize Spark session
        self.spark = spark_session or self._create_spark_session()
        
        # Initialize components (loaded lazily)
        self.model = None
        self.preprocessing_components = None
        self.feature_metadata = None
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"fraud_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(os.path.dirname(__file__), log_filename)
        
        self.logger = logging.getLogger('fraud_inference')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("✅ Fraud Inference Engine initialized")
        self.logger.info(f"📝 Logging to file: {log_path}")
        
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        jar_files = [
            "/root/research-dir/dev/jazzcash-fraud-detection/utils/clickhouse-jdbc-0.9.2-all-dependencies.jar"
        ]
        
        try:
            # Stop existing session if any
            existing_spark = SparkSession.getActiveSession()
            if existing_spark:
                existing_spark.stop()
                self.logger.info("🔄 Stopped existing Spark session")
        except Exception as e:
            self.logger.debug(f"No existing Spark session to stop: {e}")
        
        spark = SparkSession.builder \
            .appName("fraud_inference") \
            .master("spark://dfs-ai-app2:7077") \
            .config("spark.jars", ",".join(jar_files)) \
            .config("spark.executor.memory", "50g") \
            .config("spark.executor.memoryOverhead", "5g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.cores", "16") \
            .config("spark.executor.instances", "2") \
            .config("spark.sql.shuffle.partitions", "100") \
            .config("spark.default.parallelism", "48") \
            .getOrCreate()
            
        self.logger.info("✅ Spark session created for inference")
        return spark
        
    def load_model_components(self):
        """Load the trained model and all preprocessing components"""
        self.logger.info("📦 Loading trained model and preprocessing components...")
        
        try:
            # Load trained model
            self.logger.info(f"   • Loading model from: {self.model_path}")
            self.model = LogisticRegressionModel.load(self.model_path)
            
            # Load preprocessing components
            self.logger.info(f"   • Loading preprocessing components from: {self.preprocessing_path}")
            self.preprocessing_components = self._load_preprocessing_components()
            
            # Load feature metadata
            metadata_path = f"{self.preprocessing_path}/feature_metadata.json"
            with open(metadata_path, 'r') as f:
                self.feature_metadata = json.load(f)
                
            self.logger.info("✅ All components loaded successfully!")
            self.logger.info(f"   • Model: Logistic Regression")
            self.logger.info(f"   • Features: {self.feature_metadata['num_features']}")
            self.logger.info(f"   • Training date: {self.feature_metadata['training_date']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model components: {str(e)}")
            raise
            
    def _load_preprocessing_components(self) -> Dict[str, Any]:
        """Load all preprocessing components"""
        components = {}
        
        # Load categorical preprocessing pipeline
        cat_pipeline_path = f"{self.preprocessing_path}/categorical_pipeline"
        components['categorical_pipeline'] = PipelineModel.load(cat_pipeline_path)
        
        # Load vector assembler
        assembler_path = f"{self.preprocessing_path}/vector_assembler"
        components['vector_assembler'] = VectorAssembler.load(assembler_path)
        
        # Load standard scaler
        scaler_path = f"{self.preprocessing_path}/standard_scaler"
        components['standard_scaler'] = StandardScalerModel.load(scaler_path)
        
        return components
        
    def load_data_from_csv(self, csv_path: str) -> DataFrame:
        """
        Load test data from CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Spark DataFrame with the loaded data
        """
        self.logger.info(f"📁 Loading data from CSV: {csv_path}")
        
        try:
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(csv_path)
                
            row_count = df.count()
            self.logger.info(f"✅ Loaded {row_count:,} rows from CSV")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading CSV data: {str(e)}")
            raise
            
    def load_data_from_clickhouse(self, 
                                  start_date: str, 
                                  end_date: str, 
                                  limit: int = 100000,
                                  include_fraud_flag: bool = False) -> DataFrame:
        """
        Load test data from ClickHouse database
        
        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            limit: Maximum number of rows to load
            include_fraud_flag: Whether to include fraud_flag column (for testing)
            
        Returns:
            Spark DataFrame with the loaded data
        """
        self.logger.info(f"🗄️ Loading data from ClickHouse: {start_date} to {end_date}")
        
        # ClickHouse configuration
        CLICKHOUSE_CONFIG = {
            'host': 'localhost',
            'port': 9000,
            'database': 'public',
            'user': 'default',
            'password': 'DfsTeChB1'
        }
        
        url = f"jdbc:ch://{CLICKHOUSE_CONFIG['host']}:8123/{CLICKHOUSE_CONFIG['database']}"
        user = CLICKHOUSE_CONFIG['user']
        password = CLICKHOUSE_CONFIG['password']
        driver = "com.clickhouse.jdbc.ClickHouseDriver"
        
        # Get selected columns from metadata
        if not self.feature_metadata:
            self.load_model_components()
            
        selected_cols = self.feature_metadata['all_feature_cols'].copy()
        
        # Add fraud_flag if requested (for testing purposes)
        if include_fraud_flag:
            selected_cols.insert(1, 'fraud_flag')
            
        # Add cutoff_date for filtering
        if 'cutoff_date' not in selected_cols:
            selected_cols.insert(0, 'cutoff_date')
        
        query = f"""
            SELECT {', '.join(selected_cols)}
            FROM stixor_fraud_features_distributed
            WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                AND mbar_account_type_name = 'Customer Account'
            LIMIT {limit}
        """
        
        subquery = f"({query}) AS inference_data"
        
        try:
            df = (self.spark.read
                .format('jdbc')
                .option('driver', driver)
                .option('url', url)
                .option('user', user)
                .option('password', password)
                .option('dbtable', subquery)
                .option('fetchsize', '50000')
                .load())
            
            # Cache the DataFrame
            self.logger.info("📦 Caching inference DataFrame...")
            df.cache()
            
            # Trigger action to load data
            total_rows = df.count()
            self.logger.info(f"✅ Loaded {total_rows:,} rows from ClickHouse")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading ClickHouse data: {str(e)}")
            raise
            
    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """
        Apply preprocessing pipeline to the input data
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        self.logger.info("🔧 Applying preprocessing pipeline...")
        
        if not self.preprocessing_components:
            self.load_model_components()
            
        try:
            # Get string columns from metadata
            string_cols = self.feature_metadata['string_columns']
            
            # Replace empty strings with 'UNKNOWN' (same as training)
            self.logger.info("   • Handling missing categorical values...")
            df_clean = df
            for col_name in string_cols:
                if col_name in df.columns:
                    df_clean = df_clean.withColumn(
                        col_name, 
                        when((F.col(col_name) == "") | F.col(col_name).isNull(), "UNKNOWN")
                        .otherwise(F.col(col_name))
                    )
            
            # Apply categorical preprocessing (StringIndexer + OneHotEncoder)
            self.logger.info("   • Applying categorical encoding...")
            df_cat = self.preprocessing_components['categorical_pipeline'].transform(df_clean)
            
            # Apply vector assembler
            self.logger.info("   • Assembling feature vectors...")
            df_assembled = self.preprocessing_components['vector_assembler'].transform(df_cat)
            
            # Apply standard scaler
            self.logger.info("   • Scaling features...")
            df_scaled = self.preprocessing_components['standard_scaler'].transform(df_assembled)
            
            # Select only required columns for prediction
            df_final = df_scaled.select("features")
            
            self.logger.info("✅ Preprocessing completed successfully")
            return df_final
            
        except Exception as e:
            self.logger.error(f"❌ Error in preprocessing: {str(e)}")
            raise
            
    def make_predictions(self, df_preprocessed: DataFrame, original_df: DataFrame = None) -> DataFrame:
        """
        Make fraud predictions on preprocessed data
        
        Args:
            df_preprocessed: Preprocessed DataFrame with features
            original_df: Original DataFrame to include additional columns in output
            
        Returns:
            DataFrame with predictions and probabilities
        """
        self.logger.info("🔮 Making fraud predictions...")
        
        if not self.model:
            self.load_model_components()
            
        try:
            # Make predictions
            predictions = self.model.transform(df_preprocessed)
            
            # Extract probability of fraud (class 1)
            predictions_with_fraud_prob = predictions.withColumn(
                "fraud_probability",
                F.col("probability").getItem(1)
            )
            
            # Select relevant columns
            result_cols = ["prediction", "fraud_probability", "probability", "rawPrediction"]
            predictions_final = predictions_with_fraud_prob.select(*result_cols)
            
            # If original data is provided, add it to the results
            if original_df is not None:
                # Add row numbers to both DataFrames for joining
                from pyspark.sql.window import Window
                from pyspark.sql.functions import row_number
                
                window = Window.orderBy(F.monotonically_increasing_id())
                
                original_with_row = original_df.withColumn("row_id", row_number().over(window))
                predictions_with_row = predictions_final.withColumn("row_id", row_number().over(window))
                
                # Join the DataFrames
                predictions_final = original_with_row.join(
                    predictions_with_row, 
                    on="row_id", 
                    how="inner"
                ).drop("row_id")
            
            num_predictions = predictions_final.count()
            fraud_predictions = predictions_final.filter(col("prediction") == 1.0).count()
            
            self.logger.info("✅ Predictions completed successfully!")
            self.logger.info(f"   • Total predictions: {num_predictions:,}")
            self.logger.info(f"   • Fraud predictions: {fraud_predictions:,}")
            self.logger.info(f"   • Fraud rate: {(fraud_predictions/num_predictions)*100:.2f}%")
            
            return predictions_final
            
        except Exception as e:
            self.logger.error(f"❌ Error making predictions: {str(e)}")
            raise
            
    def save_predictions(self, 
                        predictions_df: DataFrame, 
                        output_path: str,
                        include_probabilities: bool = True):
        """
        Save predictions to CSV file
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Path to save the CSV file
            include_probabilities: Whether to include probability vectors
        """
        self.logger.info(f"💾 Saving predictions to: {output_path}")
        
        try:
            # Select columns to save
            if include_probabilities:
                save_df = predictions_df
            else:
                # Exclude the full probability vector to reduce file size
                cols_to_drop = ["probability", "rawPrediction"] 
                save_df = predictions_df.drop(*[c for c in cols_to_drop if c in predictions_df.columns])
            
            # Convert to Pandas for easier saving (for smaller datasets)
            # For large datasets, use Spark's CSV writer
            if save_df.count() < 1000000:  # Less than 1M rows
                pandas_df = save_df.toPandas()
                pandas_df.to_csv(output_path, index=False)
            else:
                # Use Spark's CSV writer for large datasets
                output_dir = output_path.replace('.csv', '_spark_output')
                save_df.coalesce(1).write \
                    .mode("overwrite") \
                    .option("header", "true") \
                    .csv(output_dir)
                self.logger.info(f"   • Large dataset saved to directory: {output_dir}")
                
            self.logger.info("✅ Predictions saved successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving predictions: {str(e)}")
            raise
            
    def run_inference(self, 
                     input_type: str,
                     input_path: str = None,
                     start_date: str = None,
                     end_date: str = None,
                     output_path: str = None,
                     limit: int = 100000,
                     include_fraud_flag: bool = False) -> DataFrame:
        """
        Run complete inference pipeline
        
        Args:
            input_type: Type of input ('csv' or 'clickhouse')
            input_path: Path to input file (for CSV)
            start_date: Start date (for ClickHouse)
            end_date: End date (for ClickHouse)
            output_path: Path to save predictions
            limit: Maximum rows to process
            include_fraud_flag: Include fraud_flag for evaluation
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info("🚀 Starting fraud detection inference pipeline...")
        
        try:
            # Load components if not already loaded
            if not self.model:
                self.load_model_components()
            
            # Load data based on input type
            if input_type.lower() == 'csv':
                if not input_path:
                    raise ValueError("input_path required for CSV input")
                raw_df = self.load_data_from_csv(input_path)
            elif input_type.lower() == 'clickhouse':
                if not start_date or not end_date:
                    raise ValueError("start_date and end_date required for ClickHouse input")
                raw_df = self.load_data_from_clickhouse(start_date, end_date, limit, include_fraud_flag)
            else:
                raise ValueError(f"Unsupported input_type: {input_type}")
            
            # Preprocess data
            preprocessed_df = self.preprocess_data(raw_df)
            
            # Make predictions
            predictions_df = self.make_predictions(preprocessed_df, raw_df)
            
            # Save predictions if output path provided
            if output_path:
                self.save_predictions(predictions_df, output_path)
            
            self.logger.info("🎯 Inference pipeline completed successfully!")
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in inference pipeline: {str(e)}")
            raise


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Fraud Detection Model Inference')
    
    parser.add_argument('--input_type', type=str, choices=['csv', 'clickhouse'], 
                       required=True, help='Type of input data source')
    parser.add_argument('--input_path', type=str, help='Path to CSV file (for CSV input)')
    parser.add_argument('--start_date', type=str, help='Start date for ClickHouse query (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for ClickHouse query (YYYY-MM-DD)')
    parser.add_argument('--output_path', type=str, help='Path to save predictions CSV')
    parser.add_argument('--limit', type=int, default=100000, help='Maximum rows to process')
    parser.add_argument('--include_fraud_flag', action='store_true', 
                       help='Include fraud_flag for evaluation')
    parser.add_argument('--interactive', action='store_true', 
                       help='Interactive mode (returns objects for notebook use)')
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = FraudInferenceEngine()
    
    if args.interactive:
        print("🔧 Interactive mode: Loading components...")
        engine.load_model_components()
        print("✅ Components loaded. Use 'engine' object for inference.")
        return engine
    
    # Set default output path if not provided
    if not args.output_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_path = f"fraud_predictions_{timestamp}.csv"
    
    # Run inference
    predictions = engine.run_inference(
        input_type=args.input_type,
        input_path=args.input_path,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output_path,
        limit=args.limit,
        include_fraud_flag=args.include_fraud_flag
    )
    
    print(f"🎯 Inference completed! Predictions saved to: {args.output_path}")
    return predictions


if __name__ == "__main__":
    main()