#!/usr/bin/env python3
"""
Fraud Detection Model Inference Script

This script loads the trained fraud detection model and applies it to new test data.
Supports multiple data sources: CSV files, ClickHouse database, or Spark DataFrames.

The script is configured with hard-coded values for:
- Input type: ClickHouse database
- Date range: 2025-07-01 to 2025-07-31  
- Output format: Parquet file (auto-generated path with timestamp)
- Fraud flag inclusion: Enabled for evaluation

Usage:
    python fraud_inference.py

For interactive mode (loads components and returns objects for notebook use):
    Call main() function directly and use the returned engine object.

Author: Generated for JazzCash Fraud Detection Project
Date: November 2025
"""

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
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import VectorUDT, DenseVector, SparseVector


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
        self.model_path = model_path or f"{script_dir}/spark_lr_model_june_v3"
        self.preprocessing_path = preprocessing_path or f"{script_dir}/preprocessing_components_jun_v3"
        
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
        
        # Set Python path for consistent version across driver and workers
        import sys
        python_path = sys.executable
        os.environ['PYSPARK_PYTHON'] = python_path
        os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
        
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
            .config("spark.executor.memory", "100g") \
            .config("spark.executor.memoryOverhead", "5g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.cores", "32") \
            .config("spark.executor.instances", "2") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "96") \
            .config("spark.pyspark.python", python_path) \
            .config("spark.pyspark.driver.python", python_path) \
            .getOrCreate()
            
        self.logger.info("✅ Spark session created for inference")
        self.logger.info(f"🐍 Using Python: {python_path}")
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
    
    def load_data_for_single_date(self, cutoff_date: str, include_fraud_flag: bool = False) -> DataFrame:
        """
        Load test data from ClickHouse database for a single cutoff date
        
        Args:
            cutoff_date: Specific cutoff date (YYYY-MM-DD)
            include_fraud_flag: Whether to include fraud_flag column (for testing)
            
        Returns:
            Spark DataFrame with the loaded data for that date
        """
        self.logger.info(f"📅 Loading data for cutoff_date: {cutoff_date}")
        
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
        
        query = f"""
            SELECT *
            FROM stixor_fraud_features_distributed
            WHERE cutoff_date = '{cutoff_date}'
                AND mbar_account_type_name = 'Customer Account'
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
                .option('fetchsize', '500000')
                .load())
            
            # Get row count
            total_rows = df.count()
            self.logger.info(f"✅ Loaded {total_rows:,} rows for {cutoff_date}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading ClickHouse data for {cutoff_date}: {str(e)}")
            raise
            
    def load_data_from_clickhouse(self, 
                                  start_date: str, 
                                  end_date: str, 
                                  include_fraud_flag: bool = False) -> DataFrame:
        """
        Load test data from ClickHouse database
        
        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
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
            
        # selected_cols = self.feature_metadata['all_feature_cols'].copy()        
        query = f"""
            SELECT *
            FROM stixor_fraud_features_distributed
            WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
                AND mbar_account_type_name = 'Customer Account'
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
                .option('fetchsize', '1000000')  # Fetch 100k rows at a time
                .option("partitionColumn", "cutoff_date") \
                .option('lowerBound', start_date)  # Lower bound of partition column
                .option('upperBound', end_date)    # Upper bound of partition column
                .option('numPartitions', str(30))  # Number of partitions
                .load())
            
            # Cache the DataFrame
            self.logger.info("📦 Caching inference DataFrame...")
            df.cache()            
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
            
            # Select features column AND preserve important ID/label columns
            columns_to_keep = ["features"]
            if "trans_id" in df_scaled.columns:
                columns_to_keep.append("trans_id")
            if "fraud_flag" in df_scaled.columns:
                columns_to_keep.append("fraud_flag")
                
            df_final = df_scaled.select(*columns_to_keep)
            
            self.logger.info("✅ Preprocessing completed successfully")
            return df_final
            
        except Exception as e:
            self.logger.error(f"❌ Error in preprocessing: {str(e)}")
            raise
            
    def make_predictions(self, df_preprocessed: DataFrame) -> DataFrame:
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
            df_preprocessed = self.model.transform(df_preprocessed)
            
            # Define UDF to extract fraud probability (class 1) from probability vector
            def extract_fraud_prob(probability_vector):
                """Extract fraud probability from probability vector"""
                if probability_vector is not None:
                    if isinstance(probability_vector, (DenseVector, SparseVector)):
                        # For binary classification, class 1 is at index 1
                        if len(probability_vector) > 1:
                            return float(probability_vector[1])
                        else:
                            return 0.0
                return 0.0
            
            # Register UDF
            extract_fraud_prob_udf = udf(extract_fraud_prob, DoubleType())
            
            # Extract probability of fraud (class 1)
            predictions = df_preprocessed.withColumn(
                "fraud_probability",
                extract_fraud_prob_udf(F.col("probability"))
            )
            
            # Select desired columns (dynamically based on what's available)
            desired_cols = []
            
            # Always include prediction columns
            prediction_cols = ['prediction', 'fraud_probability']
            
            # Include trans_id if available
            if 'trans_id' in predictions.columns:
                desired_cols.append('trans_id')
                
            # Include fraud_flag if available (for evaluation)
            if 'fraud_flag' in predictions.columns:
                desired_cols.append('fraud_flag')
            
            # Add prediction columns
            desired_cols.extend(prediction_cols)
            
            predictions_final = predictions.select(*desired_cols)
            
            # num_predictions = predictions_final.count()
            # fraud_predictions = predictions_final.filter(col("prediction") == 1.0).count()
            
            # self.logger.info("✅ Predictions completed successfully!")
            # self.logger.info(f"   • Total predictions: {num_predictions:,}")
            # self.logger.info(f"   • Fraud predictions: {fraud_predictions:,}")
            # self.logger.info(f"   • Fraud rate: {(fraud_predictions/num_predictions)*100:.2f}%")
            
            return predictions_final
            
        except Exception as e:
            self.logger.error(f"❌ Error making predictions: {str(e)}")
            raise
            
    def save_predictions(self, 
                        predictions_df: DataFrame, 
                        output_path: str,
                        include_probabilities: bool = True):
        """
        Save predictions to Parquet file
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Path to save the Parquet file
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
            
            # Save directly to Parquet using Spark's native writer
            # Parquet is much more efficient than CSV for large datasets
            save_df.coalesce(1).write \
                .mode("overwrite") \
                .parquet(output_path)
                
            self.logger.info("✅ Predictions saved successfully to Parquet format!")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving predictions: {str(e)}")
            raise
            
    def run_inference(self, 
                     input_type: str,   
                     input_path: str = None,
                     start_date: str = None,
                     end_date: str = None,
                     output_path: str = None,
                     include_fraud_flag: bool = False) -> DataFrame:
        """
        Run complete inference pipeline
        
        Args:
            input_type: Type of input ('csv' or 'clickhouse')
            input_path: Path to input file (for CSV)
            start_date: Start date (for ClickHouse)
            end_date: End date (for ClickHouse)
            output_path: Path to save predictions
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
                raw_df = self.load_data_from_clickhouse(start_date, end_date, include_fraud_flag)
            else:
                raise ValueError(f"Unsupported input_type: {input_type}")
            
            # Preprocess data
            preprocessed_df = self.preprocess_data(raw_df)
            
            # Make predictions
            predictions_df = self.make_predictions(preprocessed_df)
            
            # Save predictions if output path provided
            if output_path:
                self.save_predictions(predictions_df, output_path)
            
            self.logger.info("🎯 Inference pipeline completed successfully!")
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in inference pipeline: {str(e)}")
            raise

    def run_inference_date_wise(self, 
                               start_date: str,
                               end_date: str,
                               output_base_path: str,
                               include_fraud_flag: bool = False) -> None:
        """
        Run inference date-wise and save predictions to separate partitions
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_base_path: Base path for output (will create subdirectories for each date)
            include_fraud_flag: Include fraud_flag for evaluation
        """
        self.logger.info("🗓️ Starting date-wise fraud detection inference...")
        self.logger.info(f"📅 Processing dates from {start_date} to {end_date}")
        
        # Load components if not already loaded
        if not self.model:
            self.load_model_components()
        
        # Generate date range
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_dt
        processed_dates = []
        
        try:
            while current_date <= end_dt:
                cutoff_date = current_date.strftime('%Y-%m-%d')
                self.logger.info(f"\n🔄 Processing cutoff_date: {cutoff_date}")
                
                try:
                    # Load data for this specific date
                    raw_df = self.load_data_for_single_date(cutoff_date, include_fraud_flag)
                    
                    # Skip if no data for this date
                    if raw_df.count() == 0:
                        self.logger.info(f"⚠️ No data found for {cutoff_date}, skipping...")
                        current_date += timedelta(days=1)
                        continue
                    
                    # Preprocess data
                    preprocessed_df = self.preprocess_data(raw_df)
                    
                    # Make predictions
                    predictions_df = self.make_predictions(preprocessed_df)
                    
                    # Create output path for this date
                    date_output_path = f"{output_base_path}/cutoff_date={cutoff_date}"
                    
                    # Save predictions for this date
                    self.save_predictions_partitioned(predictions_df, date_output_path, cutoff_date)
                    
                    processed_dates.append(cutoff_date)
                    self.logger.info(f"✅ Completed processing for {cutoff_date}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing {cutoff_date}: {str(e)}")
                    # Continue with next date instead of failing completely
                    
                current_date += timedelta(days=1)
            
            self.logger.info("\n🎯 Date-wise inference completed successfully!")
            self.logger.info(f"📊 Processed {len(processed_dates)} dates: {', '.join(processed_dates)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in date-wise inference pipeline: {str(e)}")
            raise

    def save_predictions_partitioned(self, 
                                   predictions_df: DataFrame, 
                                   output_path: str,
                                   cutoff_date: str):
        """
        Save predictions for a specific date to a partitioned location
        
        Args:
            predictions_df: DataFrame with predictions for a specific date
            output_path: Path to save the predictions
            cutoff_date: The cutoff date being processed
        """
        self.logger.info(f"💾 Saving predictions for {cutoff_date} to: {output_path}")
        
        try:
            # Add cutoff_date column if not already present for partitioning
            if 'cutoff_date' not in predictions_df.columns:
                predictions_df = predictions_df.withColumn('cutoff_date', F.lit(cutoff_date))
            
            # Save directly to Parquet with partitioning
            predictions_df.coalesce(1).write \
                .mode("overwrite") \
                .parquet(output_path)
                
            row_count = predictions_df.count()
            self.logger.info(f"✅ Saved {row_count:,} predictions for {cutoff_date}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving predictions for {cutoff_date}: {str(e)}")
            raise


def main():
    """Main function with hard-coded configuration"""
    
    # Hard-coded configuration values
    input_type = "clickhouse"
    start_date = "2025-07-01"
    end_date = "2025-07-31"
    include_fraud_flag = True
    interactive = False
    
    # Generate output base path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_path = f"fraud_predictions_datewise_{timestamp}"
    
    # Create inference engine
    engine = FraudInferenceEngine()
    
    if interactive:
        print("🔧 Interactive mode: Loading components...")
        engine.load_model_components()
        print("✅ Components loaded. Use 'engine' object for inference.")
        return engine
    
    # Run date-wise inference with hard-coded values
    engine.run_inference_date_wise(
        start_date=start_date,
        end_date=end_date,
        output_base_path=output_base_path,
        include_fraud_flag=include_fraud_flag
    )
    
    print(f"🎯 Date-wise inference completed! Predictions saved to: {output_base_path}/")
    print(f"📁 Each date is saved in separate partition: cutoff_date=YYYY-MM-DD/")
    return None


if __name__ == "__main__":
    main()