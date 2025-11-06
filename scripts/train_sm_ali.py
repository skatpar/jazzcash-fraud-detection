# %% [markdown]
# # Spark Logistic Regression Model Training Script
# 
# This notebook provides a clean implementation for loading features from a table and training a Logistic Regression model using Apache Spark MLlib.
# 
# ## Overview
# - Load data from Spark table/DataFrame
# - Prepare features for machine learning
# - Train Logistic Regression model
# - Evaluate model performance
# - Save trained model

# %%
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from clickhouse_driver import Client
import configparser
from pathlib import Path
import logging
import os
from datetime import datetime
import json


def load_preprocessing_components(preprocessing_path, spark):
    """
    Load all saved preprocessing components for inference
    
    Args:
        preprocessing_path: Path to the directory containing saved components
        spark: SparkSession object
    
    Returns:
        dict: Dictionary containing all loaded components and metadata
    """
    components = {}
    
    # Load categorical preprocessing pipeline
    cat_pipeline_path = f"{preprocessing_path}/categorical_pipeline"
    components['categorical_pipeline'] = PipelineModel.load(cat_pipeline_path)
    
    # Load vector assembler
    assembler_path = f"{preprocessing_path}/vector_assembler"
    components['vector_assembler'] = VectorAssembler.load(assembler_path)
    
    # Load standard scaler
    scaler_path = f"{preprocessing_path}/standard_scaler"
    components['standard_scaler'] = StandardScalerModel.load(scaler_path)
    
    # Load feature metadata
    metadata_path = f"{preprocessing_path}/feature_metadata.json"
    with open(metadata_path, 'r') as f:
        components['feature_metadata'] = json.load(f)
    
    return components


def load_trained_model(model_path):
    """
    Load the trained Logistic Regression model
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        LogisticRegressionModel: Loaded model
    """
    return LogisticRegressionModel.load(model_path)

# Configure logging
log_filename = f"spark_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(os.path.dirname(__file__), log_filename)

# Create logger
logger = logging.getLogger('spark_training')
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("✅ Libraries imported successfully")
logger.info(f"📝 Logging to file: {log_path}")

# %%
jar_files = [
    "/root/research-dir/dev/jazzcash-fraud-detection/utils/clickhouse-jdbc-0.9.2-all-dependencies.jar"
]
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

try:
    # Try to stop existing Spark session if it exists
    from pyspark.sql import SparkSession
    existing_spark = SparkSession.getActiveSession()
    if existing_spark:
        existing_spark.stop()
        logger.info("🔄 Stopped existing Spark session")
except Exception as e:
    logger.info(f"Note: No existing Spark session to stop: {e}")
    pass

spark = SparkSession.builder \
    .appName("data_loading") \
    .master("spark://dfs-ai-app2:7077") \
    .config("spark.jars", ",".join(jar_files)) \
    .config("spark.executor.memory", "100g") \
    .config("spark.executor.memoryOverhead", "5g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "32") \
    .config("spark.executor.instances", "2") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "96") \
    .getOrCreate()


# %% [markdown]
# ## 1. Load Features from ClickHouse
# 
# Load the fraud detection features from the ClickHouse `stixor_fraud_features_distributed` table using the context from the fraud dataset profiling notebook.

# %%
import pyspark.sql.functions as F
from datetime import datetime

start_date = '2025-01-01'
end_date = '2025-06-30'
num_partitions = 30

selected_cols = [
'cutoff_date',
'fraud_flag',
 'trx_channel',
 'trx_type',
 'start_balance',
 'trx_amt',
 'mbar_registered_channel',
 'mbar_a_c_status',
 'mbar_a_c_level',
 'mbar_account_type_name',
 'hour_of_day',
 'day_of_week',
 'is_weekend',
 'is_night',
 'is_business_hours',
 'is_unusual_hour',
 'night_weekend_combo',
 'start_balance_log',
 'txn_txns_3d',
 'txn_total_amount_3d',
 'txn_avg_amount_3d',
 'txn_max_amount_3d',
 'txn_min_amount_3d',
 'txn_unique_recipients_3d',
 'txn_unique_channels_3d',
 'txn_unique_types_3d',
 'txn_is_high_activity_3d',
 'txn_multi_channel_recent',
 'txn_amount_deviation_from_avg',
 'txn_night_txns_3d',
 'txn_weekend_txns_3d',
 'channel_new_jc_app',
 'channel_ussd',
 'channel_ussd_api',
 'channel_payment_gateway',
 'channel_mobile_app',
 'type_transfer_c2c',
 'type_transfer_c2b',
 'type_bill_payment',
 'type_mobile_load',
 'user_total_txns_3d',
 'user_total_amount_3d',
 'user_avg_amount_3d',
 'user_median_amount_3d',
 'user_max_amount_3d',
 'user_min_amount_3d',
 'user_unique_recipients_3d',
 'user_unique_channels_3d',
 'user_unique_types_3d',
 'user_total_txns_7d',
 'user_total_amount_7d',
 'user_avg_amount_7d',
 'user_median_amount_7d',
 'user_max_amount_7d',
 'user_min_amount_7d',
 'user_unique_recipients_7d',
 'user_unique_channels_7d',
 'user_unique_types_7d',
 'user_most_used_channel_7d',
 'user_last_used_channel',
 'user_channel_diversity_score_7d',
 'user_most_used_type_7d',
 'user_last_used_type',
 'user_type_diversity_score_7d',
 'user_night_txns_7d',
 'user_weekend_txns_7d',
 'user_peak_hour_txns_7d',
 'user_off_peak_hour_txns_7d',
 'user_avg_start_balance_7d',
 'user_avg_end_balance_7d',
 'user_min_balance_7d',
 'user_max_balance_7d',
 'user_balance_volatility_7d',
 'user_avg_amount_per_recipient_7d',
 'user_max_amount_to_single_recipient_7d',
 'user_recipient_concentration_ratio_7d',
 'user_avg_time_between_txns_7d',
 'user_txn_frequency_score_7d',
 'user_days_since_last_txn']


query = f"""
    SELECT {', '.join(selected_cols)}
    FROM stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
        AND mbar_account_type_name = 'Customer Account'
"""

subquery = f"""
(
    {query}
) AS fraud_data
"""

start_time = datetime.now()
try:
    df = (spark.read
        .format('jdbc')
        .option('driver', driver)
        .option('url', url)
        .option('user', user)
        .option('password', password)
        .option('dbtable', subquery)
        .option('fetchsize', '100000')  # Fetch 100k rows at a time
        .option("partitionColumn", "cutoff_date") \
        .option('lowerBound', start_date)  # Lower bound of partition column
        .option('upperBound', end_date)    # Upper bound of partition column
        .option('numPartitions', str(num_partitions))  # Number of partitions
        .load())
    
    # Cache the DataFrame for better performance
    logger.info("📦 Caching DataFrame in memory...")
    df.cache()
    
    # Trigger action to load data
    logger.info("⏳ Counting rows (this triggers data loading)...")
    # total_rows = df.count()
    num_partitions = df.rdd.getNumPartitions()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("✅ DATA LOADED SUCCESSFULLY!")
    
except Exception as e:
    logger.error("❌ ERROR LOADING DATA!")
    logger.error(f"Error: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise


# %% [markdown]
# ## 2 Feature Assembling and Scaling

# %%
# Convert string (categorical) features to numerical form using StringIndexer and OneHotEncoder
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Feature analysis based on fraud dataset profiling insights
logger.info("🔍 Analyzing feature characteristics...")

TARGET_COLUMN='fraud_flag'
# Check target variable distribution
logger.info(f"📈 Target Variable ({TARGET_COLUMN}) Distribution:")
df.groupBy(TARGET_COLUMN).count().show()

# Get feature columns (exclude non-predictive columns from fraud analysis)
excluded_columns = [
    TARGET_COLUMN,  # Target variable
    'processed_date',  # Time identifier
    'ac_from',  # Account identifiers
    'ac_to', 
    'msisdn_from',
    'msisdn_to',
    'transaction_uuid'  # Transaction identifiers
]

# Get all numerical feature columns based on fraud profiling analysis
all_feature_cols = [col_name for col_name in df.columns if col_name not in excluded_columns]

# Identify string (categorical) columns in all_feature_cols
string_cols = []
for field in df.schema.fields:
    if field.name in all_feature_cols and field.dataType.typeName() == 'string':
        string_cols.append(field.name)

logger.info(f"🔤 String (categorical) features to encode: {string_cols}")

# Replace empty strings in string columns with 'UNKNOWN'
from pyspark.sql.functions import when
for col_name in string_cols:
    df = df.withColumn(col_name, when((F.col(col_name) == "") | F.col(col_name).isNull(), "UNKNOWN").otherwise(F.col(col_name)))

# Index and encode string columns
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in string_cols
]
encoders = [
    OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_ohe", handleInvalid="keep") for col in string_cols
]

# Apply indexers and encoders sequentially
from pyspark.ml import Pipeline

cat_pipeline = Pipeline(stages=indexers + encoders)
df_cat = cat_pipeline.fit(df).transform(df)

# Replace original string columns in all_feature_cols with their OHE columns
modeling_features = [
    f"{col}_ohe" if col in string_cols else col for col in all_feature_cols
]

# Remove any columns that are not present in df_cat (e.g., if OHE dropped some columns)
modeling_features = [col for col in modeling_features if col in df_cat.columns]

logger.info(f"🧮 Final modeling features (after encoding): {modeling_features}")

# Use df_cat as the cleaned DataFrame for further processing
df_clean = df_cat

# %%
modeling_features = modeling_features[1:]

# %% [markdown]
# ## 3. Feature Engineering & Preparation
# 
# Prepare features for machine learning by creating feature vectors and splitting the data.

# %%
# Create feature vectors using selected features
logger.info("🔧 Creating feature vectors for Spark MLlib...")

# Create feature vector using VectorAssembler
assembler = VectorAssembler(
    inputCols=modeling_features,
    outputCol="raw_features",
    handleInvalid="skip"  # Skip rows with invalid values
)

# Apply the assembler
logger.info(f"   • Assembling {len(modeling_features)} features into vector...")
df_assembled = assembler.transform(df_clean)
logger.info("✅ Feature vector created")

# Scale features using StandardScaler (important for logistic regression)
logger.info("📏 Scaling features...")
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,  # Scale to unit variance
    withMean=True  # Center the data
)

# Fit and transform the scaler
logger.info("   • Fitting scaler on training data...")
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)
logger.info("✅ Features scaled successfully")

# Prepare final dataset for ML training
df_final = df_scaled.select("features", col(TARGET_COLUMN).alias("label"))

logger.info("✅ Final dataset prepared:")
logger.info(f"   • Features: Vector of {len(modeling_features)} elements")
logger.info(f"   • Label: Binary (0=Legitimate, 1=Fraud)")
logger.info(f"   • Records: {df_final.count():,}")

# Show sample of the prepared data
logger.info("📋 Sample of Prepared Data:")
df_final.show(3, truncate=False)
# Configure Logistic Regression with optimal parameters for fraud detection
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label", 
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=100,  # Sufficient iterations for convergence
    regParam=0.01,  # L2 regularization to prevent overfitting
    elasticNetParam=0.0,  # Pure L2 regularization (Ridge)
    threshold=0.5,  # Default threshold (will optimize later)
    standardization=False,  # Features already standardized
    aggregationDepth=2,  # For better performance on large datasets
    family="binomial"  # Binary classification
)

logger.info("✅ Logistic Regression configured with fraud detection optimizations:")
logger.info(f"   • Algorithm: Binomial Logistic Regression")
logger.info(f"   • Max Iterations: {lr.getMaxIter()}")
logger.info(f"   • Regularization (L2): {lr.getRegParam()}")
logger.info(f"   • Decision Threshold: {lr.getThreshold()}")
logger.info(f"   • Feature Standardization: Disabled (pre-scaled)")
logger.info(f"   • Optimization: LBFGS (default, good for medium datasets)")

logger.info(f"🎯 Ready for training on {len(modeling_features)} engineered features from fraud profiling analysis")

# %%
# Train the Logistic Regression model
logger.info("🚀 Training Logistic Regression model...")
logger.info("⏱️  This may take a few minutes depending on dataset size...")

import time
start_time = time.time()

# Fit the model on training data
lr_model = lr.fit(df_final)

training_time = time.time() - start_time

logger.info("✅ Model training completed!")
logger.info(f"   • Training time: {training_time:.2f} seconds")
logger.info(f"   • Converged: {lr_model.summary.totalIterations < lr.getMaxIter()}")
logger.info(f"   • Iterations used: {lr_model.summary.totalIterations}/{lr.getMaxIter()}")

# Display training summary
logger.info("📊 Training Summary:")
logger.info(f"   • Objective history length: {len(lr_model.summary.objectiveHistory)}")
logger.info(f"   • Final objective value: {lr_model.summary.objectiveHistory[-1]:.6f}")

# Get coefficient summary
logger.info(f"   • Model coefficients: {len(lr_model.coefficients)} features")
logger.info(f"   • Intercept: {lr_model.intercept:.6f}")

logger.info("🎯 Model successfully trained on fraud detection features from ClickHouse!")

# Save model and all preprocessing components
base_path = "/root/research-dir/dev/jazzcash-fraud-detection/scripts"
model_path = f"{base_path}/spark_lr_model_june_v4"
preprocessing_path = f"{base_path}/preprocessing_components_jun_v4"

# Create directory for preprocessing components if it doesn't exist
import os
os.makedirs(preprocessing_path, exist_ok=True)

# 1. Save the trained model
lr_model.write().overwrite().save(model_path)
logger.info(f"✅ Model saved to {model_path}")

# 2. Save StringIndexers and OneHotEncoders (categorical preprocessing pipeline)
cat_pipeline_fitted = cat_pipeline.fit(df)
cat_pipeline_path = f"{preprocessing_path}/categorical_pipeline"
cat_pipeline_fitted.write().overwrite().save(cat_pipeline_path)
logger.info(f"✅ Categorical preprocessing pipeline (StringIndexers + OneHotEncoders) saved to {cat_pipeline_path}")

# 3. Save VectorAssembler
assembler_path = f"{preprocessing_path}/vector_assembler"
assembler.write().overwrite().save(assembler_path)
logger.info(f"✅ VectorAssembler saved to {assembler_path}")

# 4. Save StandardScaler
scaler_path = f"{preprocessing_path}/standard_scaler"
scaler_model.write().overwrite().save(scaler_path)
logger.info(f"✅ StandardScaler saved to {scaler_path}")

# 5. Save feature column names and metadata for inference
import json
feature_metadata = {
    "string_columns": string_cols,
    "modeling_features": modeling_features,
    "all_feature_cols": all_feature_cols,
    "target_column": TARGET_COLUMN,
    "excluded_columns": excluded_columns,
    "num_features": len(modeling_features),
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "feature_vector_size": len(modeling_features)
}

metadata_path = f"{preprocessing_path}/feature_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(feature_metadata, f, indent=2)
logger.info(f"✅ Feature metadata saved to {metadata_path}")

logger.info("🎯 All preprocessing components saved successfully!")
logger.info("📦 Saved components:")
logger.info(f"   • Trained Model: {model_path}")
logger.info(f"   • Categorical Pipeline: {cat_pipeline_path}")
logger.info(f"   • Vector Assembler: {assembler_path}")
logger.info(f"   • Standard Scaler: {scaler_path}")
logger.info(f"   • Feature Metadata: {metadata_path}")
logger.info(f"   • Total components: 5")

# %% [markdown]
# ## Example: How to Load and Use Saved Components for Inference
# 
# Here's a complete example of how to load all the saved components and use them for inference on new data:

# %%
"""
INFERENCE EXAMPLE - How to use saved components:

# 1. Initialize Spark
spark = SparkSession.builder.appName("fraud_inference").getOrCreate()

# 2. Load all preprocessing components
preprocessing_path = "/root/research-dir/dev/jazzcash-fraud-detection/scripts/preprocessing_components_june"
components = load_preprocessing_components(preprocessing_path, spark)

# 3. Load trained model  
model_path = "/root/research-dir/dev/jazzcash-fraud-detection/scripts/spark_lr_model_june"
trained_model = load_trained_model(model_path)

# 4. Load new data (same structure as training data)
# new_df = spark.read.format("jdbc")...

# 5. Apply same preprocessing pipeline
# Step 5a: Apply categorical preprocessing (StringIndexer + OneHotEncoder)
new_df_cat = components['categorical_pipeline'].transform(new_df)

# Step 5b: Assemble features into vector
new_df_assembled = components['vector_assembler'].transform(new_df_cat)

# Step 5c: Scale features  
new_df_scaled = components['standard_scaler'].transform(new_df_assembled)

# Step 5d: Select only required columns for prediction
new_df_final = new_df_scaled.select("features")

# 6. Make predictions
predictions = trained_model.transform(new_df_final)

# 7. Get results with probabilities
results = predictions.select("features", "prediction", "probability", "rawPrediction")
results.show()

# Access feature metadata
metadata = components['feature_metadata']
print(f"Model trained on {metadata['training_date']}")
print(f"Number of features: {metadata['num_features']}")
print(f"String columns: {metadata['string_columns']}")
"""

logger.info("📚 Inference example code added as comments above")
logger.info("✅ Training script completed successfully!")
