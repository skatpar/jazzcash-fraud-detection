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
from pylogger.info(f"📊 Model Performance Summary:")
logger.info(f"   • Training Time: {training_time:.2f} seconds")
logger.info(f"   • Training Iterations: {lr_model.summarlogger.info(f"📊 Model Performance Summary:")
logger.info(f"   • Training Time: {training_time:.2f} seconds")
logger.info(f"   • Training Iterations: {lr_model.summary.totalIterations}")
logger.info(f"   • Final Objective: {lr_model.summary.objectiveHistory[-1]:.6f}")
logger.info(f"   • Number of Features: {len(modeling_features)}")
logger.info(f"   • Training Records: {feature_metadata['training_rows']}")
logger.info(f"   • Model Components: {len(saved_paths)} saved")

# %% [markdown]
# ## 4. Model Inference on Training Data
# 
# Test the inference pipeline by making predictions on a subset of the same data

# %%
logger.info("🔮 Starting Model Inference on Training Data...")
logger.info("=" * 60)

# Test inference using the trained components directly (without reloading)
logger.info("🧪 Testing inference pipeline with trained components...")

# Step 1: Prepare a subset of the original data for inference testing
# Use the original df (before preprocessing) to simulate new data
logger.info("📊 Preparing inference test data...")
inference_sample = df.limit(1000)  # Take first 1000 rows for inference testing
sample_count = inference_sample.count()
logger.info(f"   • Using {sample_count} rows for inference testing")

# Step 2: Apply the same preprocessing pipeline as during training
logger.info("🔧 Applying preprocessing pipeline to test data...")

# Apply categorical preprocessing (use the fitted pipeline)
logger.info("   • Step 1: Applying categorical preprocessing...")
cat_pipeline_fitted = cat_pipeline.fit(df)
inference_df_cat = cat_pipeline_fitted.transform(inference_sample)

# Apply feature assembly
logger.info("   • Step 2: Assembling feature vectors...")
inference_df_assembled = assembler.transform(inference_df_cat)

# Apply feature scaling
logger.info("   • Step 3: Scaling features...")
inference_df_scaled = scaler_model.transform(inference_df_assembled)

# Prepare final dataset for prediction
logger.info("   • Step 4: Preparing final dataset for prediction...")
inference_df_final = inference_df_scaled.select("features")

logger.info("✅ Preprocessing pipeline applied successfully!")

# Step 3: Make predictions using the trained model
logger.info("🎯 Making fraud predictions...")
start_inference_time = time.time()

predictions = lr_model.transform(inference_df_final)
inference_time = time.time() - start_inference_time

logger.info(f"✅ Predictions completed in {inference_time:.2f} seconds")

# Step 4: Analyze prediction results
logger.info("📊 Analyzing prediction results...")

# Get original labels for comparison
original_labels = inference_sample.select(col(TARGET_COLUMN).alias("actual_label"))

# Add row IDs for joining
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, monotonically_increasing_id

window = Window.orderBy(monotonically_increasing_id())
predictions_with_id = predictions.withColumn("row_id", row_number().over(window))
labels_with_id = original_labels.withColumn("row_id", row_number().over(window))

# Join predictions with actual labels
results = predictions_with_id.join(labels_with_id, "row_id").drop("row_id")

# Show prediction distribution
logger.info("📈 Prediction Distribution:")
prediction_counts = results.groupBy("prediction").count().collect()
for row in prediction_counts:
    pred_label = "Fraud" if row.prediction == 1.0 else "Legitimate"
    logger.info(f"   • {pred_label}: {row.count} predictions")

# Calculate accuracy
logger.info("🎯 Accuracy Metrics:")
total_predictions = results.count()
correct_predictions = results.filter(col("prediction") == col("actual_label")).count()
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

logger.info(f"   • Total Predictions: {total_predictions}")
logger.info(f"   • Correct Predictions: {correct_predictions}")
logger.info(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Analyze fraud probabilities
logger.info("🔍 Fraud Probability Analysis:")

# Extract fraud probability (second element of probability vector)
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

def extract_fraud_prob(prob_vector):
    return float(prob_vector[1]) if prob_vector and len(prob_vector) > 1 else 0.0

extract_fraud_prob_udf = udf(extract_fraud_prob, DoubleType())
results_with_fraud_prob = results.withColumn("fraud_probability", extract_fraud_prob_udf("probability"))

# Show sample predictions
logger.info("📋 Sample Predictions (first 10):")
sample_predictions = results_with_fraud_prob.select(
    "prediction", 
    "actual_label",
    "fraud_probability"
).limit(10)

sample_results = sample_predictions.collect()
for i, row in enumerate(sample_results, 1):
    fraud_prob_pct = row.fraud_probability * 100
    actual = "Fraud" if row.actual_label == 1.0 else "Legitimate"
    predicted = "Fraud" if row.prediction == 1.0 else "Legitimate"
    logger.info(f"   {i:2d}. Actual: {actual:10s} | Predicted: {predicted:10s} | Fraud Prob: {fraud_prob_pct:5.1f}%")

# Risk categorization
high_risk_count = results_with_fraud_prob.filter(col("fraud_probability") > 0.7).count()
medium_risk_count = results_with_fraud_prob.filter(
    (col("fraud_probability") >= 0.3) & (col("fraud_probability") <= 0.7)
).count()
low_risk_count = results_with_fraud_prob.filter(col("fraud_probability") < 0.3).count()

logger.info("⚠️  Risk Distribution:")
logger.info(f"   • High-risk (>70% fraud probability): {high_risk_count}")
logger.info(f"   • Medium-risk (30-70% fraud probability): {medium_risk_count}")
logger.info(f"   • Low-risk (<30% fraud probability): {low_risk_count}")

# Performance metrics
if total_predictions > 0:
    inference_speed = total_predictions / inference_time
    logger.info(f"⚡ Inference Performance:")
    logger.info(f"   • Inference Speed: {inference_speed:.0f} predictions/second")
    logger.info(f"   • Average Time per Prediction: {inference_time*1000/total_predictions:.2f} ms")

logger.info("🎯 Inference testing completed successfully!")

# Final summary
logger.info("📈 FINAL SUMMARY:")
logger.info(f"   • Training Time: {training_time:.2f} seconds")
logger.info(f"   • Inference Time: {inference_time:.2f} seconds")
logger.info(f"   • Model Accuracy: {accuracy*100:.2f}%")
logger.info(f"   • High-risk Detections: {high_risk_count}")
logger.info(f"   • Total Components Saved: {len(saved_paths)}")
logger.info(f"   • Inference Speed: {inference_speed:.0f} predictions/sec")

logger.info("✅ Training and Inference script completed successfully!")Iterations}")
logger.info(f"   • Final Objective: {lr_model.summary.objectiveHistory[-1]:.6f}")
logger.info(f"   • Number of Features: {len(modeling_features)}")
logger.info(f"   • Training Records: {feature_metadata['training_rows']}")
logger.info(f"   • Model Components: {len(saved_paths)} saved")

# %% [markdown]
# ## 4. Model Inference on Training Data
# 
# Test the inference pipeline by loading the saved components and making predictions on the same data

# %%
logger.info("🔮 Starting Model Inference on Training Data...")
logger.info("=" * 60)

# Load all preprocessing components using the component manager
logger.info("📦 Loading saved components for inference...")
inference_components = component_manager.load_all_components()

# Validate loaded components
logger.info("🔍 Validating loaded components...")
validation_result = component_manager.validate_components(inference_components)

if not validation_result:
    logger.error("❌ Component validation failed! Cannot proceed with inference.")
    raise RuntimeError("Component validation failed")

logger.info("✅ All components loaded and validated successfully!")

# Extract individual components for cleaner code
inference_model = inference_components['model']
inference_cat_pipeline = inference_components['categorical_pipeline']
inference_assembler = inference_components['vector_assembler']
inference_scaler = inference_components['standard_scaler']
inference_metadata = inference_components['feature_metadata']

logger.info("🔧 Applying inference pipeline to training data...")

# Step 1: Prepare a subset of the original data for inference testing
# Use the original df (before preprocessing) to simulate new data
inference_sample = df.limit(1000)  # Take first 1000 rows for inference testing
logger.info(f"📊 Using {inference_sample.count()} rows for inference testing")

# Step 2: Apply the same preprocessing pipeline as during training
logger.info("   • Step 1: Applying categorical preprocessing...")
inference_df_cat = inference_cat_pipeline.transform(inference_sample)

logger.info("   • Step 2: Assembling feature vectors...")
inference_df_assembled = inference_assembler.transform(inference_df_cat)

logger.info("   • Step 3: Scaling features...")
inference_df_scaled = inference_scaler.transform(inference_df_assembled)

logger.info("   • Step 4: Preparing final dataset for prediction...")
inference_df_final = inference_df_scaled.select("features")

logger.info("✅ Preprocessing pipeline applied successfully!")

# Step 3: Make predictions using the loaded model
logger.info("🎯 Making fraud predictions...")
start_inference_time = time.time()

predictions = inference_model.transform(inference_df_final)
inference_time = time.time() - start_inference_time

logger.info(f"✅ Predictions completed in {inference_time:.2f} seconds")

# Step 4: Analyze prediction results
logger.info("📊 Analyzing prediction results...")

# Add original target labels for comparison
inference_with_labels = predictions.join(
    inference_sample.select(col(TARGET_COLUMN).alias("actual_label")),
    on=predictions.features == inference_df_scaled.select("features").rdd.zipWithIndex().map(lambda x: x[1]).collect(),
    how="inner"
)

# Since the join is complex with features, let's use a simpler approach
# Create predictions with row numbers for comparison
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Add row numbers to both datasets
window = Window.orderBy(F.monotonically_increasing_id())

predictions_with_row = predictions.withColumn("row_num", row_number().over(window))
actual_with_row = inference_sample.select(col(TARGET_COLUMN).alias("actual_label")).withColumn("row_num", row_number().over(window))

# Join on row numbers
results_comparison = predictions_with_row.join(actual_with_row, "row_num").drop("row_num")

# Show prediction statistics
logger.info("📈 Prediction Distribution:")
prediction_dist = results_comparison.groupBy("prediction").count().collect()
for row in prediction_dist:
    pred_label = "Fraud" if row.prediction == 1.0 else "Legitimate"
    logger.info(f"   • {pred_label}: {row.count} predictions")

# Show accuracy if we have actual labels
logger.info("🎯 Accuracy Metrics:")
correct_predictions = results_comparison.filter(col("prediction") == col("actual_label")).count()
total_predictions = results_comparison.count()
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

logger.info(f"   • Total Predictions: {total_predictions}")
logger.info(f"   • Correct Predictions: {correct_predictions}")
logger.info(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Show sample predictions with probabilities
logger.info("📋 Sample Predictions:")
sample_results = results_comparison.select(
    "prediction", 
    "actual_label",
    "probability"
).limit(10)

sample_results.show(truncate=False)

# Fraud probability analysis
logger.info("🔍 Fraud Probability Analysis:")
fraud_probs = results_comparison.select("prediction", "probability", "actual_label")

# Extract fraud probability (second element of probability vector)
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np

def extract_fraud_prob(prob_vector):
    return float(prob_vector[1])

extract_fraud_prob_udf = udf(extract_fraud_prob, DoubleType())
fraud_probs_extracted = fraud_probs.withColumn("fraud_probability", extract_fraud_prob_udf("probability"))

# Show fraud probability statistics
fraud_prob_stats = fraud_probs_extracted.select("fraud_probability").describe()
logger.info("📊 Fraud Probability Statistics:")
fraud_prob_stats.show()

# High-risk transactions (fraud probability > 0.7)
high_risk_count = fraud_probs_extracted.filter(col("fraud_probability") > 0.7).count()
logger.info(f"⚠️  High-risk transactions (>70% fraud probability): {high_risk_count}")

# Medium-risk transactions (fraud probability 0.3-0.7)
medium_risk_count = fraud_probs_extracted.filter(
    (col("fraud_probability") >= 0.3) & (col("fraud_probability") <= 0.7)
).count()
logger.info(f"⚡ Medium-risk transactions (30-70% fraud probability): {medium_risk_count}")

# Low-risk transactions (fraud probability < 0.3)
low_risk_count = fraud_probs_extracted.filter(col("fraud_probability") < 0.3).count()
logger.info(f"✅ Low-risk transactions (<30% fraud probability): {low_risk_count}")

logger.info("🎯 Inference testing completed successfully!")
logger.info("✅ Model inference pipeline validated and working correctly!")

# Performance summary
logger.info("📈 Final Performance Summary:")
logger.info(f"   • Training Time: {training_time:.2f} seconds")
logger.info(f"   • Inference Time: {inference_time:.2f} seconds")
logger.info(f"   • Inference Speed: {total_predictions/inference_time:.0f} predictions/second")
logger.info(f"   • Model Accuracy: {accuracy*100:.2f}%")
logger.info(f"   • High-risk Detections: {high_risk_count}")
logger.info(f"   • Components Saved: {len(saved_paths)}")

logger.info("✅ Training and Inference script completed successfully!")l.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssemblerModel, StandardScalerModel
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from clickhouse_driver import Client
import configparser
from pathlib import Path
import logging
import os
from datetime import datetime
import json
from model_utils import ModelComponentManager


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

start_date = '2025-06-01'
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
        AND fraud_flag = 0
    LIMIT 100000
    UNION ALL
    SELECT {', '.join(selected_cols)}
    FROM stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
        AND mbar_account_type_name = 'Customer Account'
        AND fraud_flag = 1    
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
    total_rows = df.count()
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

# Save model and all preprocessing components using ModelComponentManager
base_path = "/root/research-dir/dev/jazzcash-fraud-detection/scripts"
model_name = "fraud_model_june"

# Initialize the model component manager
component_manager = ModelComponentManager(base_path, model_name)

# Prepare feature metadata
feature_metadata = {
    "string_columns": string_cols,
    "modeling_features": modeling_features,
    "all_feature_cols": all_feature_cols,
    "target_column": TARGET_COLUMN,
    "excluded_columns": excluded_columns,
    "num_features": len(modeling_features),
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "feature_vector_size": len(modeling_features),
    "training_rows": df_final.count(),
    "model_type": "LogisticRegression",
    "regularization": lr.getRegParam(),
    "max_iterations": lr.getMaxIter()
}

# Save all components using the manager
logger.info("💾 Saving all model components using ModelComponentManager...")
saved_paths = component_manager.save_all_components(
    model=lr_model,
    categorical_pipeline=cat_pipeline.fit(df),
    vector_assembler=assembler,
    scaler_model=scaler_model,
    feature_metadata=feature_metadata,
    training_df=df_final
)

logger.info("🎯 All components saved successfully using ModelComponentManager!")
logger.info("📦 Saved components summary:")
for component_type, path in saved_paths.items():
    logger.info(f"   • {component_type}: {path}")

# Validate the saved components
logger.info("🔍 Validating saved components...")
loaded_components = component_manager.load_all_components()
is_valid = component_manager.validate_components(loaded_components)

if is_valid:
    logger.info("✅ Component validation successful!")
else:
    logger.error("❌ Component validation failed!")

logger.info(f"� Model Performance Summary:")
logger.info(f"   • Training Time: {training_time:.2f} seconds")
logger.info(f"   • Training Iterations: {lr_model.summary.totalIterations}")
logger.info(f"   • Final Objective: {lr_model.summary.objectiveHistory[-1]:.6f}")
logger.info(f"   • Number of Features: {len(modeling_features)}")
logger.info(f"   • Training Records: {feature_metadata['training_rows']}")
logger.info(f"   • Model Components: {len(saved_paths)} saved")
