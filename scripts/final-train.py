from ray.air.config import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig
import raydp
import logging
from datetime import datetime
import time
import os

# Spark ML imports for preprocessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# Correct import from raydp.xgboost
from raydp.xgboost import XGBoostEstimator


# --- Configuration ---

# Configure logging
log_filename = f"ray_spark_training_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ray_spark_training')
logger.info(f"📝 Logging to file: {os.path.abspath(log_filename)}")


# ==============================================================================
# --- OPTIMIZED & FIXED CONFIGURATION ---
# ==============================================================================

# Ray and Spark Configuration
APP_NAME = "RaySparkLR_Fixed"
NUM_EXECUTORS = 8
CORES_PER_EXECUTOR = 7
MEMORY_PER_EXECUTOR = "18G"

NUM_TRAINING_WORKERS = 60
CPUS_PER_TRAINING_WORKER = 1
USE_GPU = False

# ClickHouse Configuration
CLICKHOUSE_CONFIG = {
    'host': '10.205.161.118',
    'database': 'public',
    'user': 'default',
    'password': 'DfsTeChB1'
}
CLICKHOUSE_JDBC_URL = f"jdbc:ch://{CLICKHOUSE_CONFIG['host']}:8123/{CLICKHOUSE_CONFIG['database']}"
CLICKHOUSE_DRIVER = "com.clickhouse.jdbc.ClickHouseDriver"
JAR_PATH = "/root/ray/clickhouse-jdbc-0.9.3-all-dependencies.jar"

# Data and Feature Configuration
START_DATE = '2025-06-28'
END_DATE = '2025-06-30'
NUM_PARTITIONS = 200
TARGET_COLUMN = 'fraud_flag'
EXCLUDED_COLS = {TARGET_COLUMN, 'cutoff_date', 'processed_date', 'ac_from', 'ac_to', 'msisdn_from', 'msisdn_to', 'transaction_uuid'}

ALL_COLS = [
    'cutoff_date', 'fraud_flag', 'trx_channel', 'trx_type', 'start_balance', 'trx_amt',
    'mbar_registered_channel', 'mbar_a_c_status', 'mbar_a_c_level', 'mbar_account_type_name',
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours', 'is_unusual_hour',
    'night_weekend_combo', 'start_balance_log', 'txn_txns_3d', 'txn_total_amount_3d',
    'txn_avg_amount_3d', 'txn_max_amount_3d', 'txn_min_amount_3d', 'txn_unique_recipients_3d',
    'txn_unique_channels_3d', 'txn_unique_types_3d', 'txn_is_high_activity_3d',
    'txn_multi_channel_recent', 'txn_amount_deviation_from_avg', 'txn_night_txns_3d',
    'txn_weekend_txns_3d', 'channel_new_jc_app', 'channel_ussd', 'channel_ussd_api',
    'channel_payment_gateway', 'channel_mobile_app', 'type_transfer_c2c', 'type_transfer_c2b', 'type_bill_payment',
    'type_mobile_load', 'user_total_txns_3d', 'user_total_amount_3d', 'user_avg_amount_3d',
    'user_median_amount_3d', 'user_max_amount_3d', 'user_min_amount_3d', 'user_unique_recipients_3d',
    'user_unique_channels_3d', 'user_unique_types_3d', 'user_total_txns_7d', 'user_total_amount_7d',
    'user_avg_amount_7d', 'user_median_amount_7d', 'user_max_amount_7d', 'user_min_amount_7d',
    'user_unique_recipients_7d', 'user_unique_channels_7d', 'user_unique_types_7d',
    'user_most_used_channel_7d', 'user_last_used_channel', 'user_channel_diversity_score_7d',
    'user_most_used_type_7d', 'user_last_used_type', 'user_type_diversity_score_7d',
    'user_night_txns_7d', 'user_weekend_txns_7d', 'user_peak_hour_txns_7d',
    'user_off_peak_hour_txns_7d', 'user_avg_start_balance_7d', 'user_avg_end_balance_7d',
    'user_min_balance_7d', 'user_max_balance_7d', 'user_balance_volatility_7d',
    'user_avg_amount_per_recipient_7d', 'user_max_amount_to_single_recipient_7d',
    'user_recipient_concentration_ratio_7d', 'user_avg_time_between_txns_7d',
    'user_txn_frequency_score_7d', 'user_days_since_last_txn'
]


logger.info("🚀 Initializing Ray and Spark on Ray (raydp)...")
ray.init(ignore_reinit_error=True)
spark = raydp.init_spark(
    app_name=APP_NAME,
    num_executors=NUM_EXECUTORS,
    executor_cores=CORES_PER_EXECUTOR,
    executor_memory=MEMORY_PER_EXECUTOR,
    configs={
        "spark.jars": JAR_PATH,
        "spark.sql.execution.arrow.pyspark.enabled": "true",
    }
)
logger.info("✅ Ray and Spark initialized.")


logger.info(f"⏳ Loading data from ClickHouse for date range: {START_DATE} to {END_DATE}...")
query = f"""
    SELECT {', '.join(ALL_COLS)}
    FROM stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        AND mbar_account_type_name = 'Customer Account'
"""
subquery = f"({query}) AS fraud_data"

load_start_time = time.time()
try:
    df = (spark.read
        .format('jdbc')
        .option('driver', CLICKHOUSE_DRIVER)
        .option('url', CLICKHOUSE_JDBC_URL)
        .option('user', CLICKHOUSE_CONFIG['user'])
        .option('password', CLICKHOUSE_CONFIG['password'])
        .option('dbtable', subquery)
        .option('fetchsize', '100000')
        .option("partitionColumn", "cutoff_date")
        .option('lowerBound', START_DATE)
        .option('upperBound', END_DATE)
        .option('numPartitions', str(NUM_PARTITIONS))
        .load())

    total_rows = df.count()
    load_duration = time.time() - load_start_time
    logger.info(f"✅ Data loaded successfully in {load_duration:.2f} seconds.")
    logger.info(f"   • Total rows loaded: {total_rows:,}")

except Exception as e:
    logger.error(f"❌ ERROR loading data from ClickHouse: {e}", exc_info=True)
    raydp.stop_spark()

logger.info("🛠️ Defining preprocessing pipeline with Spark ML...")

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
training_df = df_scaled.select("features", col(TARGET_COLUMN).alias("label"))

logger.info("✅ Final dataset prepared:")
logger.info(f"   • Features: Vector of {len(modeling_features)} elements")
logger.info(f"   • Label: Binary (0=Legitimate, 1=Fraud)")
logger.info(f"   • Records: {training_df.count():,}")

# Show sample of the prepared data
logger.info("📋 Sample of Prepared Data:")
training_df.show(3, truncate=False)


# ----------------------

logger.info("🚂 Configuring XGBoostEstimator for Logistic Regression...")
resources_per_worker = {"CPU": CPUS_PER_TRAINING_WORKER}
if USE_GPU:
    resources_per_worker["GPU"] = 1

# Instantiate the correct XGBoostEstimator
estimator = XGBoostEstimator(
    xgboost_params={
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "num_boost_round": 100,
    },
    label_column=TARGET_COLUMN,
    num_workers=NUM_TRAINING_WORKERS,
    resources_per_worker=resources_per_worker
)

logger.info(f"🚀 Starting distributed model training with {NUM_TRAINING_WORKERS} workers...")
train_start_time = time.time()

# Use fit_on_spark with the preprocessed DataFrame
# Note: fit_on_spark does not return a Spark ML model
estimator.fit_on_spark(training_df)

train_duration = time.time() - train_start_time
logger.info(f"✅ Training completed in {train_duration:.2f} seconds.")

# Get the Ray XGBoost training results and model checkpoint
# result = estimator.get_results()
# logger.info("📊 Training Results:")
# metrics = result.metrics
# logger.info(f"   • Final AUC: {metrics.get('train-auc-mean'):.4f}")
# logger.info(f"   • Final LogLoss: {metrics.get('train-logloss-mean'):.4f}")

# Save the Ray checkpoint and the Spark pipeline separately
ray_model_path = "/root/ali-dir/ray_xgboost_model_from_raydp"
spark_pipeline_path = "/root/ali-dir/spark_preprocessing_pipeline"

# result.checkpoint.to_directory(ray_model_path)
# pipeline_model.write().overwrite().save(spark_pipeline_path)

logger.info(f"✅ Ray XGBoost model saved to: {ray_model_path}")