# Inference script to score new data using the trained Logistic Regression model

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from datetime import datetime
import logging
import os

# Configure logging
log_filename = f"spark_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(os.path.dirname(__file__), log_filename)

logger = logging.getLogger('spark_inference')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("✅ Libraries imported successfully")
logger.info(f"📝 Logging to file: {log_path}")

# Configuration
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

MODEL_PATH = "/root/research-dir/dev/jazzcash-fraud-detection/scripts/spark_lr_model_2"

logger.info(f"🔧 Model path: {MODEL_PATH}")

# Initialize Spark Session
logger.info("🚀 Initializing Spark Session for inference...")

try:
    spark.stop()
    logger.info("🔄 Stopped existing Spark session")
except:
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

logger.info("✅ Spark Session initialized")

# Load the trained Logistic Regression model
logger.info(f"📦 Loading trained model from: {MODEL_PATH}")

try:
    lr_model = LogisticRegressionModel.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    raise

logger.info(f"   • Model intercept: {lr_model.intercept:.6f}")
logger.info(f"   • Number of features: {len(lr_model.coefficients)}")

# Load inference data from ClickHouse
logger.info("📊 Loading inference data from ClickHouse...")

start_date = '2025-06-01'
end_date = '2025-07-30'
num_partitions = 60

selected_cols = [
    'cutoff_date',
    'fraud_flag',
    'trans_id',
    'ac_from',
    'ac_to',
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
    'user_days_since_last_txn'
]

query = f"""
SELECT {', '.join(selected_cols)}
FROM public.stixor_fraud_features_distributed
WHERE cutoff_date BETWEEN '{start_date}' AND '{end_date}'
AND ac_to GLOBAL IN (
    SELECT b.a_c_reference
    FROM public.fraud_distributed fraud
    GLOBAL INNER JOIN public.stixor_mbar_v_distributed b
    ON fraud.fraud_msisdn = b.a_c_reference
    WHERE b.account_type_name = 'Customer Account'
)
"""

subquery = f"""
(
    {query}
) AS inference_data
"""

try:
    df_inference = (spark.read
        .format('jdbc')
        .option('driver', driver)
        .option('url', url)
        .option('user', user)
        .option('password', password)
        .option('dbtable', subquery)
        .option('fetchsize', '100000')
        .option("partitionColumn", "cutoff_date")
        .option('lowerBound', start_date)
        .option('upperBound', end_date)
        .option('numPartitions', str(num_partitions))
        .load())

    logger.info("📦 Caching inference DataFrame...")
    df_inference.cache()


except Exception as e:
    logger.error("❌ ERROR LOADING INFERENCE DATA!")
    logger.error(f"Error: {str(e)}")
    raise

# Feature preprocessing (same as training)
logger.info("🔧 Preprocessing features for inference...")

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import when

TARGET_COLUMN = 'fraud_flag'

# Get all feature columns
excluded_columns = [
    TARGET_COLUMN,
    'processed_date',
    'trans_id',
    'ac_from',
    'ac_to',
    'msisdn_from',
    'msisdn_to',
    'transaction_uuid',
    'cutoff_date'
]

all_feature_cols = [col_name for col_name in df_inference.columns if col_name not in excluded_columns]

# Identify string columns
string_cols = []
for field in df_inference.schema.fields:
    if field.name in all_feature_cols and field.dataType.typeName() == 'string':
        string_cols.append(field.name)

logger.info(f"🔤 String features to encode: {string_cols}")

# Replace empty strings with 'UNKNOWN'
for col_name in string_cols:
    df_inference = df_inference.withColumn(
        col_name,
        when((F.col(col_name) == "") | F.col(col_name).isNull(), "UNKNOWN").otherwise(F.col(col_name))
    )

# Index and encode string columns
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in string_cols
]
encoders = [
    OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_ohe", handleInvalid="keep") for col in string_cols
]

cat_pipeline = Pipeline(stages=indexers + encoders)
df_cat = cat_pipeline.fit(df_inference).transform(df_inference)

# Replace original string columns with OHE columns
modeling_features = [
    f"{col}_ohe" if col in string_cols else col for col in all_feature_cols
]

modeling_features = [col for col in modeling_features if col in df_cat.columns]
modeling_features = modeling_features[1:]  # Remove first feature if needed

logger.info(f"🧮 Features for inference: {len(modeling_features)} features")

# Create feature vector
logger.info("📏 Creating feature vectors...")

assembler = VectorAssembler(
    inputCols=modeling_features,
    outputCol="raw_features",
    handleInvalid="skip"
)

df_assembled = assembler.transform(df_cat)

# Scale features
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

logger.info("✅ Feature scaling completed")

# Run inference
logger.info("🔮 Running inference on test data...")

start_time = datetime.now()

# Make predictions
predictions = lr_model.transform(df_scaled)

# Select relevant columns including transaction identifiers
results = predictions.select(
    'trans_id',
    'ac_from',
    'ac_to',
    col('fraud_flag').alias('actual_fraud_flag'),
    col('prediction').alias('predicted_fraud_flag'),
    col('probability').alias('fraud_probability')
)

results.cache()

# inference_time = (datetime.now() - start_time).total_seconds()

# logger.info(f"✅ Inference completed in {inference_time:.2f} seconds")
# logger.info(f"   • Records scored: {results.count():,}")

# # Display predictions
# logger.info("📊 Inference Results Summary:")

# pred_dist = results.groupBy('predicted_fraud_flag').count().collect()
# for row in pred_dist:
#     label = "🚨 Fraud" if row['predicted_fraud_flag'] == 1.0 else "✅ Non-Fraud"
#     logger.info(f"   • {label}: {row['count']:,} records")

# # Show sample predictions
# logger.info("\n📋 Sample Predictions:")
# results.show(10, truncate=False)

# # Calculate accuracy against actual fraud_flag
# logger.info("📊 Comparing predictions with actual labels...")

# comparison = results.withColumn(
#     'correct',
#     when(
#         (col('predicted_fraud_flag') == 1) & (col('actual_fraud_flag') == 1), 'TP'
#     ).when(
#         (col('predicted_fraud_flag') == 1) & (col('actual_fraud_flag') == 0), 'FP'
#     ).when(
#         (col('predicted_fraud_flag') == 0) & (col('actual_fraud_flag') == 1), 'FN'
#     ).otherwise('TN')
# )

# confusion = comparison.groupBy('correct').count().collect()

# logger.info("\n📋 Confusion Matrix:")
# for row in confusion:
#     logger.info(f"   • {row['correct']}: {row['count']:,}")

# # Calculate metrics
# tp = next((row['count'] for row in confusion if row['correct'] == 'TP'), 0)
# fp = next((row['count'] for row in confusion if row['correct'] == 'FP'), 0)
# fn = next((row['count'] for row in confusion if row['correct'] == 'FN'), 0)
# tn = next((row['count'] for row in confusion if row['correct'] == 'TN'), 0)

# accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# logger.info(f"\n🎯 Performance Metrics:")
# logger.info(f"   • Accuracy: {accuracy:.4f}")
# logger.info(f"   • Precision: {precision:.4f}")
# logger.info(f"   • Recall: {recall:.4f}")
# logger.info(f"   • F1-Score: {f1:.4f}")

# Save results to Parquet
logger.info("\n💾 Saving inference results to Parquet...")
output_path = f"/root/research-dir/dev/jazzcash-fraud-detection/results/fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results.write.mode("overwrite").parquet(output_path)

logger.info(f"✅ Results saved to Parquet:")
logger.info(f"   • Path: {output_path}")
# Clean up
results.unpersist()
logger.info("\n🧹 Cache cleaned - Spark session ready")
