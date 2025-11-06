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
# Import required libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from clickhouse_driver import Client
import configparser
from pathlib import Path
import logging
import os
from datetime import datetime

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

# # %%
# # Split data for training and evaluation based on time to avoid data leakage
# logger.info("✂️ Splitting data into training and test sets based on time (no leakage)...")

# # Define cutoff date for train/test split (adjust as needed)
# cutoff_date = '2025-06-20'  # All records before or on this date go to train, after to test

# # Ensure cutoff_date column is in the correct format (string or date)
# # If cutoff_date is string, comparison works; if date, cast as needed

# df_train = df_final.join(df_clean.select('cutoff_date'), on=df_final.rdd.zipWithIndex().map(lambda x: x[1]).collect(), how='left')

# df_train = df_final.join(df_clean.select('cutoff_date'), df_final.rdd.zipWithIndex().map(lambda x: x[1]).collect(), 'left')

# # Add cutoff_date column to df_final for splitting
# from pyspark.sql.functions import col as spark_col

# df_final_with_date = df_final.withColumn('cutoff_date', df_clean['cutoff_date'])

# train_data = df_final_with_date.filter(spark_col('cutoff_date') <= cutoff_date).drop('cutoff_date')
# test_data = df_final_with_date.filter(spark_col('cutoff_date') > cutoff_date).drop('cutoff_date')

# # Cache datasets for performance
# train_data.cache()
# test_data.cache()

# logger.info("📊 Data Split Summary:")
# train_count = train_data.count()
# test_count = test_data.count()
# logger.info(f"   • Training set: {train_count:,} records ({train_count/(train_count+test_count)*100:.1f}%)")
# logger.info(f"   • Test set: {test_count:,} records ({test_count/(train_count+test_count)*100:.1f}%)")

# # Check class distribution in both sets
# logger.info("📈 Class Distribution:")

# logger.info("Training Set:")
# train_dist = train_data.groupBy("label").count().collect()
# for row in train_dist:
#     label = "Fraud" if row['label'] == 1 else "Legitimate" 
#     count = row['count']
#     percentage = (count / train_count) * 100
#     logger.info(f"   • {label}: {count:,} ({percentage:.1f}%)")

# logger.info("Test Set:")
# test_dist = test_data.groupBy("label").count().collect()
# for row in test_dist:
#     label = "Fraud" if row['label'] == 1 else "Legitimate"
#     count = row['count'] 
#     percentage = (count / test_count) * 100
#     logger.info(f"   • {label}: {count:,} ({percentage:.1f}%)")

# logger.info("✅ Data split completed and cached for optimal performance")

# %% [markdown]
# ## 4. Logistic Regression Model Training
# 
# Train the Logistic Regression model using Spark MLlib.

# %%
# Configure Logistic Regression model optimized for fraud detection
logger.info("🤖 Configuring Logistic Regression for fraud detection...")


train_data = df_final
# Calculate class weights for imbalanced dataset (from fraud profiling insights)
fraud_count = train_data.filter(col("label") == 1).count()
legitimate_count = train_data.filter(col("label") == 0).count()
class_ratio = legitimate_count / fraud_count

logger.info("📊 Class Imbalance Analysis:")
logger.info(f"   • Legitimate transactions: {legitimate_count:,}")
logger.info(f"   • Fraud transactions: {fraud_count:,}")
logger.info(f"   • Class ratio (Legit/Fraud): {class_ratio:.2f}:1")

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
lr_model = lr.fit(train_data)

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

# Save model
model_path = "/root/ali-dir/spark_lr_model_3"  # Remove .pickle extension
lr_model.write().overwrite().save(model_path)
logger.info(f"✅ Model saved to {model_path}")

# # %% [markdown]
# # ## 5. Model Evaluation & Performance Analysis
# # 
# # Evaluate the trained model on test data using fraud detection specific metrics.

# # %%
# # Make predictions on test data
# print("🔮 Making predictions on test dataset...")

# # Generate predictions
# predictions = lr_model.transform(test_data)

# # Cache predictions for multiple evaluations
# predictions.cache()

# print("✅ Predictions generated")
# print(f"   • Test records: {predictions.count():,}")

# # Show sample predictions
# print(f"\n📋 Sample Predictions:")
# predictions.select("label", "prediction", "probability").show(5, truncate=False)

# # Quick prediction summary
# pred_summary = predictions.groupBy("prediction").count().collect()
# print(f"\n📊 Prediction Summary:")
# for row in pred_summary:
#     pred_label = "Predicted Fraud" if row['prediction'] == 1.0 else "Predicted Legitimate"
#     count = row['count']
#     percentage = (count / predictions.count()) * 100
#     print(f"   • {pred_label}: {count:,} ({percentage:.1f}%)")

# # %%
# # Comprehensive model evaluation for fraud detection
# print("📊 Evaluating model performance...")

# # Binary Classification Evaluator for AUC-ROC
# binary_evaluator = BinaryClassificationEvaluator(
#     labelCol="label",
#     rawPredictionCol="rawPrediction",
#     metricName="areaUnderROC"
# )

# # AUC-ROC Score
# auc_score = binary_evaluator.evaluate(predictions)
# print(f"🎯 AUC-ROC Score: {auc_score:.4f}")

# # AUC-PR Score  
# binary_evaluator_pr = BinaryClassificationEvaluator(
#     labelCol="label", 
#     rawPredictionCol="rawPrediction",
#     metricName="areaUnderPR"
# )
# auc_pr_score = binary_evaluator_pr.evaluate(predictions)
# print(f"🎯 AUC-PR Score: {auc_pr_score:.4f}")

# # Multiclass evaluator for additional metrics
# multiclass_evaluator = MulticlassClassificationEvaluator(
#     labelCol="label",
#     predictionCol="prediction"
# )

# # Accuracy
# accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
# print(f"🎯 Accuracy: {accuracy:.4f}")

# # Precision and Recall for fraud class (class 1)
# fraud_precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "precisionByLabel", multiclass_evaluator.metricLabel: 1.0})
# fraud_recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "recallByLabel", multiclass_evaluator.metricLabel: 1.0})
# fraud_f1 = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "fMeasureByLabel", multiclass_evaluator.metricLabel: 1.0})

# print(f"\n🚨 Fraud Detection Performance:")
# print(f"   • Fraud Precision: {fraud_precision:.4f} (of predicted frauds, how many are actual frauds)")
# print(f"   • Fraud Recall: {fraud_recall:.4f} (of actual frauds, how many are detected)")  
# print(f"   • Fraud F1-Score: {fraud_f1:.4f} (harmonic mean of precision and recall)")

# # Performance interpretation
# print(f"\n📈 Performance Interpretation:")
# if auc_score >= 0.9:
#     auc_rating = "Excellent"
# elif auc_score >= 0.8:
#     auc_rating = "Good" 
# elif auc_score >= 0.7:
#     auc_rating = "Fair"
# else:
#     auc_rating = "Needs Improvement"

# print(f"   • AUC-ROC ({auc_score:.3f}): {auc_rating}")
# print(f"   • Model can distinguish fraud from legitimate transactions")
# print(f"   • Fraud detection rate: {fraud_recall:.1%}")
# print(f"   • False positive rate: {1-fraud_precision:.1%} (of fraud predictions)")

# print(f"\n✅ Model evaluation completed with {auc_rating.lower()} performance for fraud detection")

# # %%
# # Confusion Matrix and detailed performance analysis
# print("📊 Creating confusion matrix for detailed analysis...")

# # Create confusion matrix using Spark SQL
# predictions.createOrReplaceTempView("predictions_table")

# confusion_matrix = spark.sql("""
#     SELECT 
#         label as actual,
#         prediction as predicted,
#         COUNT(*) as count
#     FROM predictions_table 
#     GROUP BY label, prediction
#     ORDER BY label, prediction
# """).collect()

# print(f"\n📋 Confusion Matrix:")
# print(f"                    Predicted")
# print(f"                Legit    Fraud")
# print(f"Actual  Legit    {confusion_matrix[0]['count']:>6}   {confusion_matrix[1]['count']:>6}")
# print(f"        Fraud    {confusion_matrix[2]['count']:>6}   {confusion_matrix[3]['count']:>6}")

# # Calculate detailed metrics
# tn = confusion_matrix[0]['count']  # True Negatives
# fp = confusion_matrix[1]['count']  # False Positives  
# fn = confusion_matrix[2]['count']  # False Negatives
# tp = confusion_matrix[3]['count']  # True Positives

# # Calculate business-relevant metrics
# false_positive_rate = fp / (fp + tn)
# false_negative_rate = fn / (fn + tp)
# true_positive_rate = tp / (tp + fn)  # Same as recall
# true_negative_rate = tn / (tn + fp)  # Specificity

# print(f"\n💼 Business Impact Metrics:")
# print(f"   • True Positives (Frauds Caught): {tp:,}")
# print(f"   • False Negatives (Frauds Missed): {fn:,}")
# print(f"   • False Positives (False Alarms): {fp:,}")
# print(f"   • True Negatives (Correct Legit): {tn:,}")

# print(f"\n📈 Key Rates:")
# print(f"   • Fraud Detection Rate: {true_positive_rate:.1%} (caught {tp} of {tp+fn} frauds)")
# print(f"   • False Positive Rate: {false_positive_rate:.1%} (false alarms on legit txns)")
# print(f"   • False Negative Rate: {false_negative_rate:.1%} (missed frauds)")
# print(f"   • Specificity: {true_negative_rate:.1%} (correctly identified legit txns)")

# # Cost-benefit analysis (illustrative)
# avg_fraud_amount = 50000  # PKR (example)
# investigation_cost = 500   # PKR per investigation

# fraud_prevented = tp * avg_fraud_amount
# investigation_costs = (tp + fp) * investigation_cost
# net_benefit = fraud_prevented - investigation_costs

# print(f"\n💰 Estimated Financial Impact (Illustrative):")
# print(f"   • Fraud Prevented: PKR {fraud_prevented:,}")
# print(f"   • Investigation Costs: PKR {investigation_costs:,}")
# print(f"   • Net Benefit: PKR {net_benefit:,}")

# print(f"\n✅ Detailed performance analysis completed")

# # %% [markdown]
# # ## 6. Model Persistence & Deployment Preparation
# # 
# # Save the trained model and prepare for production deployment.

# # %%
# # Save the trained model and preprocessing pipeline
# print("💾 Saving trained model and pipeline...")

# import os
# from datetime import datetime

# # Create model directory
# model_base_path = "/root/research-dir/dev/jazzcash-fraud-detection/models"
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_path = f"{model_base_path}/fraud_logistic_regression_{timestamp}"

# # Ensure directory exists
# os.makedirs(model_path, exist_ok=True)

# # Save the complete trained model
# lr_model_path = f"{model_path}/logistic_regression_model"
# lr_model.write().overwrite().save(lr_model_path)

# # Save the feature scaler
# scaler_path = f"{model_path}/feature_scaler"
# scaler_model.write().overwrite().save(scaler_path)

# # Save feature assembler
# assembler_path = f"{model_path}/feature_assembler"
# assembler.write().overwrite().save(assembler_path)

# print(f"✅ Model components saved:")
# print(f"   • Logistic Regression Model: {lr_model_path}")
# print(f"   • Feature Scaler: {scaler_path}")
# print(f"   • Feature Assembler: {assembler_path}")

# # Save model metadata
# metadata = {
#     "model_type": "LogisticRegression",
#     "training_date": timestamp,
#     "features_count": len(modeling_features),
#     "training_records": train_count,
#     "test_records": test_count,
#     "auc_score": auc_score,
#     "accuracy": accuracy,
#     "fraud_precision": fraud_precision,
#     "fraud_recall": fraud_recall,
#     "selected_features": modeling_features,
#     "source_table": TABLE_NAME,
#     "clickhouse_database": CLICKHOUSE_CONFIG['database']
# }

# # Save metadata as JSON
# import json
# metadata_path = f"{model_path}/model_metadata.json"
# with open(metadata_path, 'w') as f:
#     json.dump(metadata, f, indent=2)

# print(f"   • Model Metadata: {metadata_path}")

# print(f"\n📦 Model Package Created: {model_path}")
# print(f"🚀 Ready for production deployment!")

# # Display deployment readiness checklist
# print(f"\n✅ Deployment Readiness Checklist:")
# print(f"   ✓ Model trained and validated")
# print(f"   ✓ Performance metrics documented")
# print(f"   ✓ Model artifacts saved")
# print(f"   ✓ Feature pipeline preserved")
# print(f"   ✓ ClickHouse connection configured")
# print(f"   ✓ Metadata and lineage documented")

# print(f"\n🎯 Next Steps for Production:")
# print(f"   1. Deploy model to production Spark cluster")
# print(f"   2. Create real-time scoring API")
# print(f"   3. Set up monitoring and alerting")
# print(f"   4. Implement A/B testing framework")
# print(f"   5. Schedule model retraining pipeline")

# # %%
# # Summary and final information
# print("=" * 80)
# print("🎉 FRAUD DETECTION MODEL TRAINING COMPLETED SUCCESSFULLY!")
# print("=" * 80)

# print(f"\n📊 FINAL MODEL SUMMARY:")
# print(f"   • Algorithm: Logistic Regression (Spark MLlib)")
# print(f"   • Data Source: ClickHouse {TABLE_NAME}")
# print(f"   • Features: {len(modeling_features)} engineered features")
# print(f"   • Training Data: {train_count:,} records")
# print(f"   • Test Data: {test_count:,} records")

# print(f"\n🎯 PERFORMANCE METRICS:")
# print(f"   • AUC-ROC: {auc_score:.4f}")
# print(f"   • AUC-PR: {auc_pr_score:.4f}")
# print(f"   • Accuracy: {accuracy:.4f}")
# print(f"   • Fraud Precision: {fraud_precision:.4f}")
# print(f"   • Fraud Recall: {fraud_recall:.4f}")
# print(f"   • Fraud F1-Score: {fraud_f1:.4f}")

# print(f"\n💼 BUSINESS IMPACT:")
# print(f"   • Fraud Detection Rate: {true_positive_rate:.1%}")
# print(f"   • False Positive Rate: {false_positive_rate:.1%}")
# print(f"   • Estimated Frauds Caught: {tp:,}")
# print(f"   • Estimated False Alarms: {fp:,}")

# print(f"\n🔧 TECHNICAL DETAILS:")
# print(f"   • Feature Engineering: Based on fraud profiling analysis")
# print(f"   • Data Pipeline: ClickHouse → Spark → MLlib")
# print(f"   • Model Training: {training_time:.1f} seconds")
# print(f"   • Convergence: {lr_model.summary.totalIterations} iterations")

# print(f"\n📦 ARTIFACTS CREATED:")
# print(f"   • Trained Model: {lr_model_path}")
# print(f"   • Feature Pipeline: {scaler_path}")
# print(f"   • Model Metadata: {metadata_path}")

# print(f"\n🚀 PRODUCTION READY:")
# print(f"   ✓ Model can be loaded for real-time scoring")
# print(f"   ✓ Feature pipeline preserved for consistency")  
# print(f"   ✓ ClickHouse integration established")
# print(f"   ✓ Performance benchmarks documented")

# print(f"\n💡 RECOMMENDATIONS:")
# print(f"   • Deploy with fraud_recall threshold optimization")
# print(f"   • Monitor model drift with weekly retraining")
# print(f"   • Implement A/B testing vs current fraud system")
# print(f"   • Set up real-time feature computation pipeline")

# print("=" * 80)
# print("✅ Fraud detection model ready for production deployment!")
# print("📞 Contact MLOps team for production deployment assistance")
# print("=" * 80)

# # Clean up cached data
# train_data.unpersist()
# test_data.unpersist()
# predictions.unpersist()

# print("\n🧹 Cache cleaned up - Spark session ready for reuse")

# # %%



