# %% [markdown]
# # Fraud Centroid Cosine Similarity Analysis
# 
# ## Overview
# 
# This notebook implements a **fraud detection analysis technique** using **cosine similarity** to measure how similar transactions are to known fraudulent behavior patterns.
# 
# ### Key Concepts
# 
# 1. **Fraud Centroid**: The "average" or mean vector representation of all known fraudulent transactions. This serves as a prototype of typical fraud behavior.
# 
# 2. **Cosine Similarity**: A metric that measures the similarity between two vectors by calculating the cosine of the angle between them. Values range from -1 (opposite) to 1 (identical), with higher values indicating more similarity.
# 
# 3. **Feature Engineering**: We use 42 engineered features including:
#    - Transaction characteristics (amount, balance)
#    - Temporal patterns (hour, day, weekend, night)
#    - Behavioral aggregates (3-day and 7-day rolling statistics)
#    - Channel and type indicators (one-hot encoded)
# 
# ### Analysis Types
# 
# This notebook provides two types of analysis:
# 
# 1. **Query-Based Analysis**: Calculate similarity for a large set of transactions matching a SQL query (e.g., all non-fraud transactions in a date range)
# 
# 2. **Single Transaction Analysis**: Calculate similarity for a specific transaction by its ID
# 
# ---

# %% [markdown]
# ## Part 1: Setup and Configuration
# 
# ### 1.1 Import Required Libraries
# 
# We'll need the following libraries:
# - **PySpark**: For distributed computing and handling large datasets
# - **NumPy**: For numerical operations (centroid calculation, cosine similarity)
# - **Pandas**: For final result manipulation
# - **Logging**: For tracking execution progress

# %%
import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, mean, stddev, count,
    min as spark_min, max as spark_max
)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Summarizer


# %% [markdown]
# ### 1.2 Configure Environment Variables
# 
# PySpark needs to know which Python interpreter to use for both the driver and executors. We set these environment variables to ensure consistency across the cluster.

# %%
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# %% [markdown]
# ### 1.3 Define Feature Columns
# 
# These are the 42 engineered features used for fraud detection. They capture:
# 
# | Category | Features | Description |
# |----------|----------|-------------|
# | **Transaction Basics** | `start_balance`, `trx_amt` | Account balance before transaction, transaction amount |
# | **Temporal Features** | `hour_of_day`, `day_of_week`, `is_weekend`, `is_night`, etc. | When the transaction occurred |
# | **3-Day Aggregates** | `txn_txns_3d`, `txn_total_amount_3d`, etc. | Transaction patterns in last 3 days |
# | **7-Day Aggregates** | `user_total_txns_7d`, `user_avg_amount_7d`, etc. | Longer-term user behavior |
# | **Channel Indicators** | `channel_new_jc_app`, `channel_ussd`, etc. | One-hot encoded transaction channels |
# | **Type Indicators** | `type_transfer_c2c`, `type_transfer_c2b`, etc. | One-hot encoded transaction types |

# %%
FEATURE_COLS = [
    'start_balance', 'trx_amt',
    
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 
    'is_business_hours', 'is_unusual_hour', 'night_weekend_combo',
    
    'txn_txns_3d', 'txn_total_amount_3d', 'txn_avg_amount_3d',
    'txn_max_amount_3d', 'txn_min_amount_3d', 'txn_unique_recipients_3d',
    'txn_unique_channels_3d', 'txn_unique_types_3d', 'txn_is_high_activity_3d',
    'txn_multi_channel_recent', 'txn_amount_deviation_from_avg',
    'txn_night_txns_3d', 'txn_weekend_txns_3d',
    
    'channel_new_jc_app', 'channel_ussd', 'channel_ussd_api',
    'channel_payment_gateway', 'channel_mobile_app',
    
    'type_transfer_c2c', 'type_transfer_c2b', 'type_bill_payment',
    'type_mobile_load',
    
    'user_total_txns_3d', 'user_total_amount_3d',
    'user_avg_amount_3d', 'user_max_amount_3d', 'user_unique_recipients_3d',
    'user_unique_channels_3d',
    
    'user_total_txns_7d', 'user_avg_amount_7d',
    'user_max_amount_7d', 'user_night_txns_7d', 'user_weekend_txns_7d'
]

for i, feat in enumerate(FEATURE_COLS, 1):
    print(f"   {i:2d}. {feat}")

# %% [markdown]
# ### 1.4 Initialize Spark Session
# 
# We'll create a Spark session connected to:
# - **Spark Cluster**: Running at `spark://10.205.161.118:7077`
# - **ClickHouse Database**: For reading transaction data
# 
# The configuration includes:
# - **150GB executor memory** with 32 cores per executor
# - **2 executor instances** for parallel processing
# - **ClickHouse Spark connector** packages for database access

# %%
def initialize_spark():
    
    packages = [
        "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
        "com.clickhouse:clickhouse-client:0.9.4",
        "com.clickhouse:clickhouse-http-client:0.9.4",
        "org.apache.httpcomponents.client5:httpclient5:5.2.1"
    ]
    
    spark = (SparkSession.builder
        .appName("FraudCentroidSimilarityAnalysis")
        .master("spark://10.205.161.118:7077")
        .config("spark.jars.packages", ",".join(packages))
        .config("spark.executor.memory", "150g")
        .config("spark.executor.memoryOverhead", "5g")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.cores", "32")
        .config("spark.executor.instances", "2")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "96")
        .getOrCreate()
    )
    
    spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
    spark.conf.set("spark.sql.catalog.clickhouse.host", "localhost")
    spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
    spark.conf.set("spark.sql.catalog.clickhouse.http_port", "8123")
    spark.conf.set("spark.sql.catalog.clickhouse.user", "default")
    spark.conf.set("spark.sql.catalog.clickhouse.password", "DfsTeChB1")
    spark.conf.set("spark.sql.catalog.clickhouse.database", "public")
    
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"{spark.version})")
    return spark

spark = initialize_spark()

# %% [markdown]
# ---
# 
# ## Part 2: Core Functions
# 
# ### 2.1 Data Loading Functions
# 
# These functions handle loading data from ClickHouse:
# - `load_fraud_transactions()`: Load known fraud transactions for centroid calculation
# - `load_query_transactions()`: Load transactions matching a custom SQL WHERE clause
# - `load_single_transaction()`: Load a specific transaction by ID

# %%
def load_fraud_transactions(spark, start_date, end_date):
    
    query = f"""
    SELECT *
    FROM clickhouse.public.stixor_fraud_features_distributed
    WHERE fraud_flag = 1
    AND cutoff_date >= '{start_date}' 
    AND cutoff_date <= '{end_date}'
    """
    
    df = spark.sql(query)
    count = df.count()
    logger.info(f"Loaded {count:,} fraud transactions")
    return df


def load_query_transactions(spark, where_clause):

    query = f"""
    SELECT *
    FROM clickhouse.public.stixor_fraud_features_distributed
    WHERE {where_clause}
    """
    
    df = spark.sql(query)
    count = df.count()
    logger.info(f"Loaded {count:,} transactions matching query")
    return df


def load_single_transaction(spark, trans_id):
    
    query = f"""
    SELECT *
    FROM clickhouse.public.stixor_fraud_features_distributed
    WHERE trans_id = '{trans_id}'
    """
    
    df = spark.sql(query)
    count = df.count()
    
    if count == 0:
        logger.error(f"Transaction {trans_id} not found!")
        return None
    
    return df

# %% [markdown]
# ### 2.3 Feature Vectorization and Scaling
# 
# **Why scale features?** Features have different scales (e.g., transaction amount in thousands vs. binary indicators). Without scaling, high-magnitude features would dominate the cosine similarity calculation.
# 
# We use `StandardScaler` with:
# - `withStd=True`: Scale by standard deviation
# - `withMean=False`: Don't center (required for sparse vectors)
# 
# **Important**: We fit the scaler **only on fraud data**, then apply the same scaler to target data. This ensures the fraud centroid is computed in a consistent feature space.

# %%
def vectorize_and_scale(df, feature_cols):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    df = assembler.transform(df)
    
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    return df, scaler_model


def apply_vectorize_and_scale(df, feature_cols, scaler_model):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    df = assembler.transform(df)
    
    df = scaler_model.transform(df)
    
    return df


# %% [markdown]
# ### 2.4 Fraud Centroid Calculation
# 
# The **fraud centroid** is the mean vector of all fraud transactions. It represents the "typical" fraud pattern in our feature space.
# 
# Mathematically:
# $$\text{centroid} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i$$
# 
# where $\mathbf{x}_i$ is the feature vector of the $i$-th fraud transaction.

# %%
def calculate_fraud_centroid(fraud_df):
    
    fraud_features = fraud_df.select('features')
    centroid = fraud_features.select(
        Summarizer.mean(col('features')).alias('centroid')
    ).collect()[0]['centroid']
    
    centroid_array = centroid.toArray()
    centroid_norm = float(np.linalg.norm(centroid_array))
    
    logger.info(f"   Dimensions: {len(centroid_array)}")
    logger.info(f"   L2 Norm: {centroid_norm:.6f}")
    
    return centroid_array, centroid_norm



# %% [markdown]
# ### 2.5 Cosine Similarity Calculation
# 
# **Cosine similarity** measures the angle between two vectors:
# 
# $$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$
# 
# Where:
# - $\mathbf{a} \cdot \mathbf{b}$ is the dot product
# - $\|\mathbf{a}\|$ and $\|\mathbf{b}\|$ are the L2 norms
# 
# **Interpretation**:
# | Similarity | Meaning |
# |------------|---------|
# | 0.9 - 1.0 |  VERY HIGH - Strongly resembles fraud pattern |
# | 0.7 - 0.9 |  HIGH - Similar to fraud pattern |
# | 0.5 - 0.7 |  MODERATE - Some fraud-like characteristics |
# | 0.3 - 0.5 |  LOW - Weak similarity to fraud |
# | 0.0 - 0.3 |  VERY LOW - Dissimilar to fraud pattern |

# %%
def compute_cosine_similarities(spark, target_df, centroid_array, centroid_norm):
    
    centroid_broadcast = spark.sparkContext.broadcast(centroid_array)
    centroid_norm_broadcast = spark.sparkContext.broadcast(centroid_norm)
    
    def cosine_sim(features):
        if features is None:
            return 0.0
        
        arr = features.toArray()
        norm = np.linalg.norm(arr)
        c_norm = centroid_norm_broadcast.value
        
        if norm == 0 or c_norm == 0:
            return 0.0
        
        return float(np.dot(arr, centroid_broadcast.value) / (norm * c_norm))
    
    cosine_udf = udf(cosine_sim, DoubleType())
    result_df = target_df.withColumn('cosine_similarity', cosine_udf(col('features')))
    
    return result_df



# %% [markdown]
# ### 2.6 Results Display Functions
# 
# These functions format and display the analysis results:
# - Summary statistics (mean, median, std, min, max)
# - Distribution across similarity ranges
# - Interpretation of results

# %%
def display_query_results(result_df, similarity_threshold=0.7):

    stats = result_df.select(
        count('cosine_similarity').alias('total_count'),
        mean('cosine_similarity').alias('mean_sim'),
        stddev('cosine_similarity').alias('std_sim'),
        spark_min('cosine_similarity').alias('min_sim'),
        spark_max('cosine_similarity').alias('max_sim')
    ).collect()[0]
    
    percentiles = result_df.approxQuantile('cosine_similarity', [0.25, 0.5, 0.75, 0.90, 0.95, 0.99], 0.01)
    
    print(f"\n Summary Statistics:")
    print(f"   Total Transactions Analyzed: {stats['total_count']:,}")
    print(f"   Mean Similarity:    {stats['mean_sim']:.6f}")
    print(f"   Median Similarity:  {percentiles[1]:.6f}")
    print(f"   Std Deviation:      {stats['std_sim']:.6f}")
    print(f"   Min Similarity:     {stats['min_sim']:.6f}")
    print(f"   Max Similarity:     {stats['max_sim']:.6f}")
    
    print(f"\n Percentiles:")
    print(f"   25th percentile:    {percentiles[0]:.6f}")
    print(f"   50th percentile:    {percentiles[1]:.6f}")
    print(f"   75th percentile:    {percentiles[2]:.6f}")
    print(f"   90th percentile:    {percentiles[3]:.6f}")
    print(f"   95th percentile:    {percentiles[4]:.6f}")
    print(f"   99th percentile:    {percentiles[5]:.6f}")
    
    total = stats['total_count']
    
    print(f"\n Threshold Analysis (>= {similarity_threshold}):")
    
    above_threshold = result_df.filter(
        col('cosine_similarity') >= similarity_threshold
    ).count()
    
    below_threshold = total - above_threshold
    
    pct_above = (above_threshold / total * 100) if total > 0 else 0
    pct_below = (below_threshold / total * 100) if total > 0 else 0
    
    bar_above = "█" * int(pct_above / 2)
    bar_below = "█" * int(pct_below / 2)
    
    print(f"   >= {similarity_threshold} (HIGH RISK):   {above_threshold:>10,} ({pct_above:>6.2f}%) {bar_above}")
    print(f"   <  {similarity_threshold} (LOWER RISK): {below_threshold:>10,} ({pct_below:>6.2f}%) {bar_below}")
    
    
    return stats, percentiles, above_threshold


def display_single_transaction_result(result_df, trans_id):
    """
    Display detailed results for a single transaction analysis.
    
    Args:
        result_df: Pandas DataFrame with transaction details
        trans_id: Transaction ID being analyzed
    """
    print(f" COSINE SIMILARITY ANALYSIS FOR TRANSACTION: {trans_id}")
    
    if result_df.empty:
        print(" No results to display")
        return
    
    row = result_df.iloc[0]
    similarity = row['cosine_similarity']
    
    print(f"\n Transaction Details:")
    print(f"   Transaction ID:  {row['trans_id']}")
    print(f"   Amount:          {row['trx_amt']:,.2f}")
    print(f"   Channel:         {row.get('trx_channel', 'N/A')}")
    print(f"   Type:            {row.get('trx_type', 'N/A')}")
    print(f"   Date:            {row.get('cutoff_date', 'N/A')}")
    print(f"   Fraud Flag:      {row.get('fraud_flag', 'N/A')}")
    
    print(f"\n COSINE SIMILARITY TO FRAUD CENTROID: {similarity:.6f}")
    
    # Interpretation
    if similarity >= 0.9:
        print("     VERY HIGH similarity to fraud pattern!")
        interpretation = "This transaction strongly resembles known fraud behavior."
    elif similarity >= 0.7:
        print("     HIGH similarity to fraud pattern")
        interpretation = "This transaction has significant similarities to fraud."
    elif similarity >= 0.5:
        print("    MODERATE similarity to fraud pattern")
        interpretation = "This transaction shows some fraud-like characteristics."
    elif similarity >= 0.3:
        print("     LOW similarity to fraud pattern")
        interpretation = "This transaction has weak similarity to fraud."
    else:
        print("    VERY LOW similarity to fraud pattern")
        interpretation = "This transaction is dissimilar to known fraud patterns."
    
    print(f"\n {interpretation}")



# %% [markdown]
# ### 2.7 Result Saving Functions
# 
# Functions to save results to CSV files for further analysis.

# %%
import glob
import shutil

def save_results_to_csv(result_df, output_path):
    output_dir = output_path.replace('.csv', '_temp')
    
    result_df.select(
        'trans_id', 'cutoff_date', 'trx_amt', 'trx_channel', 'trx_type', 
        'fraud_flag', 'cosine_similarity'
    ).coalesce(1).write.mode('overwrite').option('header', 'true').csv(output_dir)
    
    part_files = glob.glob(os.path.join(output_dir, 'part-*.csv'))
    if part_files:
        shutil.move(part_files[0], output_path)
        shutil.rmtree(output_dir)
        logger.info(f"Results saved to: {output_path}")
    else:
        logger.warning(f"Results saved to directory: {output_dir}")


def get_top_similar_transactions(result_df, n=100):

    return result_df.select(
        'trans_id', 'cutoff_date', 'trx_amt', 'trx_channel', 'trx_type',
        'fraud_flag', 'cosine_similarity'
    ).orderBy(col('cosine_similarity').desc()).limit(n).toPandas()



# %% [markdown]
# ---
# 
# ## Part 3: Query-Based Analysis
# 
# This section analyzes a **batch of transactions** matching a SQL query and calculates their cosine similarity to the fraud centroid.
# 
# ### Use Cases:
# - Analyze all non-fraud transactions in a specific date range
# - Investigate transactions from a specific channel or type
# - Find suspicious transactions that weren't flagged as fraud
# - Compare fraud similarity across different segments
# 
# ### 3.1 Configuration
# 
# Define the parameters for the analysis:

# %%

# Date range for computing the fraud centroid
FRAUD_START_DATE = "2025-01-01"
FRAUD_END_DATE = "2025-06-30"

# SQL WHERE clause for transactions to analyze
QUERY_WHERE_CLAUSE = """
    fraud_flag = 0 
    AND data_date BETWEEN '2025-07-01' AND '2025-07-30' 
"""


SIMILARITY_THRESHOLD = 0.85

OUTPUT_DIR = "/root/research-dir/dev/jazzcash-fraud-detection/analysis/cosine_similarity"
TOP_N_RESULTS = 100 

print(f"   Fraud Centroid Period: {FRAUD_START_DATE} to {FRAUD_END_DATE}")
print(f"   Query: {QUERY_WHERE_CLAUSE.strip()}")
print(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}")
print(f"   Top N Results: {TOP_N_RESULTS}")

# %% [markdown]
# ### 3.2 Load Data
# 
# Load the fraud transactions (for centroid) and query transactions (to analyze).

# %%
fraud_df = load_fraud_transactions(spark, FRAUD_START_DATE, FRAUD_END_DATE)

query_df = load_query_transactions(spark, QUERY_WHERE_CLAUSE)

# %% [markdown]
# ### 3.3 Vectorize and Scale Fraud Transactions
# 
# Vectorize and scale the fraud transactions. The scaler is fit **only on fraud data** to establish the feature space for the centroid.

# %%
fraud_df_scaled, scaler_model = vectorize_and_scale(fraud_df, FEATURE_COLS)



# %% [markdown]
# ### 3.4 Apply Scaler to Query Transactions
# 
# Apply the **same scaler** (fitted on fraud data) to the query transactions.

# %%
query_df_scaled = apply_vectorize_and_scale(query_df, FEATURE_COLS, scaler_model)


# %% [markdown]
# ### 3.5 Calculate Fraud Centroid
# 
# Compute the mean vector of all fraud transactions.

# %%
centroid_array, centroid_norm = calculate_fraud_centroid(fraud_df_scaled)

print(f"\n First 10 centroid feature values:")
for i, (feat, val) in enumerate(zip(FEATURE_COLS[:10], centroid_array[:10])):
    print(f"   {feat}: {val:.6f}")

# %% [markdown]
# ### 3.6 Compute Cosine Similarities
# 
# Calculate the similarity between each query transaction and the fraud centroid.

# %%
result_df = compute_cosine_similarities(spark, query_df_scaled, centroid_array, centroid_norm)

print(f"Computed similarities for {result_df.count():,} transactions")

# %% [markdown]
# ### 3.7 Display Results
# 
# View the comprehensive analysis results including summary statistics and distribution.

# %%
stats, percentiles, high_risk_count = display_query_results(result_df, similarity_threshold=SIMILARITY_THRESHOLD)
stats, percentiles, high_risk_count = display_query_results(result_df, similarity_threshold=0.87)



# %% [markdown]
# ### 3.8 View Top Similar Transactions
# 
# Examine the transactions with highest similarity to fraud patterns.

# %%
top_similar = get_top_similar_transactions(result_df, TOP_N_RESULTS)

display(top_similar.head(20))

# %% [markdown]
# ### 3.9 Save Results (Optional)
# 
# Save the full results to a CSV file for further analysis.

# %%
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = os.path.join(OUTPUT_DIR, f'fraud_centroid_similarity_query_{timestamp}.csv')
# save_results_to_csv(result_df, output_path)
# top_similar.to_csv(os.path.join(OUTPUT_DIR, f'top_{TOP_N_RESULTS}_similar_{timestamp}.csv'), index=False)


# %% [markdown]
# ---
# 
# ## Part 4: Single Transaction Analysis
# 
# This section analyzes a **specific transaction by ID** and calculates its cosine similarity to the fraud centroid.
# 
# ### Use Cases:
# - Investigate a specific flagged transaction
# - Verify if a known fraud case matches the fraud pattern
# - Analyze customer complaints about blocked transactions
# - Deep-dive into edge cases
# 
# ### 4.1 Configuration
# 
# Specify the transaction ID to analyze:

# %%

TRANSACTION_ID = "89942951756"


SINGLE_FRAUD_START_DATE = FRAUD_START_DATE  # "2025-01-01"
SINGLE_FRAUD_END_DATE = FRAUD_END_DATE      # "2025-06-30"

print("Single Transaction Analysis Configuration:")
print(f"   Transaction ID: {TRANSACTION_ID}")
print(f"   Fraud Centroid Period: {SINGLE_FRAUD_START_DATE} to {SINGLE_FRAUD_END_DATE}")

# %% [markdown]
# ### 4.2 Load Transaction Data
# 
# Load the specific transaction and fraud transactions for centroid.

# %%
single_txn_df = load_single_transaction(spark, TRANSACTION_ID)

if single_txn_df is None:
    print("Transaction not found")
else:
    print("\n Transaction Preview:")
    single_txn_df.select('trans_id', 'trx_amt', 'trx_channel', 'trx_type', 'cutoff_date', 'fraud_flag').show()

# %% [markdown]
# ### 4.3 Load Fraud Transactions for Centroid
# 
# If we haven't already loaded fraud transactions (or want to use a different date range), load them here.

# %%



try:
    _ = fraud_df.count()  # Check if fraud_df exists
    single_fraud_df = fraud_df
except:
    print("Loading fraud transactions...")
    single_fraud_df = load_fraud_transactions(spark, SINGLE_FRAUD_START_DATE, SINGLE_FRAUD_END_DATE)

# %% [markdown]
# ### 4.4 Vectorize and Scale Features
# 
# Vectorize and scale the fraud transactions, then apply the same scaler to the single transaction.

# %%
if single_txn_df is not None:
    
    single_fraud_df_scaled, single_scaler_model = vectorize_and_scale(single_fraud_df, FEATURE_COLS)
    
    single_txn_df_scaled = apply_vectorize_and_scale(single_txn_df, FEATURE_COLS, single_scaler_model)
    


# %% [markdown]
# ### 4.5 Calculate Fraud Centroid
# 
# Compute the centroid using the fraud transactions.

# %%
if single_txn_df is not None:
    single_centroid_array, single_centroid_norm = calculate_fraud_centroid(single_fraud_df_scaled)

# %% [markdown]
# ### 4.6 Compute Cosine Similarity
# 
# Calculate the similarity between the single transaction and fraud centroid.

# %%
if single_txn_df is not None:
    single_result_df = compute_cosine_similarities(
        spark, single_txn_df_scaled, single_centroid_array, single_centroid_norm
    )
    
    # Convert to Pandas for display
    single_result_pd = single_result_df.toPandas()
    


# %% [markdown]
# ### 4.7 Display Results
# 
# View the detailed analysis results for the single transaction.

# %%
if single_txn_df is not None:
    display_single_transaction_result(single_result_pd, TRANSACTION_ID)

# %% [markdown]
# ### 4.8 Feature Contribution Analysis (Optional)
# 
# Examine which features contribute most to the similarity score.

# %%
if single_txn_df is not None:
    txn_features = single_txn_df_scaled.select('features').collect()[0]['features'].toArray()
    
    contributions = txn_features * single_centroid_array
    
    feature_contrib_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'txn_value': txn_features,
        'centroid_value': single_centroid_array,
        'contribution': contributions
    })
    
    feature_contrib_df['abs_contribution'] = np.abs(feature_contrib_df['contribution'])
    feature_contrib_df = feature_contrib_df.sort_values('abs_contribution', ascending=False)
    
    display(feature_contrib_df.head(15)[['feature', 'txn_value', 'centroid_value', 'contribution']])

# %% [markdown]
# ---
# 
# ## Part 5: Cleanup
# 
# Stop the Spark session when analysis is complete to free up cluster resources.

# %%
# spark.stop()



# %% [markdown]
# ---
# 
# ## Appendix: Quick Reference
# 
# ### Common Query Examples
# 
# | Use Case | Query WHERE Clause |
# |----------|-------------------|
# | Non-fraud in September | `fraud_flag = 0 AND cutoff_date >= '2025-09-01' AND cutoff_date <= '2025-09-30'` |
# | High-value transfers | `trx_amt > 100000 AND fraud_flag = 0` |
# | NEW_JC_APP C2B transfers | `trx_channel = 'NEW_JC_APP' AND trx_type = 'Transfer(C2B)'` |
# | Night transactions | `is_night = 1 AND fraud_flag = 0` |
# | Verify fraud patterns | `fraud_flag = 1 AND cutoff_date >= '2025-07-01'` |
# | Morning transactions (6AM-12PM) | `fraud_flag = 0 AND hour(trans_initiate_time) BETWEEN 6 AND 12` |
# 
# ### Similarity Interpretation Guide
# 
# | Similarity Score | Risk Level | Recommended Action |
# |-----------------|------------|-------------------|
# | 0.90 - 1.00 |  Critical | Immediate investigation required |
# | 0.70 - 0.90 |  High | Priority review |
# | 0.50 - 0.70 |  Moderate | Monitor closely |
# | 0.30 - 0.50 |  Low | Standard processing |
# | 0.00 - 0.30 |  Very Low | No action needed |
# 
# ### Feature Categories
# 
# 1. **Transaction Features**: `start_balance`, `trx_amt`
# 2. **Temporal Features**: `hour_of_day`, `day_of_week`, `is_weekend`, `is_night`, `is_business_hours`, `is_unusual_hour`, `night_weekend_combo`
# 3. **3-Day Transaction Aggregates**: `txn_txns_3d`, `txn_total_amount_3d`, `txn_avg_amount_3d`, `txn_max_amount_3d`, `txn_min_amount_3d`, etc.
# 4. **Channel Indicators**: `channel_new_jc_app`, `channel_ussd`, `channel_ussd_api`, `channel_payment_gateway`, `channel_mobile_app`
# 5. **Type Indicators**: `type_transfer_c2c`, `type_transfer_c2b`, `type_bill_payment`, `type_mobile_load`
# 6. **User Aggregates**: `user_total_txns_3d`, `user_total_amount_3d`, `user_avg_amount_3d`, `user_total_txns_7d`, etc.

# %%



