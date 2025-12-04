#!/usr/bin/env python3
"""
Cosine Similarity Fraud Analysis + t-SNE Visualization
=====================================================
Loads data from ClickHouse using Spark, calculates cosine similarity,
runs t-SNE, and visualizes clusters in 3D, highlighting high-similarity non-fraud transactions.

Usage:
    python cosine_similarity_fraud_analysis_tsne.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --top-n 1000
"""

import os
import argparse
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Summarizer
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def initialize_spark():
    os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/envs/fraud-spark/bin/python'
    packages = [
        "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
        "com.clickhouse:clickhouse-client:0.9.4",
        "com.clickhouse:clickhouse-http-client:0.9.4",
        "org.apache.httpcomponents.client5:httpclient5:5.2.1"
    ]
    spark = (SparkSession.builder
        .appName("CosineSimilarityFraudAnalysis")
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
    spark.conf.set("spark.clickhouse.write.format", "json")
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_data(spark, start_date, end_date):
    query = f"""
    SELECT *
    FROM clickhouse.public.stixor_fraud_features_distributed
    WHERE (cutoff_date >= '{start_date}' AND cutoff_date <= '{end_date}'
    AND mbar_account_type_name = 'Customer Account') OR fraud_flag = 1
    """
    return spark.sql(query)

def prepare_feature_vectors(df):
    feature_cols = [
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
        'type_mobile_load', 'user_total_txns_3d', 'user_total_amount_3d',
        'user_avg_amount_3d', 'user_max_amount_3d', 'user_unique_recipients_3d',
        'user_unique_channels_3d', 'user_total_txns_7d', 'user_avg_amount_7d',
        'user_max_amount_7d', 'user_night_txns_7d', 'user_weekend_txns_7d'
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip")
    df = assembler.transform(df)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    return df, feature_cols

def calculate_cosine_similarity(df):
    fraud_raw = df.filter(col('fraud_flag') == 1).select('features_raw')

    non_fraud_df = df.filter(col('fraud_flag') == 0).limit(100000)

    fraud_centroid = fraud_raw.select(Summarizer.mean(col('features_raw')).alias('centroid')).collect()[0]['centroid']
    centroid_array = fraud_centroid.toArray()
    centroid_norm = float(np.linalg.norm(centroid_array))
    def cosine_with_centroid(features):
        if features is None:
            return 0.0
        arr = features.toArray()
        norm = np.linalg.norm(arr)
        if norm == 0 or centroid_norm == 0:
            return 0.0
        dot_product = np.dot(arr, centroid_array)
        return float(dot_product / (norm * centroid_norm))
    cosine_udf = F.udf(cosine_with_centroid, DoubleType())
    df = df.withColumn('cosine_similarity', cosine_udf(col('features_raw')))
    return df

def main():
    parser = argparse.ArgumentParser(description='Fraud t-SNE visualization from ClickHouse via Spark')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=1000, help='Number of top similar transactions to plot')
    args = parser.parse_args()

    spark = initialize_spark()
    df = load_data(spark, args.start_date, args.end_date)
    df, feature_cols = prepare_feature_vectors(df)
    df = calculate_cosine_similarity(df)
    pdf = df.select(
        'fraud_flag', 'cosine_similarity', *feature_cols
    ).toPandas()
    # For visualization, sample top-N by similarity
    pdf = pdf.sort_values('cosine_similarity', ascending=False).head(args.top_n)
    X = pdf[feature_cols].values
    fraud_idx = pdf['fraud_flag'] == 1
    nonfraud_idx = pdf['fraud_flag'] == 0
    similarity = pdf['cosine_similarity'].values
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)
    pdf['tsne_x'] = X_embedded[:,0]
    pdf['tsne_y'] = X_embedded[:,1]
    pdf['tsne_z'] = X_embedded[:,2]
    sns.set(style='whitegrid', context='notebook')
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    sim_norm = (similarity - similarity.min()) / (similarity.max() - similarity.min())
    sizes = 20 + sim_norm * 80
    nonfraud = pdf[nonfraud_idx]
    ax.scatter(nonfraud['tsne_x'], nonfraud['tsne_y'], nonfraud['tsne_z'],
               c='dodgerblue', s=sizes[nonfraud_idx], alpha=0.5, label='Non-Fraud', edgecolor='w')
    fraud = pdf[fraud_idx]
    ax.scatter(fraud['tsne_x'], fraud['tsne_y'], fraud['tsne_z'],
               c='crimson', s=sizes[fraud_idx], alpha=0.7, label='Fraud', edgecolor='k')
    high_sim = nonfraud[nonfraud['cosine_similarity'] > 0.8]
    ax.scatter(high_sim['tsne_x'], high_sim['tsne_y'], high_sim['tsne_z'],
               c='gold', s=120, alpha=0.9, label='High-Sim Non-Fraud', edgecolor='black')
    ax.set_xlabel('t-SNE X')
    ax.set_ylabel('t-SNE Y')
    ax.set_zlabel('t-SNE Z')
    ax.set_title('Fraud & Non-Fraud Transaction Clusters (t-SNE 3D)')
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()
    spark.stop()

if __name__ == '__main__':
    main()
