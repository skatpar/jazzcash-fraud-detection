"""
Temporal feature engineering
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, hour, dayofweek, dayofmonth, month, when, lit
)
from src.utils.logger import get_logger

logger = get_logger("temporal_features")


def create_temporal_features(df: DataFrame) -> DataFrame:
    """
    Create temporal features

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with temporal features
    """
    logger.info("Creating temporal features...")

    # Hour-based features
    df = df.withColumn("trx_hour", hour("trans_initiate_time"))
    df = df.withColumn("trx_day_of_week", dayofweek("trans_initiate_time"))
    df = df.withColumn("trx_day_of_month", dayofmonth("trans_initiate_time"))
    df = df.withColumn("trx_month", month("trans_initiate_time"))

    # Weekend indicator
    df = df.withColumn(
        "is_weekend",
        col("trx_day_of_week").isin([1, 7]).cast("int")
    )

    # Business hours (9 AM - 5 PM on weekdays)
    df = df.withColumn(
        "is_business_hours",
        ((col("trx_hour") >= 9) & (col("trx_hour") <= 17) & ~col("is_weekend")).cast("int")
    )

    # Odd hours (midnight to 5 AM)
    df = df.withColumn(
        "is_odd_hour",
        ((col("trx_hour") >= 0) & (col("trx_hour") <= 5)).cast("int")
    )

    # Time buckets
    df = df.withColumn(
        "time_bucket",
        when((col("trx_hour") >= 0) & (col("trx_hour") < 6), "night")
        .when((col("trx_hour") >= 6) & (col("trx_hour") < 12), "morning")
        .when((col("trx_hour") >= 12) & (col("trx_hour") < 18), "afternoon")
        .otherwise("evening")
    )

    # Peak hours (8-10 AM, 6-8 PM)
    df = df.withColumn(
        "is_peak_hour",
        (((col("trx_hour") >= 8) & (col("trx_hour") <= 10)) |
         ((col("trx_hour") >= 18) & (col("trx_hour") <= 20))).cast("int")
    )

    logger.info("Temporal features created")
    return df
