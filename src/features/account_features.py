"""
Account-level feature engineering
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, datediff, year, when, coalesce, lit, count, broadcast
)
from src.utils.logger import get_logger

logger = get_logger("account_features")


def create_account_features(df: DataFrame) -> DataFrame:
    """
    Create account profile features

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with account features
    """
    logger.info("Creating account features...")

    # Sender account features
    if "from_registered_date_time" in df.columns:
        df = df.withColumn(
            "from_account_age_days",
            datediff(col("data_date"), col("from_registered_date_time"))
        )
        df = df.withColumn(
            "is_from_new_account",
            (col("from_account_age_days") < 30).cast("int")
        )

    if "from_last_modified_date_time" in df.columns:
        df = df.withColumn(
            "from_days_inactive",
            datediff(col("data_date"), col("from_last_modified_date_time"))
        )

    if "from_year_of_birth" in df.columns:
        df = df.withColumn(
            "from_customer_age",
            year(col("data_date")) - col("from_year_of_birth")
        )
        df = df.withColumn(
            "is_from_young_customer",
            (col("from_customer_age") < 25).cast("int")
        )
        df = df.withColumn(
            "is_from_senior_customer",
            (col("from_customer_age") > 60).cast("int")
        )

    # Dormancy indicators
    if "from_dormant_date" in df.columns:
        df = df.withColumn(
            "from_is_dormant",
            col("from_dormant_date").isNotNull().cast("int")
        )

    if "from_re_active_date" in df.columns:
        df = df.withColumn(
            "from_is_reactivated",
            col("from_re_active_date").isNotNull().cast("int")
        )

    # Trust and security features
    if "from_trust_level" in df.columns:
        df = df.withColumn(
            "from_trust_score",
            when(col("from_trust_level") == "HIGH", 3)
            .when(col("from_trust_level") == "MEDIUM", 2)
            .when(col("from_trust_level") == "LOW", 1)
            .otherwise(0)
        )

    if "from_mpin_status" in df.columns:
        df = df.withColumn(
            "from_has_mpin",
            (col("from_mpin_status") == "ACTIVE").cast("int")
        )

    if "from_filer" in df.columns:
        df = df.withColumn(
            "from_is_filer",
            (col("from_filer") == "Y").cast("int")
        )

    # Receiver account features (similar)
    if "to_registered_date_time" in df.columns:
        df = df.withColumn(
            "to_account_age_days",
            datediff(col("data_date"), col("to_registered_date_time"))
        )
        df = df.withColumn(
            "is_to_new_account",
            (col("to_account_age_days") < 30).cast("int")
        )

    if "to_year_of_birth" in df.columns:
        df = df.withColumn(
            "to_customer_age",
            year(col("data_date")) - col("to_year_of_birth")
        )

    # Geographic features
    if "from_city" in df.columns and "to_city" in df.columns:
        df = df.withColumn(
            "is_same_city",
            (col("from_city") == col("to_city")).cast("int")
        )

    if "from_region" in df.columns and "to_region" in df.columns:
        df = df.withColumn(
            "is_same_region",
            (col("from_region") == col("to_region")).cast("int")
        )

    logger.info("Account features created")
    return df


def create_categorical_encoding(df: DataFrame) -> DataFrame:
    """
    Create frequency and target encoding for categorical features

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with encoded features
    """
    logger.info("Creating categorical encodings...")

    # Frequency encoding for channels
    channel_freq = df.groupBy("trx_channel").agg(
        count("*").alias("channel_count")
    )
    total_count = df.count()
    channel_freq = channel_freq.withColumn(
        "channel_frequency",
        col("channel_count") / total_count
    )

    df = df.join(broadcast(channel_freq), "trx_channel", "left")

    # Frequency encoding for transaction type
    type_freq = df.groupBy("trx_type").agg(
        count("*").alias("type_count")
    )
    type_freq = type_freq.withColumn(
        "type_frequency",
        col("type_count") / total_count
    )

    df = df.join(broadcast(type_freq), "trx_type", "left")

    # Rare category flags
    df = df.withColumn(
        "is_rare_channel",
        (col("channel_frequency") < 0.01).cast("int")
    )
    df = df.withColumn(
        "is_rare_type",
        (col("type_frequency") < 0.01).cast("int")
    )

    # Channel/type specific flags
    df = df.withColumn(
        "is_new_jc_app",
        (col("trx_channel") == "NEW_JC_APP").cast("int")
    )
    df = df.withColumn(
        "is_payment_gateway",
        (col("trx_channel") == "PAYMENT GATEWAY").cast("int")
    )
    df = df.withColumn(
        "is_c2c_transfer",
        col("trx_type").contains("C2C").cast("int")
    )
    df = df.withColumn(
        "is_c2b_transfer",
        col("trx_type").contains("C2B").cast("int")
    )

    logger.info("Categorical encodings created")
    return df
