"""
Transaction-level feature engineering
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, log, when, coalesce, lit, count, sum as spark_sum,
    avg, max as spark_max, min as spark_min, stddev, approx_count_distinct,
    datediff, unix_timestamp, lag, concat, expr
)
from src.utils.logger import get_logger
from src.utils.spark_utils import load_config

logger = get_logger("transaction_features")


class TransactionFeatureEngineer:
    """Generate transaction-level features"""

    def __init__(self, config_path: str = "config/feature_config.yaml"):
        """
        Initialize feature engineer

        Args:
            config_path: Path to feature configuration
        """
        self.config = load_config(config_path)
        self.time_windows = self.config.get('time_windows', {}).get('all', [1, 3, 7, 15, 30])
        logger.info(f"Initialized with time windows: {self.time_windows}")

    def create_basic_features(self, df: DataFrame) -> DataFrame:
        """
        Create basic transaction features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with basic features
        """
        logger.info("Creating basic transaction features...")

        # Amount-based features
        df = df.withColumn("log_trx_amt", log(col("trx_amt") + 1))
        df = df.withColumn("total_cost", col("trx_amt") + col("fee") + col("fed"))
        df = df.withColumn("fee_ratio", col("fee") / (col("trx_amt") + 0.01))
        df = df.withColumn("fed_ratio", col("fed") / (col("trx_amt") + 0.01))

        # Balance-related features
        df = df.withColumn(
            "balance_change_pct",
            (col("end_balance") - col("start_balance")) / (col("start_balance") + 1)
        )
        df = df.withColumn(
            "balance_to_amount_ratio",
            col("start_balance") / (col("trx_amt") + 1)
        )

        # Round amount indicators
        df = df.withColumn("is_round_amount", (col("trx_amt") % 1000 == 0).cast("int"))
        df = df.withColumn("is_round_100", (col("trx_amt") % 100 == 0).cast("int"))
        df = df.withColumn("is_round_50", (col("trx_amt") % 50 == 0).cast("int"))

        # Amount buckets
        df = df.withColumn(
            "amount_bucket",
            when(col("trx_amt") < 500, "very_low")
            .when(col("trx_amt") < 2000, "low")
            .when(col("trx_amt") < 10000, "medium")
            .when(col("trx_amt") < 50000, "high")
            .otherwise("very_high")
        )

        logger.info("Basic transaction features created")
        return df

    def create_aggregation_features(self, df: DataFrame) -> DataFrame:
        """
        Create time-window aggregation features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with aggregation features
        """
        logger.info("Creating aggregation features (this may take a while)...")

        # Create features for sender account (ac_from)
        df = self._create_sender_aggregations(df)

        # Create features for receiver account (ac_to)
        df = self._create_receiver_aggregations(df)

        logger.info("Aggregation features created")
        return df

    def _create_sender_aggregations(self, df: DataFrame) -> DataFrame:
        """Create aggregation features for sender account"""
        logger.info("Creating sender aggregation features...")

        for window_days in self.time_windows:
            logger.debug(f"Processing {window_days}-day window...")

            # Define window specification
            window_spec = Window \
                .partitionBy("ac_from") \
                .orderBy(col("trans_initiate_time").cast("long")) \
                .rangeBetween(-window_days * 86400, -1)

            # Transaction count
            df = df.withColumn(
                f"from_tx_count_{window_days}d",
                coalesce(count("trans_id").over(window_spec), lit(0))
            )

            # Total amount
            df = df.withColumn(
                f"from_total_amt_{window_days}d",
                coalesce(spark_sum("trx_amt").over(window_spec), lit(0.0))
            )

            # Average amount
            df = df.withColumn(
                f"from_avg_amt_{window_days}d",
                coalesce(avg("trx_amt").over(window_spec), lit(0.0))
            )

            # Max and min amounts
            df = df.withColumn(
                f"from_max_amt_{window_days}d",
                coalesce(spark_max("trx_amt").over(window_spec), lit(0.0))
            )
            df = df.withColumn(
                f"from_min_amt_{window_days}d",
                coalesce(spark_min("trx_amt").over(window_spec), lit(0.0))
            )

            # Standard deviation
            df = df.withColumn(
                f"from_std_amt_{window_days}d",
                coalesce(stddev("trx_amt").over(window_spec), lit(0.0))
            )

            # Unique recipients
            df = df.withColumn(
                f"from_unique_recipients_{window_days}d",
                coalesce(approx_count_distinct("ac_to").over(window_spec), lit(0))
            )

            # Unique channels
            df = df.withColumn(
                f"from_unique_channels_{window_days}d",
                coalesce(approx_count_distinct("trx_channel").over(window_spec), lit(0))
            )

            # Unique transaction types
            df = df.withColumn(
                f"from_unique_types_{window_days}d",
                coalesce(approx_count_distinct("trx_type").over(window_spec), lit(0))
            )

        # Velocity features (comparing short vs long term)
        if 1 in self.time_windows and 7 in self.time_windows:
            df = df.withColumn(
                "velocity_1d_to_7d",
                col("from_tx_count_1d") / (col("from_tx_count_7d") + 1)
            )
            df = df.withColumn(
                "amount_acceleration",
                col("from_avg_amt_1d") / (col("from_avg_amt_7d") + 1)
            )

        # Burst detection
        if 1 in self.time_windows and 7 in self.time_windows:
            df = df.withColumn(
                "is_activity_burst",
                ((col("from_tx_count_1d") > 2 * col("from_tx_count_7d")) &
                 (col("from_tx_count_1d") > 5)).cast("int")
            )

        logger.info("Sender aggregation features created")
        return df

    def _create_receiver_aggregations(self, df: DataFrame) -> DataFrame:
        """Create aggregation features for receiver account"""
        logger.info("Creating receiver aggregation features...")

        for window_days in self.time_windows:
            window_spec = Window \
                .partitionBy("ac_to") \
                .orderBy(col("trans_initiate_time").cast("long")) \
                .rangeBetween(-window_days * 86400, -1)

            # Transaction count
            df = df.withColumn(
                f"to_tx_count_{window_days}d",
                coalesce(count("trans_id").over(window_spec), lit(0))
            )

            # Total amount received
            df = df.withColumn(
                f"to_total_amt_{window_days}d",
                coalesce(spark_sum("trx_amt").over(window_spec), lit(0.0))
            )

            # Average amount received
            df = df.withColumn(
                f"to_avg_amt_{window_days}d",
                coalesce(avg("trx_amt").over(window_spec), lit(0.0))
            )

            # Unique senders
            df = df.withColumn(
                f"to_unique_senders_{window_days}d",
                coalesce(approx_count_distinct("ac_from").over(window_spec), lit(0))
            )

        logger.info("Receiver aggregation features created")
        return df

    def create_interaction_features(self, df: DataFrame) -> DataFrame:
        """
        Create interaction features between sender and receiver

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")

        # Count interactions between specific sender-receiver pairs
        window_pair = Window.partitionBy("ac_from", "ac_to")

        df = df.withColumn(
            "from_to_interaction_count",
            count("trans_id").over(window_pair)
        )

        df = df.withColumn(
            "is_frequent_pair",
            (col("from_to_interaction_count") > 5).cast("int")
        )

        df = df.withColumn(
            "is_first_interaction",
            (col("from_to_interaction_count") == 1).cast("int")
        )

        # Amount comparison features
        if "from_avg_amt_7d" in df.columns:
            df = df.withColumn(
                "amt_vs_sender_avg_ratio",
                col("trx_amt") / (col("from_avg_amt_7d") + 1)
            )

        if "to_avg_amt_7d" in df.columns:
            df = df.withColumn(
                "amt_vs_receiver_avg_ratio",
                col("trx_amt") / (col("to_avg_amt_7d") + 1)
            )

        logger.info("Interaction features created")
        return df

    def create_all_features(self, df: DataFrame) -> DataFrame:
        """
        Create all transaction features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all transaction features
        """
        logger.info("Creating all transaction features...")

        df = self.create_basic_features(df)
        df = self.create_aggregation_features(df)
        df = self.create_interaction_features(df)

        logger.info("All transaction features created")
        return df


def main():
    """Main function for testing"""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("TransactionFeatures") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # Load cleaned data
    df = spark.read.parquet("data/processed/cleaned_dataset.parquet")

    # Create features
    feature_engineer = TransactionFeatureEngineer()
    df_features = feature_engineer.create_all_features(df)

    # Save
    df_features.write.mode("overwrite") \
        .partitionBy("data_date") \
        .parquet("data/features/transaction_features.parquet")

    spark.stop()


if __name__ == "__main__":
    main()
