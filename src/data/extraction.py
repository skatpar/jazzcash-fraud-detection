"""
Data extraction module for fraud detection pipeline
Handles loading data from PostgreSQL database
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, broadcast
from typing import Optional, List, Tuple
from src.utils.spark_utils import (
    read_table_from_postgres,
    create_date_predicates,
    write_parquet,
    load_config,
    log_df_stats
)
from src.utils.logger import get_logger

logger = get_logger("data_extraction")


class DataExtractor:
    """Extract data from PostgreSQL database"""

    def __init__(
        self,
        spark: SparkSession,
        db_config_path: str = "config/db_config.yaml"
    ):
        """
        Initialize DataExtractor

        Args:
            spark: Spark session
            db_config_path: Path to database configuration
        """
        self.spark = spark
        self.db_config = load_config(db_config_path)
        self.processing_mode = self.db_config['processing_mode']['current']

        logger.info(f"Initialized DataExtractor with mode: {self.processing_mode}")

    def extract_iar_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sample: bool = False
    ) -> DataFrame:
        """
        Extract IAR (transaction) data

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_sample: Whether to use sample data

        Returns:
            DataFrame: IAR data
        """
        logger.info("Extracting IAR data...")

        # Determine which table to use
        if use_sample or self.processing_mode == 'sample':
            table_config = self.db_config['tables']['iar_sample']
            logger.info(f"Using sample table: {table_config['name']}")

            df = read_table_from_postgres(
                self.spark,
                table_config['name']
            )

        else:
            # Use date-based table
            mode_config = self.db_config['processing_mode'][self.processing_mode]
            table_name = mode_config['table']
            table_config = self.db_config['tables'].get(
                'iar_july' if 'jul' in table_name else 'iar_main'
            )

            # Get date range
            if not start_date:
                start_date = mode_config['date_range'][0]
            if not end_date:
                end_date = mode_config['date_range'][1]

            logger.info(f"Using table: {table_name}")
            logger.info(f"Date range: {start_date} to {end_date}")

            # Create predicates for parallel loading
            num_partitions = table_config.get('partitions', 100)
            predicates = create_date_predicates(
                'data_date',
                start_date,
                end_date,
                num_partitions
            )

            logger.info(f"Created {len(predicates)} predicates for parallel loading")

            df = read_table_from_postgres(
                self.spark,
                f"public.{table_name}",
                predicates=predicates
            )

        log_df_stats(df, "IAR Data", logger)
        return df

    def extract_mbar_data(self) -> DataFrame:
        """
        Extract MBAR (account/customer) data

        Returns:
            DataFrame: MBAR data
        """
        logger.info("Extracting MBAR data...")

        table_config = self.db_config['tables']['mbar']
        table_name = table_config['name']

        # Create predicates based on account reference ranges
        # This helps with parallel loading
        num_partitions = table_config.get('partitions', 50)

        df = read_table_from_postgres(
            self.spark,
            table_name
        )

        # Repartition for better performance
        df = df.repartition(num_partitions)

        log_df_stats(df, "MBAR Data", logger)
        return df

    def extract_fraud_data(
        self,
        july_only: bool = False
    ) -> DataFrame:
        """
        Extract fraud cases data

        Args:
            july_only: Whether to filter for July 2025 only

        Returns:
            DataFrame: Fraud data
        """
        logger.info("Extracting fraud data...")

        table_config = self.db_config['tables']['fraud']
        table_name = table_config['name']

        df = read_table_from_postgres(
            self.spark,
            table_name
        )

        if july_only:
            logger.info("Filtering for July 2025 fraud cases")
            df = df.filter(
                (col("transaction_datetime") >= "2025-07-01") &
                (col("transaction_datetime") < "2025-08-01")
            )

        log_df_stats(df, "Fraud Data", logger)
        return df

    def combine_iar_mbar(
        self,
        df_iar: DataFrame,
        df_mbar: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Combine IAR with MBAR data for both sender and receiver

        Args:
            df_iar: IAR DataFrame
            df_mbar: MBAR DataFrame

        Returns:
            Tuple of (df_with_sender_info, df_with_both_info)
        """
        logger.info("Combining IAR with MBAR data...")

        # Cache MBAR for multiple joins
        df_mbar = df_mbar.cache()

        # Join 1: Add sender (ac_from) information
        logger.info("Joining sender account information...")
        df_mbar_from = df_mbar.selectExpr(
            "a_c_reference as ac_from",
            *[f"{col} as from_{col}" for col in df_mbar.columns if col != 'a_c_reference']
        )

        df_with_sender = df_iar.join(
            broadcast(df_mbar_from) if df_mbar_from.count() < 100000000 else df_mbar_from,
            on="ac_from",
            how="left"
        )

        log_df_stats(df_with_sender, "IAR with Sender Info", logger)

        # Join 2: Add receiver (ac_to) information
        logger.info("Joining receiver account information...")
        df_mbar_to = df_mbar.selectExpr(
            "a_c_reference as ac_to",
            *[f"{col} as to_{col}" for col in df_mbar.columns if col != 'a_c_reference']
        )

        df_with_both = df_with_sender.join(
            broadcast(df_mbar_to) if df_mbar_to.count() < 100000000 else df_mbar_to,
            on="ac_to",
            how="left"
        )

        log_df_stats(df_with_both, "IAR with Both Account Info", logger)

        # Unpersist MBAR
        df_mbar.unpersist()

        return df_with_sender, df_with_both

    def add_fraud_labels(
        self,
        df_transactions: DataFrame,
        df_fraud: DataFrame
    ) -> DataFrame:
        """
        Add fraud labels to transaction data

        Args:
            df_transactions: Transaction DataFrame
            df_fraud: Fraud DataFrame

        Returns:
            DataFrame: Transactions with fraud labels
        """
        logger.info("Adding fraud labels...")

        # Select relevant fraud columns
        df_fraud_labeled = df_fraud.select(
            "trans_id",
            "complaint_num",
            "victim_msisdn",
            "fraud_msisdn"
        ).withColumn("is_fraud", col("trans_id").isNotNull().cast("int"))

        # Join with transactions
        df_labeled = df_transactions.join(
            broadcast(df_fraud_labeled),
            on="trans_id",
            how="left"
        )

        # Fill null fraud labels with 0
        from pyspark.sql.functions import when, coalesce, lit

        df_labeled = df_labeled.withColumn(
            "is_fraud",
            coalesce(col("is_fraud"), lit(0))
        )

        # Log fraud statistics
        fraud_count = df_labeled.filter(col("is_fraud") == 1).count()
        total_count = df_labeled.count()
        fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0

        logger.info(f"Total transactions: {total_count:,}")
        logger.info(f"Fraud transactions: {fraud_count:,}")
        logger.info(f"Fraud rate: {fraud_rate:.4f}%")

        return df_labeled

    def extract_and_combine_all(
        self,
        use_sample: bool = False,
        save_intermediate: bool = True
    ) -> DataFrame:
        """
        Extract and combine all data sources

        Args:
            use_sample: Whether to use sample data
            save_intermediate: Whether to save intermediate results

        Returns:
            DataFrame: Master dataset with all information
        """
        logger.info("Starting full data extraction and combination...")

        # Step 1: Extract IAR data
        df_iar = self.extract_iar_data(use_sample=use_sample)

        if save_intermediate:
            logger.info("Saving raw IAR data...")
            write_parquet(
                df_iar,
                "data/raw/iar_data.parquet",
                partition_by=["data_date"]
            )

        # Step 2: Extract MBAR data
        df_mbar = self.extract_mbar_data()

        if save_intermediate:
            logger.info("Saving raw MBAR data...")
            write_parquet(
                df_mbar,
                "data/raw/mbar_data.parquet",
                coalesce=10
            )

        # Step 3: Extract fraud data
        df_fraud = self.extract_fraud_data(july_only=(self.processing_mode == 'july'))

        if save_intermediate:
            logger.info("Saving raw fraud data...")
            write_parquet(
                df_fraud,
                "data/raw/fraud_data.parquet",
                coalesce=1
            )

        # Step 4: Combine IAR with MBAR
        _, df_combined = self.combine_iar_mbar(df_iar, df_mbar)

        if save_intermediate:
            logger.info("Saving combined IAR+MBAR data...")
            write_parquet(
                df_combined,
                "data/raw/iar_mbar_combined.parquet",
                partition_by=["data_date"],
                coalesce=50
            )

        # Step 5: Add fraud labels
        df_master = self.add_fraud_labels(df_combined, df_fraud)

        logger.info("Data extraction and combination complete!")
        return df_master


def main():
    """Main function for testing"""
    from src.utils.spark_utils import create_spark_session

    # Create Spark session
    spark = create_spark_session()

    # Initialize extractor
    extractor = DataExtractor(spark)

    # Extract and combine all data
    df_master = extractor.extract_and_combine_all(use_sample=True)

    # Save master dataset
    logger.info("Saving master dataset...")
    write_parquet(
        df_master,
        "data/processed/master_dataset.parquet",
        partition_by=["data_date", "is_fraud"]
    )

    logger.info("Extraction complete!")

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()
