"""
Data cleaning module for fraud detection pipeline
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, when, coalesce, lit, mean, stddev, expr,
    abs as spark_abs, count, approxQuantile, trim, upper,
    isnull, isnan, lag, unix_timestamp, hour, dayofweek,
    to_date, to_timestamp, datediff
)
from typing import List, Dict, Optional
from src.utils.spark_utils import load_config, write_parquet
from src.utils.logger import get_logger

logger = get_logger("data_cleaning")


class DataCleaner:
    """Clean and preprocess fraud detection data"""

    def __init__(
        self,
        feature_config_path: str = "config/feature_config.yaml"
    ):
        """
        Initialize DataCleaner

        Args:
            feature_config_path: Path to feature configuration
        """
        self.config = load_config(feature_config_path)
        self.missing_config = self.config.get('missing_values', {})
        self.outlier_config = self.config.get('outliers', {})
        self.quality_config = self.config.get('data_quality', {})

        logger.info("Initialized DataCleaner")

    def remove_duplicates(self, df: DataFrame) -> DataFrame:
        """
        Remove duplicate transactions

        Args:
            df: Input DataFrame

        Returns:
            DataFrame: Deduplicated data
        """
        logger.info("Removing duplicates...")

        initial_count = df.count()
        df_dedup = df.dropDuplicates(["trans_id"])
        final_count = df_dedup.count()

        duplicates_removed = initial_count - final_count
        logger.info(f"Removed {duplicates_removed:,} duplicate transactions")

        return df_dedup

    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values based on configuration

        Args:
            df: Input DataFrame

        Returns:
            DataFrame: Data with missing values handled
        """
        logger.info("Handling missing values...")

        # Log missing value statistics
        self._log_missing_stats(df)

        # Handle numerical columns
        num_config = self.missing_config.get('numerical', {})
        default_strategy = num_config.get('default_strategy', 'median')

        # Get numerical columns
        numerical_cols = [
            'trx_amt', 'fee', 'fed', 'start_balance', 'end_balance',
            'from_year_of_birth', 'to_year_of_birth'
        ]

        for col_name in numerical_cols:
            if col_name in df.columns:
                df = self._impute_numerical(df, col_name, default_strategy)

        # Handle categorical columns
        cat_config = self.missing_config.get('categorical', {})
        fill_value = cat_config.get('fill_value', 'UNKNOWN')

        categorical_cols = [
            'from_city', 'to_city', 'from_region', 'to_region',
            'from_a_c_status', 'to_a_c_status', 'from_account_type_name',
            'to_account_type_name', 'utility_company'
        ]

        for col_name in categorical_cols:
            if col_name in df.columns:
                df = df.fillna({col_name: fill_value})

        # Create missing indicators
        if self.missing_config.get('create_indicators', True):
            indicator_cols = self.missing_config.get('indicator_columns', [])
            for col_name in indicator_cols:
                if col_name in df.columns:
                    indicator_name = f"is_{col_name}_missing"
                    df = df.withColumn(
                        indicator_name,
                        col(col_name).isNull().cast("int")
                    )

        logger.info("Missing value handling complete")
        return df

    def _impute_numerical(
        self,
        df: DataFrame,
        col_name: str,
        strategy: str = 'median'
    ) -> DataFrame:
        """
        Impute numerical column

        Args:
            df: DataFrame
            col_name: Column name
            strategy: Imputation strategy (mean, median, constant)

        Returns:
            DataFrame: Data with imputed column
        """
        if strategy == 'median':
            # Calculate median
            median_value = df.approxQuantile(col_name, [0.5], 0.01)[0]
            df = df.withColumn(
                col_name,
                coalesce(col(col_name), lit(median_value))
            )
            logger.debug(f"Imputed {col_name} with median: {median_value}")

        elif strategy == 'mean':
            # Calculate mean
            mean_value = df.select(mean(col(col_name))).first()[0]
            df = df.withColumn(
                col_name,
                coalesce(col(col_name), lit(mean_value))
            )
            logger.debug(f"Imputed {col_name} with mean: {mean_value}")

        elif strategy == 'constant':
            df = df.withColumn(
                col_name,
                coalesce(col(col_name), lit(0))
            )

        return df

    def detect_outliers(self, df: DataFrame) -> DataFrame:
        """
        Detect outliers and add flags

        Args:
            df: Input DataFrame

        Returns:
            DataFrame: Data with outlier flags
        """
        logger.info("Detecting outliers...")

        method = self.outlier_config.get('method', 'iqr')
        columns = self.outlier_config.get('columns', ['trx_amt'])

        if method == 'iqr':
            df = self._detect_outliers_iqr(df, columns)
        elif method == 'zscore':
            df = self._detect_outliers_zscore(df, columns)

        logger.info("Outlier detection complete")
        return df

    def _detect_outliers_iqr(
        self,
        df: DataFrame,
        columns: List[str]
    ) -> DataFrame:
        """
        Detect outliers using IQR method

        Args:
            df: DataFrame
            columns: Columns to check

        Returns:
            DataFrame: Data with outlier flags
        """
        multiplier = self.outlier_config.get('iqr', {}).get('multiplier', 3.0)

        for col_name in columns:
            if col_name not in df.columns:
                continue

            # Calculate quartiles
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Add outlier flag
            flag_name = f"is_{col_name}_outlier"
            df = df.withColumn(
                flag_name,
                ((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).cast("int")
            )

            # Log statistics
            outlier_count = df.filter(col(flag_name) == 1).count()
            total_count = df.count()
            outlier_pct = (outlier_count / total_count * 100) if total_count > 0 else 0

            logger.info(f"{col_name} - Outliers: {outlier_count:,} ({outlier_pct:.2f}%)")
            logger.debug(f"  IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

        return df

    def _detect_outliers_zscore(
        self,
        df: DataFrame,
        columns: List[str]
    ) -> DataFrame:
        """
        Detect outliers using Z-score method

        Args:
            df: DataFrame
            columns: Columns to check

        Returns:
            DataFrame: Data with outlier flags
        """
        threshold = self.outlier_config.get('zscore', {}).get('threshold', 3.0)

        for col_name in columns:
            if col_name not in df.columns:
                continue

            # Calculate mean and stddev
            stats = df.select(
                mean(col(col_name)).alias('mean'),
                stddev(col(col_name)).alias('std')
            ).first()

            col_mean = stats['mean']
            col_std = stats['std']

            if col_std == 0:
                logger.warning(f"{col_name} has zero standard deviation, skipping")
                continue

            # Calculate z-score and flag
            flag_name = f"is_{col_name}_outlier"
            df = df.withColumn(
                "z_score",
                (col(col_name) - col_mean) / col_std
            ).withColumn(
                flag_name,
                (spark_abs(col("z_score")) > threshold).cast("int")
            ).drop("z_score")

            # Log statistics
            outlier_count = df.filter(col(flag_name) == 1).count()
            total_count = df.count()
            outlier_pct = (outlier_count / total_count * 100) if total_count > 0 else 0

            logger.info(f"{col_name} - Outliers: {outlier_count:,} ({outlier_pct:.2f}%)")

        return df

    def standardize_formats(self, df: DataFrame) -> DataFrame:
        """
        Standardize data formats

        Args:
            df: Input DataFrame

        Returns:
            DataFrame: Standardized data
        """
        logger.info("Standardizing data formats...")

        # Standardize dates
        if 'data_date' in df.columns:
            df = df.withColumn('data_date', to_date(col('data_date')))

        if 'trans_initiate_time' in df.columns:
            df = df.withColumn('trans_initiate_time', to_timestamp(col('trans_initiate_time')))

        # Standardize string columns (trim and uppercase)
        string_cols = [
            'trx_channel', 'trx_type', 'trx_status',
            'from_city', 'to_city', 'from_region', 'to_region'
        ]

        for col_name in string_cols:
            if col_name in df.columns:
                df = df.withColumn(col_name, trim(upper(col(col_name))))

        logger.info("Format standardization complete")
        return df

    def validate_data_quality(self, df: DataFrame) -> DataFrame:
        """
        Validate data quality based on configured checks

        Args:
            df: Input DataFrame

        Returns:
            DataFrame: Validated data
        """
        logger.info("Validating data quality...")

        if not self.quality_config.get('enabled', True):
            logger.info("Data quality checks disabled")
            return df

        checks = self.quality_config.get('checks', [])

        for check in checks:
            check_name = check['name']
            check_type = check['type']
            column = check['column']

            logger.debug(f"Running check: {check_name}")

            if check_type == 'not_null':
                null_count = df.filter(col(column).isNull()).count()
                if null_count > 0:
                    logger.warning(f"Check failed: {check_name} - {null_count:,} null values found")
                else:
                    logger.debug(f"Check passed: {check_name}")

            elif check_type == 'unique':
                total = df.count()
                distinct = df.select(column).distinct().count()
                if total != distinct:
                    duplicates = total - distinct
                    logger.warning(f"Check failed: {check_name} - {duplicates:,} duplicates found")
                else:
                    logger.debug(f"Check passed: {check_name}")

            elif check_type == 'greater_than':
                value = check['value']
                violation_count = df.filter(col(column) <= value).count()
                if violation_count > 0:
                    logger.warning(f"Check failed: {check_name} - {violation_count:,} violations")
                else:
                    logger.debug(f"Check passed: {check_name}")

        logger.info("Data quality validation complete")
        return df

    def clean_pipeline(self, df: DataFrame) -> DataFrame:
        """
        Complete data cleaning pipeline

        Args:
            df: Input DataFrame

        Returns:
            DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning pipeline...")

        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 2: Standardize formats
        df = self.standardize_formats(df)

        # Step 3: Handle missing values
        df = self.handle_missing_values(df)

        # Step 4: Detect outliers
        df = self.detect_outliers(df)

        # Step 5: Validate data quality
        df = self.validate_data_quality(df)

        logger.info("Data cleaning pipeline complete!")
        return df

    def _log_missing_stats(self, df: DataFrame):
        """Log missing value statistics"""
        logger.info("Missing value statistics:")

        for col_name in df.columns:
            null_count = df.filter(col(col_name).isNull()).count()
            total_count = df.count()
            null_pct = (null_count / total_count * 100) if total_count > 0 else 0

            if null_count > 0:
                logger.info(f"  {col_name}: {null_count:,} ({null_pct:.2f}%)")


def main():
    """Main function for testing"""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("DataCleaning").getOrCreate()

    # Load data
    df = spark.read.parquet("data/processed/master_dataset.parquet")

    # Initialize cleaner
    cleaner = DataCleaner()

    # Clean data
    df_clean = cleaner.clean_pipeline(df)

    # Save cleaned data
    write_parquet(
        df_clean,
        "data/processed/cleaned_dataset.parquet",
        partition_by=["data_date"]
    )

    spark.stop()


if __name__ == "__main__":
    main()
