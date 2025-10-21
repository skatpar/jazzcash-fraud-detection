"""
Data validation utilities
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, isnan, isnull
from src.utils.logger import get_logger

logger = get_logger("data_validation")


def validate_schema(df: DataFrame, expected_columns: list) -> bool:
    """
    Validate DataFrame schema

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names

    Returns:
        bool: True if valid
    """
    actual_columns = set(df.columns)
    expected_columns = set(expected_columns)

    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    if missing:
        logger.warning(f"Missing columns: {missing}")
    if extra:
        logger.info(f"Extra columns: {extra}")

    return len(missing) == 0


def validate_row_count(df: DataFrame, min_rows: int = 1) -> bool:
    """Validate minimum row count"""
    count = df.count()
    logger.info(f"Row count: {count:,}")
    return count >= min_rows


def generate_data_profile(df: DataFrame) -> dict:
    """
    Generate data profile statistics

    Args:
        df: DataFrame to profile

    Returns:
        dict: Profile statistics
    """
    profile = {
        'row_count': df.count(),
        'column_count': len(df.columns),
        'columns': df.columns
    }

    logger.info(f"Data Profile: {profile['row_count']:,} rows, {profile['column_count']} columns")

    return profile
