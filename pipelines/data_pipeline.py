"""
Data extraction and cleaning pipeline
"""
import sys
sys.path.append('.')

from pyspark.sql import SparkSession
from src.data.extraction import DataExtractor
from src.data.cleaning import DataCleaner
from src.utils.spark_utils import create_spark_session, write_parquet
from src.utils.logger import get_logger

logger = get_logger("data_pipeline")


def run_data_pipeline(
    use_sample: bool = True,
    save_raw: bool = True,
    save_cleaned: bool = True
):
    """
    Run complete data pipeline

    Args:
        use_sample: Whether to use sample data
        save_raw: Whether to save raw data
        save_cleaned: Whether to save cleaned data
    """
    logger.info("=" * 80)
    logger.info("STARTING DATA PIPELINE")
    logger.info("=" * 80)

    # Create Spark session
    logger.info("\n[1/4] Creating Spark session...")
    spark = create_spark_session(
        app_name="FraudDetection-DataPipeline",
        jdbc_driver_path="utils/postgresql-42.7.1.jar"
    )

    try:
        # Extract data
        logger.info("\n[2/4] Extracting data from PostgreSQL...")
        extractor = DataExtractor(spark)

        df_master = extractor.extract_and_combine_all(
            use_sample=use_sample,
            save_intermediate=save_raw
        )

        # Save raw master dataset
        if save_raw:
            logger.info("\n[3/4] Saving raw master dataset...")
            write_parquet(
                df_master,
                "data/processed/master_dataset.parquet",
                partition_by=["data_date", "is_fraud"]
            )

        # Clean data
        logger.info("\n[4/4] Cleaning data...")
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_pipeline(df_master)

        # Save cleaned dataset
        if save_cleaned:
            logger.info("Saving cleaned dataset...")
            write_parquet(
                df_cleaned,
                "data/processed/cleaned_dataset.parquet",
                partition_by=["data_date"]
            )

        logger.info("\n" + "=" * 80)
        logger.info("DATA PIPELINE COMPLETE")
        logger.info("=" * 80)

        return df_cleaned

    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run data pipeline')
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'july', 'full'],
                       help='Processing mode')
    parser.add_argument('--skip-raw', action='store_true',
                       help='Skip saving raw data')

    args = parser.parse_args()

    run_data_pipeline(
        use_sample=(args.mode == 'sample'),
        save_raw=not args.skip_raw,
        save_cleaned=True
    )
