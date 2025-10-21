"""
Feature engineering and selection pipeline
"""
import sys
sys.path.append('.')

from pyspark.sql import SparkSession
from src.features.transaction_features import TransactionFeatureEngineer
from src.features.temporal_features import create_temporal_features
from src.features.account_features import create_account_features, create_categorical_encoding
from src.utils.spark_utils import create_spark_session, write_parquet
from src.utils.logger import get_logger

logger = get_logger("feature_pipeline")


def run_feature_pipeline(
    input_path: str = "data/processed/cleaned_dataset.parquet",
    output_path: str = "data/features/feature_dataset.parquet"
):
    """
    Run complete feature engineering pipeline

    Args:
        input_path: Input data path
        output_path: Output data path
    """
    logger.info("=" * 80)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)

    # Create Spark session
    logger.info("\n[1/6] Creating Spark session...")
    spark = create_spark_session(app_name="FraudDetection-FeaturePipeline")

    try:
        # Load cleaned data
        logger.info(f"\n[2/6] Loading cleaned data from {input_path}...")
        df = spark.read.parquet(input_path)

        initial_cols = len(df.columns)
        logger.info(f"Initial columns: {initial_cols}")

        # Create temporal features
        logger.info("\n[3/6] Creating temporal features...")
        df = create_temporal_features(df)

        # Create account features
        logger.info("\n[4/6] Creating account features...")
        df = create_account_features(df)
        df = create_categorical_encoding(df)

        # Create transaction features
        logger.info("\n[5/6] Creating transaction features...")
        tx_engineer = TransactionFeatureEngineer()
        df = tx_engineer.create_all_features(df)

        final_cols = len(df.columns)
        new_features = final_cols - initial_cols
        logger.info(f"\nTotal features created: {new_features}")
        logger.info(f"Final column count: {final_cols}")

        # Save feature dataset
        logger.info(f"\n[6/6] Saving feature dataset to {output_path}...")
        write_parquet(
            df,
            output_path,
            partition_by=["data_date"]
        )

        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
        logger.info("=" * 80)

        return df

    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run feature pipeline')
    parser.add_argument('--input', type=str,
                       default='data/processed/cleaned_dataset.parquet',
                       help='Input data path')
    parser.add_argument('--output', type=str,
                       default='data/features/feature_dataset.parquet',
                       help='Output data path')

    args = parser.parse_args()

    run_feature_pipeline(
        input_path=args.input,
        output_path=args.output
    )
