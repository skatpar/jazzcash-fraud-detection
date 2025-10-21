"""
Main pipeline orchestrator for end-to-end fraud detection
"""
import sys
sys.path.append('.')

import argparse
from datetime import datetime
from pipelines.data_pipeline import run_data_pipeline
from pipelines.feature_pipeline import run_feature_pipeline
from pipelines.model_pipeline import run_model_pipeline
from src.utils.logger import get_logger

logger = get_logger("main_pipeline")


def run_full_pipeline(
    mode: str = "sample",
    model_type: str = "xgboost",
    skip_data: bool = False,
    skip_features: bool = False,
    sample_size: int = None
):
    """
    Run complete end-to-end pipeline

    Args:
        mode: Processing mode (sample, july, full)
        model_type: Model type to train
        skip_data: Skip data extraction (use existing)
        skip_features: Skip feature engineering (use existing)
        sample_size: Sample size for model training
    """
    start_time = datetime.now()

    logger.info("#" * 80)
    logger.info("#" + " " * 78 + "#")
    logger.info("#" + " " * 20 + "FRAUD DETECTION PIPELINE" + " " * 34 + "#")
    logger.info("#" + " " * 78 + "#")
    logger.info("#" * 80)
    logger.info(f"\nMode: {mode}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Start Time: {start_time}")

    try:
        # Stage 1: Data Extraction and Cleaning
        if not skip_data:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 1/3: DATA EXTRACTION AND CLEANING")
            logger.info("=" * 80)

            run_data_pipeline(
                use_sample=(mode == "sample"),
                save_raw=True,
                save_cleaned=True
            )

            logger.info("\n✓ Stage 1 complete")
        else:
            logger.info("\n⊳ Skipping Stage 1: Data Extraction")

        # Stage 2: Feature Engineering
        if not skip_features:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2/3: FEATURE ENGINEERING")
            logger.info("=" * 80)

            run_feature_pipeline(
                input_path="data/processed/cleaned_dataset.parquet",
                output_path="data/features/feature_dataset.parquet"
            )

            logger.info("\n✓ Stage 2 complete")
        else:
            logger.info("\n⊳ Skipping Stage 2: Feature Engineering")

        # Stage 3: Model Training and Evaluation
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3/3: MODEL TRAINING AND EVALUATION")
        logger.info("=" * 80)

        model, metrics = run_model_pipeline(
            input_path="data/features/feature_dataset.parquet",
            model_type=model_type,
            sample_size=sample_size
        )

        logger.info("\n✓ Stage 3 complete")

        # Pipeline Summary
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "#" * 80)
        logger.info("#" + " " * 78 + "#")
        logger.info("#" + " " * 25 + "PIPELINE COMPLETE" + " " * 37 + "#")
        logger.info("#" + " " * 78 + "#")
        logger.info("#" * 80)

        logger.info(f"\nTotal Duration: {duration}")
        logger.info(f"End Time: {end_time}")

        logger.info("\n" + "-" * 80)
        logger.info("FINAL MODEL METRICS:")
        logger.info("-" * 80)
        logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:      {metrics['pr_auc']:.4f}")
        logger.info(f"F1-Score:    {metrics['f1_score']:.4f}")
        logger.info(f"Precision:   {metrics['precision']:.4f}")
        logger.info(f"Recall:      {metrics['recall']:.4f}")
        logger.info(f"MCC:         {metrics['mcc']:.4f}")
        logger.info(f"Top 5% Prec: {metrics['top_5_precision']:.4f}")
        logger.info("-" * 80)

        logger.info("\nOutput Files:")
        logger.info("  - Data: data/processed/cleaned_dataset.parquet")
        logger.info("  - Features: data/features/feature_dataset.parquet")
        logger.info(f"  - Model: data/models/{model_type}_model.pkl")
        logger.info(f"  - Metrics: data/models/{model_type}_metrics.csv")
        logger.info("  - Plots: plots/model/")

        return model, metrics

    except Exception as e:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(f"{'!' * 80}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run complete fraud detection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample data (development)
  python pipelines/main_pipeline.py --mode sample

  # Run with July data
  python pipelines/main_pipeline.py --mode july --model xgboost

  # Skip data extraction, use existing cleaned data
  python pipelines/main_pipeline.py --skip-data --skip-features

  # Train with limited sample size
  python pipelines/main_pipeline.py --mode sample --sample 100000
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='sample',
        choices=['sample', 'july', 'full'],
        help='Processing mode (default: sample)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['random_forest', 'xgboost'],
        help='Model type to train (default: xgboost)'
    )

    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data extraction and cleaning stage'
    )

    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature engineering stage'
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size for model training (default: use all data)'
    )

    args = parser.parse_args()

    run_full_pipeline(
        mode=args.mode,
        model_type=args.model,
        skip_data=args.skip_data,
        skip_features=args.skip_features,
        sample_size=args.sample
    )
