"""
Model training and evaluation pipeline
"""
import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from src.models.train import (
    prepare_train_test_split, train_random_forest, train_xgboost,
    get_feature_importance, save_model
)
from src.models.evaluate import evaluate_model, get_top_k_precision
from src.visualization.plots import (
    plot_feature_importance, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix
)
from src.utils.logger import get_logger

logger = get_logger("model_pipeline")


def run_model_pipeline(
    input_path: str = "data/features/feature_dataset.parquet",
    model_type: str = "xgboost",
    sample_size: int = None
):
    """
    Run complete model training and evaluation pipeline

    Args:
        input_path: Feature dataset path
        model_type: Model type (random_forest or xgboost)
        sample_size: Sample size for training (None = use all)
    """
    logger.info("=" * 80)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    # Create output directories
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("plots/model").mkdir(parents=True, exist_ok=True)

    try:
        # Load feature data
        logger.info(f"\n[1/6] Loading feature data from {input_path}...")
        df = pd.read_parquet(input_path)

        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size:,} rows...")
            # Stratified sampling
            fraud = df[df['is_fraud'] == 1]
            non_fraud = df[df['is_fraud'] == 0].sample(
                n=min(sample_size, len(df[df['is_fraud'] == 0])),
                random_state=42
            )
            df = pd.concat([fraud, non_fraud])

        logger.info(f"Dataset size: {len(df):,}")

        # Prepare train/test split
        logger.info("\n[2/6] Preparing train/test split...")
        X_train, X_test, y_train, y_test = prepare_train_test_split(df)

        # Train model
        logger.info(f"\n[3/6] Training {model_type} model...")
        if model_type == "random_forest":
            model = train_random_forest(X_train, y_train)
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Get feature importance
        logger.info("\n[4/6] Extracting feature importance...")
        importance_df = get_feature_importance(model, X_train.columns.tolist())
        importance_df.to_csv(f"data/models/{model_type}_feature_importance.csv", index=False)

        # Plot feature importance
        plot_feature_importance(
            importance_df,
            top_n=30,
            save_path=f"plots/model/{model_type}_feature_importance.png"
        )

        # Evaluate model
        logger.info("\n[5/6] Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        # Create evaluation plots
        logger.info("Creating evaluation plots...")
        plot_roc_curve(
            y_test,
            metrics['y_pred_proba'],
            save_path=f"plots/model/{model_type}_roc_curve.png"
        )

        plot_precision_recall_curve(
            y_test,
            metrics['y_pred_proba'],
            save_path=f"plots/model/{model_type}_pr_curve.png"
        )

        plot_confusion_matrix(
            y_test,
            metrics['y_pred'],
            save_path=f"plots/model/{model_type}_confusion_matrix.png"
        )

        # Calculate precision in top 5%
        top_5_precision = get_top_k_precision(y_test, metrics['y_pred_proba'], k=0.05)
        metrics['top_5_precision'] = top_5_precision

        # Save model
        logger.info(f"\n[6/6] Saving model...")
        model_path = f"data/models/{model_type}_model.pkl"
        save_model(model, model_path)

        # Save metrics
        metrics_df = pd.DataFrame([{
            'model_type': model_type,
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'mcc': metrics['mcc'],
            'top_5_precision': metrics['top_5_precision']
        }])
        metrics_df.to_csv(f"data/models/{model_type}_metrics.csv", index=False)

        logger.info("\n" + "=" * 80)
        logger.info("MODEL TRAINING PIPELINE COMPLETE")
        logger.info("=" * 80)

        return model, metrics

    except Exception as e:
        logger.error(f"Model pipeline failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run model pipeline')
    parser.add_argument('--input', type=str,
                       default='data/features/feature_dataset.parquet',
                       help='Input feature data path')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['random_forest', 'xgboost'],
                       help='Model type')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for training')

    args = parser.parse_args()

    run_model_pipeline(
        input_path=args.input,
        model_type=args.model,
        sample_size=args.sample
    )
