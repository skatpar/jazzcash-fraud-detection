"""
Model evaluation module
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
    f1_score, precision_score, recall_score, matthews_corrcoef
)
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger("model_evaluation")


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }

    logger.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    metrics['confusion_matrix'] = cm
    metrics['y_pred'] = y_pred
    metrics['y_pred_proba'] = y_pred_proba

    return metrics


def calculate_precision_at_recall(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_recall: float = 0.9
) -> Tuple[float, float]:
    """
    Calculate precision at target recall

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        target_recall: Target recall level

    Returns:
        Tuple of (precision, threshold)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Find precision at target recall
    idx = np.argmax(recall >= target_recall)

    if idx == 0 and recall[0] < target_recall:
        logger.warning(f"Cannot achieve {target_recall} recall")
        return 0.0, 1.0

    return precision[idx], thresholds[idx]


def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Calculate metrics at specific threshold

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Probability threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'threshold': threshold
    }


def get_top_k_precision(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: float = 0.05
) -> float:
    """
    Calculate precision in top k% predictions

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        k: Top percentage

    Returns:
        Precision in top k%
    """
    n = int(len(y_true) * k)
    top_k_idx = np.argsort(y_pred_proba)[-n:]

    precision = y_true.iloc[top_k_idx].mean() if isinstance(y_true, pd.Series) else y_true[top_k_idx].mean()

    logger.info(f"Precision in top {k*100}%: {precision:.4f}")

    return precision
