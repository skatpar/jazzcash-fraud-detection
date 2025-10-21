"""
Model training module
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
from typing import Tuple, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("model_training")


def prepare_train_test_split(
    df_pandas: pd.DataFrame,
    target_col: str = "is_fraud",
    test_size: float = 0.2,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare train/test split

    Args:
        df_pandas: Input DataFrame
        target_col: Target column name
        test_size: Test set size
        stratify: Whether to stratify

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparing train/test split...")

    # Separate features and target
    X = df_pandas.drop(columns=[target_col])
    y = df_pandas[target_col]

    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if stratify else None
    )

    logger.info(f"Train set: {len(X_train):,} rows")
    logger.info(f"Test set: {len(X_test):,} rows")
    logger.info(f"Fraud rate in train: {y_train.mean():.4%}")
    logger.info(f"Fraud rate in test: {y_test.mean():.4%}")

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> RandomForestClassifier:
    """
    Train Random Forest model

    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Model parameters

    Returns:
        Trained model
    """
    logger.info("Training Random Forest model...")

    # Default parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    params.update(kwargs)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Random Forest training complete")

    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> xgb.XGBClassifier:
    """
    Train XGBoost model

    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Model parameters

    Returns:
        Trained model
    """
    logger.info("Training XGBoost model...")

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Default parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    params.update(kwargs)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("XGBoost training complete")

    return model


def get_feature_importance(
    model,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract feature importance from model

    Args:
        model: Trained model
        feature_names: Feature names

    Returns:
        DataFrame with feature importance
    """
    logger.info("Extracting feature importance...")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance_df


def save_model(model, path: str):
    """
    Save trained model

    Args:
        model: Model to save
        path: Save path
    """
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)
    logger.info("Model saved")


def load_model(path: str):
    """
    Load trained model

    Args:
        path: Model path

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {path}")
    model = joblib.load(path)
    logger.info("Model loaded")
    return model
