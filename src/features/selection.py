"""
Feature selection module
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils.logger import get_logger

logger = get_logger("feature_selection")


def calculate_correlation(
    df_pandas: pd.DataFrame,
    target_col: str = "is_fraud",
    threshold: float = 0.9
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate feature correlations and identify highly correlated pairs

    Args:
        df_pandas: Pandas DataFrame
        target_col: Target column name
        threshold: Correlation threshold

    Returns:
        Tuple of (correlation matrix, features to remove)
    """
    logger.info("Calculating feature correlations...")

    # Select numerical columns only
    numerical_cols = df_pandas.select_dtypes(include=[np.number]).columns.tolist()

    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    # Calculate correlation matrix
    corr_matrix = df_pandas[numerical_cols + [target_col]].corr()

    # Find highly correlated features
    features_to_remove = set()

    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Remove feature with lower correlation to target
                feat_i = numerical_cols[i]
                feat_j = numerical_cols[j]

                corr_i = abs(corr_matrix.loc[feat_i, target_col])
                corr_j = abs(corr_matrix.loc[feat_j, target_col])

                if corr_i > corr_j:
                    features_to_remove.add(feat_j)
                else:
                    features_to_remove.add(feat_i)

    logger.info(f"Found {len(features_to_remove)} highly correlated features to remove")

    return corr_matrix, list(features_to_remove)


def calculate_vif(
    df_pandas: pd.DataFrame,
    features: List[str],
    threshold: float = 10.0
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor

    Args:
        df_pandas: Pandas DataFrame
        features: List of features
        threshold: VIF threshold

    Returns:
        DataFrame with VIF scores
    """
    logger.info("Calculating VIF...")

    vif_data = pd.DataFrame()
    vif_data["Feature"] = features

    try:
        vif_data["VIF"] = [
            variance_inflation_factor(df_pandas[features].values, i)
            for i in range(len(features))
        ]

        high_vif = vif_data[vif_data["VIF"] > threshold]
        logger.info(f"Found {len(high_vif)} features with VIF > {threshold}")

        return vif_data.sort_values("VIF", ascending=False)

    except Exception as e:
        logger.error(f"Error calculating VIF: {e}")
        return vif_data


def calculate_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 50
) -> pd.DataFrame:
    """
    Calculate mutual information scores

    Args:
        X: Feature DataFrame
        y: Target series
        top_k: Number of top features to return

    Returns:
        DataFrame with MI scores
    """
    logger.info("Calculating mutual information scores...")

    mi_scores = mutual_info_classif(X, y, random_state=42)

    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    logger.info(f"Top {top_k} features by MI score")

    return mi_df.head(top_k)


def select_features_by_importance(
    feature_importance_df: pd.DataFrame,
    top_k: int = 100
) -> List[str]:
    """
    Select top k features by importance

    Args:
        feature_importance_df: DataFrame with feature importance
        top_k: Number of features to select

    Returns:
        List of selected features
    """
    logger.info(f"Selecting top {top_k} features...")

    top_features = feature_importance_df.head(top_k)['feature'].tolist()

    logger.info(f"Selected {len(top_features)} features")

    return top_features


def remove_low_variance_features(
    df_pandas: pd.DataFrame,
    threshold: float = 0.01
) -> List[str]:
    """
    Identify low variance features

    Args:
        df_pandas: Pandas DataFrame
        threshold: Variance threshold

    Returns:
        List of low variance features
    """
    logger.info("Identifying low variance features...")

    numerical_cols = df_pandas.select_dtypes(include=[np.number]).columns
    variances = df_pandas[numerical_cols].var()

    low_var_features = variances[variances < threshold].index.tolist()

    logger.info(f"Found {len(low_var_features)} low variance features")

    return low_var_features


def create_feature_importance_report(
    corr_matrix: pd.DataFrame,
    vif_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    target_col: str = "is_fraud"
) -> pd.DataFrame:
    """
    Create comprehensive feature importance report

    Args:
        corr_matrix: Correlation matrix
        vif_df: VIF DataFrame
        mi_df: Mutual information DataFrame
        target_col: Target column

    Returns:
        Combined importance report
    """
    logger.info("Creating feature importance report...")

    # Get target correlations
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)

    # Merge all metrics
    report = pd.DataFrame({
        'feature': target_corr.index,
        'target_correlation': target_corr.values
    })

    # Add VIF if available
    if not vif_df.empty:
        report = report.merge(
            vif_df[['Feature', 'VIF']].rename(columns={'Feature': 'feature'}),
            on='feature',
            how='left'
        )

    # Add MI scores if available
    if not mi_df.empty:
        report = report.merge(
            mi_df[['feature', 'mi_score']],
            on='feature',
            how='left'
        )

    report = report.sort_values('target_correlation', ascending=False)

    logger.info("Feature importance report created")

    return report
