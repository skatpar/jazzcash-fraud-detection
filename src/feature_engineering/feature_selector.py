"""
Feature Selection Module for JazzCash Fraud Detection
Implements correlation analysis, VIF, and various feature selection methods
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Handles feature selection and importance analysis"""

    def __init__(self, target_col: str = 'is_fraud'):
        self.target_col = target_col
        self.feature_importance_scores = {}
        self.selected_features = []

    def correlation_analysis(self,
                            df: pd.DataFrame,
                            threshold: float = 0.9) -> Tuple[pd.DataFrame, List[Tuple]]:
        """Analyze correlations and identify highly correlated pairs"""
        logger.info("Performing correlation analysis...")

        # Select only numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_col]

        # Compute correlation matrix
        corr_matrix = df[numerical_cols].corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        logger.info(f"Found {len(high_corr_pairs)} feature pairs with |r| > {threshold}")

        # Correlation with target
        if self.target_col in df.columns:
            target_corr = df[numerical_cols].corrwith(df[self.target_col]).abs()
            target_corr = target_corr.sort_values(ascending=False)

            logger.info(f"\nTop 10 features correlated with target:")
            for feature, corr in target_corr.head(10).items():
                logger.info(f"  {feature}: {corr:.4f}")

            return corr_matrix, high_corr_pairs, target_corr
        else:
            return corr_matrix, high_corr_pairs, None

    def calculate_vif(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for features"""
        logger.info("Calculating VIF for multicollinearity detection...")

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target_col]

        # Remove any features with NaN or inf
        X = df[features].replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        valid_features = X.columns.tolist()

        vif_data = pd.DataFrame()
        vif_data['feature'] = valid_features

        vif_values = []
        for i in range(len(valid_features)):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_values.append(vif)
            except:
                vif_values.append(np.nan)

        vif_data['VIF'] = vif_values
        vif_data = vif_data.sort_values('VIF', ascending=False)

        # Flag high VIF features
        high_vif = vif_data[vif_data['VIF'] > 10]
        logger.info(f"Found {len(high_vif)} features with VIF > 10")

        return vif_data

    def iterative_vif_reduction(self,
                                df: pd.DataFrame,
                                features: List[str] = None,
                                threshold: float = 10.0) -> List[str]:
        """Iteratively remove features with highest VIF until all are below threshold"""
        logger.info(f"Performing iterative VIF reduction (threshold={threshold})...")

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target_col]

        current_features = features.copy()
        iteration = 0

        while True:
            iteration += 1
            vif_data = self.calculate_vif(df, current_features)

            max_vif = vif_data['VIF'].max()

            if max_vif <= threshold:
                logger.info(f"All features have VIF <= {threshold} after {iteration} iterations")
                break

            # Remove feature with highest VIF
            feature_to_remove = vif_data.iloc[0]['feature']
            current_features.remove(feature_to_remove)
            logger.info(f"  Iteration {iteration}: Removed '{feature_to_remove}' (VIF={max_vif:.2f})")

        logger.info(f"Reduced from {len(features)} to {len(current_features)} features")

        return current_features

    def mutual_information_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate mutual information scores for features"""
        logger.info("Calculating mutual information scores...")

        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        self.feature_importance_scores['mutual_information'] = mi_df

        logger.info(f"Top 10 features by mutual information:")
        for idx, row in mi_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['mi_score']:.4f}")

        return mi_df

    def random_forest_importance(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 n_estimators: int = 100) -> pd.DataFrame:
        """Calculate feature importance using Random Forest"""
        logger.info("Calculating Random Forest feature importance...")

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
        rf_model.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_importance_scores['random_forest'] = importance_df

        logger.info(f"Top 10 features by Random Forest importance:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def lasso_feature_selection(self,
                                X: pd.DataFrame,
                                y: pd.Series,
                                C: float = 0.01) -> List[str]:
        """Select features using L1 (LASSO) regularization"""
        logger.info("Performing LASSO feature selection...")

        lasso = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=C,
            max_iter=1000,
            random_state=42
        )
        lasso.fit(X, y)

        # Get features with non-zero coefficients
        selected_features = X.columns[lasso.coef_[0] != 0].tolist()

        logger.info(f"LASSO selected {len(selected_features)} features out of {X.shape[1]}")

        self.feature_importance_scores['lasso'] = pd.DataFrame({
            'feature': X.columns,
            'coefficient': lasso.coef_[0]
        }).sort_values('coefficient', ascending=False, key=abs)

        return selected_features

    def recursive_feature_elimination(self,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      n_features_to_select: int = 50) -> List[str]:
        """Perform Recursive Feature Elimination"""
        logger.info(f"Performing RFE to select {n_features_to_select} features...")

        estimator = GradientBoostingClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=5
        )

        rfe = RFE(
            estimator,
            n_features_to_select=n_features_to_select,
            step=5
        )

        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_].tolist()

        logger.info(f"RFE selected {len(selected_features)} features")

        # Get feature rankings
        ranking_df = pd.DataFrame({
            'feature': X.columns,
            'ranking': rfe.ranking_
        }).sort_values('ranking')

        self.feature_importance_scores['rfe'] = ranking_df

        return selected_features

    def ensemble_feature_importance(self,
                                    X: pd.DataFrame,
                                    y: pd.Series) -> pd.DataFrame:
        """Combine multiple feature importance methods"""
        logger.info("Creating ensemble feature importance scores...")

        # Get scores from multiple methods
        mi_scores = self.mutual_information_scores(X, y)
        rf_scores = self.random_forest_importance(X, y)

        # Normalize scores to 0-1 range
        def normalize_scores(scores):
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # Create combined dataframe
        combined_df = pd.DataFrame({'feature': X.columns})

        # Add normalized scores
        combined_df = combined_df.merge(
            mi_scores[['feature', 'mi_score']],
            on='feature',
            how='left'
        )
        combined_df['mi_score_norm'] = normalize_scores(combined_df['mi_score'].fillna(0))

        combined_df = combined_df.merge(
            rf_scores[['feature', 'importance']],
            on='feature',
            how='left'
        )
        combined_df['rf_importance_norm'] = normalize_scores(combined_df['importance'].fillna(0))

        # Calculate average importance
        combined_df['avg_importance'] = combined_df[
            ['mi_score_norm', 'rf_importance_norm']
        ].mean(axis=1)

        combined_df = combined_df.sort_values('avg_importance', ascending=False)

        logger.info(f"Top 10 features by ensemble importance:")
        for idx, row in combined_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['avg_importance']:.4f}")

        return combined_df

    def select_features_multistage(self,
                                   df: pd.DataFrame,
                                   target_col: str = None,
                                   n_final_features: int = 50) -> List[str]:
        """Multi-stage feature selection pipeline"""
        logger.info("Starting multi-stage feature selection...")

        if target_col is None:
            target_col = self.target_col

        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Stage 1: Remove near-zero variance features
        logger.info("Stage 1: Removing near-zero variance features...")
        variance = X.var()
        low_variance_features = variance[variance < 0.01].index.tolist()
        X = X.drop(columns=low_variance_features)
        logger.info(f"  Removed {len(low_variance_features)} low-variance features")

        # Stage 2: Remove highly correlated features
        logger.info("Stage 2: Removing highly correlated features...")
        corr_matrix, high_corr_pairs, target_corr = self.correlation_analysis(
            X.assign(**{target_col: y}),
            threshold=0.95
        )

        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            # Keep the feature with higher correlation to target
            if target_corr[feat1] > target_corr[feat2]:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)

        X = X.drop(columns=list(features_to_remove))
        logger.info(f"  Removed {len(features_to_remove)} highly correlated features")

        # Stage 3: VIF-based reduction
        logger.info("Stage 3: VIF-based multicollinearity reduction...")
        vif_selected_features = self.iterative_vif_reduction(X, threshold=10.0)
        X = X[vif_selected_features]
        logger.info(f"  Retained {len(vif_selected_features)} features after VIF reduction")

        # Stage 4: Ensemble importance-based selection
        logger.info(f"Stage 4: Selecting top {n_final_features} by ensemble importance...")
        ensemble_scores = self.ensemble_feature_importance(X, y)
        final_features = ensemble_scores.head(n_final_features)['feature'].tolist()

        logger.info(f"\nFinal selection: {len(final_features)} features")

        self.selected_features = final_features

        return final_features

    def validate_feature_sets(self,
                             df: pd.DataFrame,
                             feature_sets: Dict[str, List[str]],
                             target_col: str = None,
                             cv: int = 5) -> pd.DataFrame:
        """Compare performance of different feature sets"""
        logger.info("Validating feature sets with cross-validation...")

        if target_col is None:
            target_col = self.target_col

        y = df[target_col]
        results = []

        for name, features in feature_sets.items():
            X = df[features]

            # Try different models
            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5)
            }

            for model_name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                    results.append({
                        'feature_set': name,
                        'model': model_name,
                        'n_features': len(features),
                        'mean_auc': scores.mean(),
                        'std_auc': scores.std()
                    })
                    logger.info(f"  {name} + {model_name}: AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
                except Exception as e:
                    logger.warning(f"  Failed to validate {name} + {model_name}: {str(e)}")

        results_df = pd.DataFrame(results).sort_values('mean_auc', ascending=False)

        return results_df


if __name__ == "__main__":
    # Example usage
    selector = FeatureSelector(target_col='is_fraud')

    # Load feature-engineered dataset
    df = pd.read_csv('data/features/feature_engineered_dataset.csv')

    # Perform multi-stage feature selection
    selected_features = selector.select_features_multistage(
        df,
        n_final_features=50
    )

    # Save selected features
    with open('data/features/selected_features.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    logger.info(f"Selected features saved to data/features/selected_features.txt")

    # Validate different feature sets
    feature_sets = {
        'top_20': selected_features[:20],
        'top_50': selected_features[:50],
        'all': df.drop(columns=['is_fraud']).columns.tolist()
    }

    validation_results = selector.validate_feature_sets(df, feature_sets)
    validation_results.to_csv('data/features/feature_validation_results.csv', index=False)
    logger.info("Validation results saved")
