"""
JazzCash Fraud Detection - Main Pipeline
Orchestrates the complete data processing and feature engineering pipeline
"""

import argparse
import logging
import yaml
from pathlib import Path

from src.data_integration.merge_datasets import DataIntegrator
from src.data_integration.data_cleaner import DataCleaner
from src.feature_engineering.feature_builder import FeatureEngineer
from src.feature_engineering.feature_selector import FeatureSelector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_data_integration(config: dict) -> None:
    """Run data integration phase"""
    logger.info("=" * 80)
    logger.info("PHASE 1: DATA INTEGRATION")
    logger.info("=" * 80)

    integrator = DataIntegrator(config=config.get('integration', {}))

    integrated_df = integrator.integrate_all(
        iar_path=config['data']['raw']['iar_transactions'],
        mbar_path=config['data']['raw']['mbar_customers'],
        fraud_path=config['data']['raw']['fraud_labels'],
        output_path=config['data']['processed']['integrated']
    )

    logger.info(f"Data integration complete. Shape: {integrated_df.shape}")
    logger.info(f"Saved to: {config['data']['processed']['integrated']}")

    return integrated_df


def run_data_cleaning(config: dict, integrated_df=None) -> None:
    """Run data cleaning phase"""
    logger.info("=" * 80)
    logger.info("PHASE 2: DATA CLEANING")
    logger.info("=" * 80)

    cleaner = DataCleaner()

    # Load integrated data if not provided
    if integrated_df is None:
        import pandas as pd
        integrated_df = pd.read_csv(config['data']['processed']['integrated'])

    clean_df = cleaner.clean_all(integrated_df, generate_report=True)

    # Save cleaned data
    clean_df.to_csv(config['data']['processed']['cleaned'], index=False)
    logger.info(f"Data cleaning complete. Shape: {clean_df.shape}")
    logger.info(f"Saved to: {config['data']['processed']['cleaned']}")

    return clean_df


def run_feature_engineering(config: dict, clean_df=None) -> None:
    """Run feature engineering phase"""
    logger.info("=" * 80)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 80)

    engineer = FeatureEngineer()

    # Load cleaned data if not provided
    if clean_df is None:
        import pandas as pd
        clean_df = pd.read_csv(config['data']['processed']['cleaned'])

    # Get feature engineering config
    fe_config = config.get('feature_engineering', {})
    aggregation_config = fe_config.get('aggregation', {})

    feature_df = engineer.engineer_all_features(
        clean_df,
        include_aggregations=aggregation_config.get('enabled', True),
        aggregation_windows=aggregation_config.get('windows', [1, 7, 30])
    )

    # Save feature-engineered data
    feature_df.to_csv(config['data']['features']['engineered'], index=False)
    logger.info(f"Feature engineering complete. Shape: {feature_df.shape}")
    logger.info(f"Saved to: {config['data']['features']['engineered']}")

    return feature_df


def run_feature_selection(config: dict, feature_df=None) -> None:
    """Run feature selection phase"""
    logger.info("=" * 80)
    logger.info("PHASE 4: FEATURE SELECTION")
    logger.info("=" * 80)

    # Get target column from config
    target_col = config.get('modeling', {}).get('target', 'is_fraud')
    selector = FeatureSelector(target_col=target_col)

    # Load feature-engineered data if not provided
    if feature_df is None:
        import pandas as pd
        feature_df = pd.read_csv(config['data']['features']['engineered'])

    # Get feature selection config
    fs_config = config.get('feature_selection', {})
    n_features = fs_config.get('final', {}).get('n_features', 50)

    # Perform multi-stage feature selection
    selected_features = selector.select_features_multistage(
        feature_df,
        target_col=target_col,
        n_final_features=n_features
    )

    # Save selected features
    selected_features_path = config['data']['features']['selected']
    with open(selected_features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    logger.info(f"Feature selection complete. Selected {len(selected_features)} features")
    logger.info(f"Saved to: {selected_features_path}")

    # Validate feature sets
    feature_sets = {
        'top_20': selected_features[:20],
        'top_30': selected_features[:30],
        'top_50': selected_features,
    }

    validation_results = selector.validate_feature_sets(
        feature_df,
        feature_sets,
        target_col=target_col,
        cv=5
    )

    # Save validation results
    validation_path = 'data/features/feature_validation_results.csv'
    validation_results.to_csv(validation_path, index=False)
    logger.info(f"Feature validation results saved to: {validation_path}")

    return selected_features


def main():
    """Main pipeline orchestrator"""
    parser = argparse.ArgumentParser(description='JazzCash Fraud Detection Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['all', 'integration', 'cleaning', 'feature_engineering', 'feature_selection'],
        default='all',
        help='Which phase to run'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    logger.info("JazzCash Fraud Detection Pipeline")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Running phase: {args.phase}")

    # Create output directories if they don't exist
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/features').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)

    # Run selected phase(s)
    if args.phase == 'all':
        # Run complete pipeline
        integrated_df = run_data_integration(config)
        clean_df = run_data_cleaning(config, integrated_df)
        feature_df = run_feature_engineering(config, clean_df)
        selected_features = run_feature_selection(config, feature_df)

    elif args.phase == 'integration':
        run_data_integration(config)

    elif args.phase == 'cleaning':
        run_data_cleaning(config)

    elif args.phase == 'feature_engineering':
        run_feature_engineering(config)

    elif args.phase == 'feature_selection':
        run_feature_selection(config)

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
