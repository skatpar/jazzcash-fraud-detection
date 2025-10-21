"""
Data Cleaning Module for JazzCash Fraud Detection
Handles missing values, outliers, and data quality issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning and quality improvement"""

    def __init__(self):
        self.cleaning_stats = {}
        self.imputation_values = {}

    def generate_missing_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive missing value report"""
        logger.info("Generating missing value profile...")

        missing_profile = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_pct': (df.isnull().sum() / len(df) * 100).values,
            'dtype': df.dtypes.values
        })

        missing_profile = missing_profile[missing_profile['missing_count'] > 0].sort_values(
            'missing_pct', ascending=False
        )

        logger.info(f"Found {len(missing_profile)} columns with missing values")

        return missing_profile

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using feature-specific strategies"""
        logger.info("Imputing missing values...")
        df = df.copy()

        # Numerical features
        numerical_strategies = {
            'balance': ('forward_fill', 'median'),  # Forward fill by customer, then median
            'age': ('median', 'location'),  # Median by location
            'limits': ('median', 'account_type_sender'),  # Median by account type
        }

        for col, (method, group_by) in numerical_strategies.items():
            if col in df.columns and df[col].isnull().any():
                if method == 'forward_fill' and group_by in df.columns:
                    # Forward fill within groups
                    df[col] = df.groupby(group_by)[col].fillna(method='ffill')
                    # Fill remaining with overall median
                    df[col] = df[col].fillna(df[col].median())
                elif method == 'median' and group_by in df.columns:
                    # Median by group
                    group_medians = df.groupby(group_by)[col].transform('median')
                    df[col] = df[col].fillna(group_medians)
                    # Fill remaining with overall median
                    df[col] = df[col].fillna(df[col].median())

                logger.info(f"Imputed {col} using {method}")

        # Categorical features
        categorical_strategies = {
            'channel': 'mode',
            'type': 'UNKNOWN',
            'location': 'UNSPECIFIED',
            'kyc_status_sender': 'NOT_VERIFIED',
            'kyc_status_receiver': 'NOT_VERIFIED',
            'account_type_sender': 'mode',
            'account_type_receiver': 'mode',
        }

        for col, strategy in categorical_strategies.items():
            if col in df.columns and df[col].isnull().any():
                if strategy == 'mode':
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN'
                    df[col] = df[col].fillna(mode_value)
                else:
                    df[col] = df[col].fillna(strategy)

                logger.info(f"Imputed {col} using {strategy}")

        # Create missingness indicators for important features
        missing_indicators = ['balance', 'location', 'kyc_status_sender', 'age']
        for col in missing_indicators:
            if col in df.columns:
                indicator_col = f'missing_{col}'
                df[indicator_col] = df[col].isnull().astype(int)

        return df

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str, multiplier: float = 3.0) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        return outliers

    def detect_and_flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and flag outliers without removing them"""
        logger.info("Detecting and flagging outliers...")
        df = df.copy()

        # Amount outliers (using IQR)
        if 'amount' in df.columns:
            df['is_amount_outlier'] = self.detect_outliers_iqr(df, 'amount', multiplier=3.0).astype(int)
            outlier_count = df['is_amount_outlier'].sum()
            logger.info(f"Flagged {outlier_count} amount outliers ({outlier_count/len(df):.2%})")

        # Balance outliers
        if 'balance' in df.columns:
            df['is_balance_outlier'] = self.detect_outliers_iqr(df, 'balance', multiplier=3.0).astype(int)
            df['is_negative_balance'] = (df['balance'] < 0).astype(int)

        # Age outliers
        if 'age' in df.columns:
            df['is_age_outlier'] = ((df['age'] < 13) | (df['age'] > 100)).astype(int)

        # Multivariate outliers using Isolation Forest
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col not in
                       ['tid', 'is_fraud', 'is_amount_outlier', 'is_balance_outlier', 'is_age_outlier']]

        if len(feature_cols) >= 2:
            # Prepare data for Isolation Forest (drop NaN)
            iso_data = df[feature_cols].dropna()
            if len(iso_data) > 0:
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                df.loc[iso_data.index, 'anomaly_score'] = iso_forest.fit_predict(iso_data)
                df['anomaly_score'] = df['anomaly_score'].fillna(1)  # 1 = normal, -1 = anomaly
                df['is_multivariate_anomaly'] = (df['anomaly_score'] == -1).astype(int)

                anomaly_count = df['is_multivariate_anomaly'].sum()
                logger.info(f"Detected {anomaly_count} multivariate anomalies ({anomaly_count/len(df):.2%})")

        return df

    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats and values"""
        logger.info("Standardizing data...")
        df = df.copy()

        # Standardize amount precision
        if 'amount' in df.columns:
            df['amount'] = df['amount'].round(2)

        # Standardize categorical values
        categorical_mappings = {
            'channel': {
                'MOBILE': 'MOBILE',
                'mobile': 'MOBILE',
                'Mobile': 'MOBILE',
                'WEB': 'WEB',
                'web': 'WEB',
                'Web': 'WEB',
                'AGENT': 'AGENT',
                'agent': 'AGENT',
                'Agent': 'AGENT',
                'ATM': 'ATM',
                'atm': 'ATM',
            },
            'type': {
                'TRANSFER': 'TRANSFER',
                'transfer': 'TRANSFER',
                'PAYMENT': 'PAYMENT',
                'payment': 'PAYMENT',
                'WITHDRAWAL': 'WITHDRAWAL',
                'withdrawal': 'WITHDRAWAL',
                'DEPOSIT': 'DEPOSIT',
                'deposit': 'DEPOSIT',
            }
        }

        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        return df

    def apply_data_quality_rules(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply hard and soft data quality rules"""
        logger.info("Applying data quality rules...")
        df = df.copy()

        initial_count = len(df)

        # Hard rules (drop records)
        hard_rules = []

        # Rule 1: Missing critical fields
        if 'tid' in df.columns:
            hard_rules.append(df['tid'].notna())
        if 'time' in df.columns:
            hard_rules.append(df['time'].notna())
        if 'amount' in df.columns:
            hard_rules.append(df['amount'].notna())

        # Rule 2: Invalid amounts
        if 'amount' in df.columns:
            hard_rules.append(df['amount'] > 0)

        # Rule 3: Duplicate transaction IDs
        if 'tid' in df.columns:
            df = df.drop_duplicates(subset='tid', keep='first')

        # Apply all hard rules
        if hard_rules:
            mask = pd.concat(hard_rules, axis=1).all(axis=1)
            removed_df = df[~mask]
            df = df[mask]

            removed_count = initial_count - len(df)
            logger.info(f"Removed {removed_count} records due to hard rules ({removed_count/initial_count:.2%})")

        # Soft rules (flag for review)
        if 'time' in df.columns and 'registration_date_sender' in df.columns:
            df['flag_invalid_temporal_order'] = (df['time'] < df['registration_date_sender']).astype(int)

        if 'amount' in df.columns:
            high_value_threshold = df['amount'].quantile(0.999)
            df['flag_high_value_transaction'] = (df['amount'] > high_value_threshold).astype(int)

        return df, removed_df

    def clean_all(self, df: pd.DataFrame, generate_report: bool = True) -> pd.DataFrame:
        """Complete data cleaning pipeline"""
        logger.info("Starting data cleaning pipeline...")

        initial_shape = df.shape

        # Step 1: Generate missing value profile
        if generate_report:
            missing_profile = self.generate_missing_profile(df)
            print("\nMissing Value Profile:")
            print(missing_profile.to_string(index=False))

        # Step 2: Impute missing values
        df = self.impute_missing_values(df)

        # Step 3: Standardize data
        df = self.standardize_data(df)

        # Step 4: Detect and flag outliers
        df = self.detect_and_flag_outliers(df)

        # Step 5: Apply data quality rules
        df, removed_df = self.apply_data_quality_rules(df)

        final_shape = df.shape

        logger.info(f"Cleaning complete: {initial_shape} -> {final_shape}")
        logger.info(f"Removed {initial_shape[0] - final_shape[0]} rows ({(initial_shape[0] - final_shape[0])/initial_shape[0]:.2%})")

        return df


if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()

    # Load integrated dataset
    df = pd.read_csv('data/processed/integrated_dataset.csv')

    # Clean data
    clean_df = cleaner.clean_all(df, generate_report=True)

    # Save cleaned dataset
    clean_df.to_csv('data/processed/cleaned_dataset.csv', index=False)
    logger.info("Cleaned dataset saved")
