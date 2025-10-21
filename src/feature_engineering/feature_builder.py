"""
Feature Engineering Module for JazzCash Fraud Detection
Creates transaction, aggregation, and behavioral features
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for fraud detection"""

    def __init__(self):
        self.feature_catalog = {}

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from transaction time"""
        logger.info("Creating temporal features...")
        df = df.copy()

        if 'time' not in df.columns:
            logger.warning("'time' column not found, skipping temporal features")
            return df

        # Ensure datetime type
        df['time'] = pd.to_datetime(df['time'])

        # Time components
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['year'] = df['time'].dt.year

        # Binary indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)  # 12 AM - 6 AM
        df['is_evening'] = df['hour'].isin(range(18, 24)).astype(int)  # 6 PM - 12 AM
        df['is_business_hours'] = df['hour'].isin(range(9, 18)).astype(int)  # 9 AM - 6 PM

        # Cyclical encoding for hour (preserves circular nature)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week cyclical encoding
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        logger.info("Created 16 temporal features")

        return df

    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-level features"""
        logger.info("Creating transaction-level features...")
        df = df.copy()

        if 'amount' in df.columns:
            # Amount transformations
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_sqrt'] = np.sqrt(df['amount'])

            # Round amount indicators
            df['amount_round_100'] = (df['amount'] % 100 == 0).astype(int)
            df['amount_round_1000'] = (df['amount'] % 1000 == 0).astype(int)

            # Amount-balance ratio
            if 'balance' in df.columns:
                df['amount_to_balance_ratio'] = df['amount'] / (df['balance'] + 1)
                df['balance_after_txn'] = df['balance'] - df['amount']
                df['balance_after_txn_log'] = np.log1p(df['balance_after_txn'].clip(lower=0))

        # Customer tenure features
        if 'time' in df.columns and 'registration_date_sender' in df.columns:
            df['sender_tenure_days'] = (
                pd.to_datetime(df['time']) - pd.to_datetime(df['registration_date_sender'])
            ).dt.days

            df['sender_tenure_months'] = df['sender_tenure_days'] / 30.0
            df['is_new_sender'] = (df['sender_tenure_days'] < 30).astype(int)
            df['is_very_new_sender'] = (df['sender_tenure_days'] < 7).astype(int)

        if 'time' in df.columns and 'registration_date_receiver' in df.columns:
            df['receiver_tenure_days'] = (
                pd.to_datetime(df['time']) - pd.to_datetime(df['registration_date_receiver'])
            ).dt.days

            df['receiver_tenure_months'] = df['receiver_tenure_days'] / 30.0
            df['is_new_receiver'] = (df['receiver_tenure_days'] < 30).astype(int)

        logger.info(f"Created {df.shape[1] - df.shape[1]} transaction features")

        return df

    def create_aggregation_features(self,
                                   df: pd.DataFrame,
                                   windows: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """Create aggregation features over time windows"""
        logger.info("Creating aggregation features...")
        df = df.copy()

        # Ensure data is sorted by time
        df = df.sort_values('time')

        for window in windows:
            logger.info(f"  Processing {window}-day window...")

            # Sender aggregations
            if 'sender' in df.columns:
                # Transaction count
                df[f'sender_txn_count_{window}d'] = df.groupby('sender').rolling(
                    f'{window}D', on='time'
                )['tid'].count().reset_index(drop=True)

                # Total amount sent
                if 'amount' in df.columns:
                    df[f'sender_amount_sum_{window}d'] = df.groupby('sender').rolling(
                        f'{window}D', on='time'
                    )['amount'].sum().reset_index(drop=True)

                    df[f'sender_amount_avg_{window}d'] = df.groupby('sender').rolling(
                        f'{window}D', on='time'
                    )['amount'].mean().reset_index(drop=True)

                    df[f'sender_amount_max_{window}d'] = df.groupby('sender').rolling(
                        f'{window}D', on='time'
                    )['amount'].max().reset_index(drop=True)

                    df[f'sender_amount_std_{window}d'] = df.groupby('sender').rolling(
                        f'{window}D', on='time'
                    )['amount'].std().reset_index(drop=True)

                # Unique receivers
                if 'receiver' in df.columns:
                    df[f'sender_unique_receivers_{window}d'] = df.groupby('sender').rolling(
                        f'{window}D', on='time'
                    )['receiver'].apply(lambda x: x.nunique()).reset_index(drop=True)

            # Receiver aggregations
            if 'receiver' in df.columns:
                # Transaction count
                df[f'receiver_txn_count_{window}d'] = df.groupby('receiver').rolling(
                    f'{window}D', on='time'
                )['tid'].count().reset_index(drop=True)

                # Unique senders
                if 'sender' in df.columns:
                    df[f'receiver_unique_senders_{window}d'] = df.groupby('receiver').rolling(
                        f'{window}D', on='time'
                    )['sender'].apply(lambda x: x.nunique()).reset_index(drop=True)

                # Total amount received
                if 'amount' in df.columns:
                    df[f'receiver_amount_sum_{window}d'] = df.groupby('receiver').rolling(
                        f'{window}D', on='time'
                    )['amount'].sum().reset_index(drop=True)

        # Velocity features (change detection)
        if 'sender_amount_sum_1d' in df.columns and 'sender_amount_avg_30d' in df.columns:
            df['amount_spike'] = df['sender_amount_sum_1d'] / (df['sender_amount_avg_30d'] * 30 + 1)

        if 'sender_txn_count_1d' in df.columns and 'sender_txn_count_30d' in df.columns:
            df['txn_frequency_spike'] = df['sender_txn_count_1d'] / (df['sender_txn_count_30d'] / 30 + 1)

        logger.info(f"Created aggregation features for {len(windows)} time windows")

        return df

    def create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create network-based features"""
        logger.info("Creating network features...")
        df = df.copy()

        if 'sender' in df.columns and 'receiver' in df.columns:
            # Sender-receiver relationship history
            df = df.sort_values('time')
            df['sender_receiver_prev_txns'] = df.groupby(
                ['sender', 'receiver']
            ).cumcount()

            df['is_first_time_receiver'] = (df['sender_receiver_prev_txns'] == 0).astype(int)

            # Sender degree (how many unique receivers)
            sender_degree = df.groupby('sender')['receiver'].nunique().to_dict()
            df['sender_degree'] = df['sender'].map(sender_degree)

            # Receiver degree (how many unique senders)
            receiver_degree = df.groupby('receiver')['sender'].nunique().to_dict()
            df['receiver_degree'] = df['receiver'].map(receiver_degree)

            # High degree flags (potential mule accounts)
            df['is_high_degree_sender'] = (df['sender_degree'] > df['sender_degree'].quantile(0.95)).astype(int)
            df['is_high_degree_receiver'] = (df['receiver_degree'] > df['receiver_degree'].quantile(0.95)).astype(int)

        logger.info("Created network features")

        return df

    def create_categorical_encoding(self,
                                    df: pd.DataFrame,
                                    categorical_cols: List[str] = None) -> pd.DataFrame:
        """Create frequency and target encoding for categorical features"""
        logger.info("Creating categorical encodings...")
        df = df.copy()

        if categorical_cols is None:
            categorical_cols = ['channel', 'type', 'location', 'account_type_sender']

        # Frequency encoding
        for col in categorical_cols:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_freq'] = df[col].map(freq)

                # Rare category flag
                rare_threshold = 0.01
                df[f'{col}_is_rare'] = (df[f'{col}_freq'] < rare_threshold).astype(int)

        logger.info(f"Created frequency encodings for {len(categorical_cols)} columns")

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        logger.info("Creating interaction features...")
        df = df.copy()

        # Channel × Type interaction
        if 'channel' in df.columns and 'type' in df.columns:
            df['channel_type'] = df['channel'].astype(str) + '_' + df['type'].astype(str)

        # High amount indicators by channel
        if 'amount' in df.columns and 'channel' in df.columns:
            high_amount_threshold = df['amount'].quantile(0.9)
            df['high_amount_mobile'] = (
                (df['amount'] > high_amount_threshold) &
                (df['channel'] == 'MOBILE')
            ).astype(int)

            df['high_amount_web'] = (
                (df['amount'] > high_amount_threshold) &
                (df['channel'] == 'WEB')
            ).astype(int)

        # Age × Amount interaction
        if 'age' in df.columns and 'amount_log' in df.columns:
            df['age_amount_interaction'] = df['age'] * df['amount_log']

        # New account × High amount
        if 'is_new_sender' in df.columns and 'amount' in df.columns:
            high_amount_threshold = df['amount'].quantile(0.9)
            df['new_account_high_amount'] = (
                (df['is_new_sender'] == 1) &
                (df['amount'] > high_amount_threshold)
            ).astype(int)

        logger.info("Created interaction features")

        return df

    def engineer_all_features(self,
                             df: pd.DataFrame,
                             include_aggregations: bool = True,
                             aggregation_windows: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")

        initial_features = df.shape[1]

        # Step 1: Temporal features
        df = self.create_temporal_features(df)

        # Step 2: Transaction features
        df = self.create_transaction_features(df)

        # Step 3: Aggregation features (optional, computationally expensive)
        if include_aggregations:
            df = self.create_aggregation_features(df, windows=aggregation_windows)

        # Step 4: Network features
        df = self.create_network_features(df)

        # Step 5: Categorical encoding
        df = self.create_categorical_encoding(df)

        # Step 6: Interaction features
        df = self.create_interaction_features(df)

        final_features = df.shape[1]
        new_features = final_features - initial_features

        logger.info(f"Feature engineering complete: {initial_features} -> {final_features} features")
        logger.info(f"Created {new_features} new features")

        return df


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()

    # Load cleaned dataset
    df = pd.read_csv('data/processed/cleaned_dataset.csv')

    # Engineer features
    feature_df = engineer.engineer_all_features(
        df,
        include_aggregations=True,
        aggregation_windows=[1, 7, 30]
    )

    # Save feature-engineered dataset
    feature_df.to_csv('data/features/feature_engineered_dataset.csv', index=False)
    logger.info("Feature-engineered dataset saved")
