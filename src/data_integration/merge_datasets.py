"""
Data Integration Module for JazzCash Fraud Detection
Merges IAR transactions, Mbar customer data, and fraud labels
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrator:
    """Handles merging of multiple datasets for fraud detection"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.merge_stats = {}

    def load_datasets(self,
                     iar_path: str,
                     mbar_path: str,
                     fraud_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three datasets"""
        logger.info("Loading datasets...")

        iar_df = pd.read_csv(iar_path)
        mbar_df = pd.read_csv(mbar_path)
        fraud_df = pd.read_csv(fraud_path)

        logger.info(f"IAR transactions: {len(iar_df):,} rows")
        logger.info(f"Mbar customers: {len(mbar_df):,} rows")
        logger.info(f"Fraud labels: {len(fraud_df):,} rows")

        return iar_df, mbar_df, fraud_df

    def standardize_schemas(self,
                           iar_df: pd.DataFrame,
                           mbar_df: pd.DataFrame,
                           fraud_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Standardize column names and data types"""
        logger.info("Standardizing schemas...")

        # Convert datetime columns
        if 'time' in iar_df.columns:
            iar_df['time'] = pd.to_datetime(iar_df['time'])
        if 'registration_date' in mbar_df.columns:
            mbar_df['registration_date'] = pd.to_datetime(mbar_df['registration_date'])

        # Standardize string columns to lowercase
        for df in [iar_df, mbar_df, fraud_df]:
            for col in df.select_dtypes(include=['object']).columns:
                if col not in ['time', 'registration_date']:
                    df[col] = df[col].astype(str).str.strip().str.upper()

        # Ensure consistent customer ID naming
        if 'customer_id' not in mbar_df.columns and 'sender' in mbar_df.columns:
            mbar_df = mbar_df.rename(columns={'sender': 'customer_id'})

        return iar_df, mbar_df, fraud_df

    def merge_fraud_labels(self,
                          iar_df: pd.DataFrame,
                          fraud_df: pd.DataFrame) -> pd.DataFrame:
        """Merge fraud labels with IAR transactions"""
        logger.info("Merging fraud labels...")

        # Create fraud indicator in fraud dataset
        fraud_df['is_fraud'] = 1

        # Left join IAR with fraud labels
        merged_df = iar_df.merge(
            fraud_df[['tid', 'is_fraud']],
            on='tid',
            how='left',
            indicator=True
        )

        # Fill NaN fraud labels with 0 (legitimate transactions)
        merged_df['is_fraud'] = merged_df['is_fraud'].fillna(0).astype(int)

        # Calculate merge statistics
        self.merge_stats['total_transactions'] = len(merged_df)
        self.merge_stats['fraud_transactions'] = merged_df['is_fraud'].sum()
        self.merge_stats['fraud_rate'] = merged_df['is_fraud'].mean()

        logger.info(f"Fraud rate: {self.merge_stats['fraud_rate']:.4%}")

        # Drop merge indicator
        merged_df = merged_df.drop('_merge', axis=1)

        return merged_df

    def enrich_with_customer_data(self,
                                  trans_df: pd.DataFrame,
                                  mbar_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich transactions with sender and receiver customer data"""
        logger.info("Enriching with customer data...")

        # Ensure customer_id column exists in mbar
        if 'customer_id' not in mbar_df.columns:
            raise ValueError("mbar_df must have 'customer_id' column")

        # Merge sender information
        enriched_df = trans_df.merge(
            mbar_df,
            left_on='sender',
            right_on='customer_id',
            how='left',
            suffixes=('', '_sender')
        )

        # Calculate sender match rate
        sender_match_rate = enriched_df['customer_id'].notna().mean()
        self.merge_stats['sender_match_rate'] = sender_match_rate
        logger.info(f"Sender match rate: {sender_match_rate:.2%}")

        # Merge receiver information
        enriched_df = enriched_df.merge(
            mbar_df,
            left_on='receiver',
            right_on='customer_id',
            how='left',
            suffixes=('_sender', '_receiver')
        )

        # Calculate receiver match rate
        receiver_match_rate = enriched_df['customer_id_receiver'].notna().mean()
        self.merge_stats['receiver_match_rate'] = receiver_match_rate
        logger.info(f"Receiver match rate: {receiver_match_rate:.2%}")

        return enriched_df

    def validate_integration(self, df: pd.DataFrame) -> bool:
        """Validate the integrated dataset"""
        logger.info("Validating integrated dataset...")

        # Check 1: No duplicate transaction IDs
        assert df['tid'].is_unique, "Duplicate transaction IDs found!"
        logger.info("✓ No duplicate transaction IDs")

        # Check 2: All transactions have fraud labels
        assert df['is_fraud'].notna().all(), "Missing fraud labels!"
        logger.info("✓ All transactions have fraud labels")

        # Check 3: Temporal consistency (transaction time >= sender registration)
        if 'registration_date_sender' in df.columns:
            invalid_temporal = df[df['time'] < df['registration_date_sender']]
            if len(invalid_temporal) > 0:
                logger.warning(f"Found {len(invalid_temporal)} transactions before sender registration")
                df['invalid_temporal_order'] = (df['time'] < df['registration_date_sender']).astype(int)
            else:
                logger.info("✓ Temporal consistency validated")

        # Check 4: Data coverage
        total_rows = len(df)
        complete_rows = df.dropna().shape[0]
        completeness = complete_rows / total_rows
        logger.info(f"Data completeness: {completeness:.2%}")

        return True

    def integrate_all(self,
                     iar_path: str,
                     mbar_path: str,
                     fraud_path: str,
                     output_path: str = None) -> pd.DataFrame:
        """Complete integration pipeline"""

        # Load datasets
        iar_df, mbar_df, fraud_df = self.load_datasets(iar_path, mbar_path, fraud_path)

        # Standardize schemas
        iar_df, mbar_df, fraud_df = self.standardize_schemas(iar_df, mbar_df, fraud_df)

        # Merge fraud labels
        merged_df = self.merge_fraud_labels(iar_df, fraud_df)

        # Enrich with customer data
        enriched_df = self.enrich_with_customer_data(merged_df, mbar_df)

        # Validate
        self.validate_integration(enriched_df)

        # Save if output path provided
        if output_path:
            logger.info(f"Saving integrated dataset to {output_path}")
            enriched_df.to_csv(output_path, index=False)

        logger.info("Integration complete!")
        logger.info(f"Final dataset shape: {enriched_df.shape}")

        return enriched_df


if __name__ == "__main__":
    # Example usage
    integrator = DataIntegrator()

    # Update paths to your actual data files
    integrated_df = integrator.integrate_all(
        iar_path='data/raw/iar_transactions.csv',
        mbar_path='data/raw/mbar_customers.csv',
        fraud_path='data/raw/fraud_labels.csv',
        output_path='data/processed/integrated_dataset.csv'
    )

    print("\nIntegration Statistics:")
    for key, value in integrator.merge_stats.items():
        print(f"  {key}: {value}")
