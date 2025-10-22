"""
Prepare dummy dataset for pipeline

This script combines the generated dummy MBAR and IAR data similar to the extraction pipeline,
creating a ready-to-use dataset for the fraud detection pipeline.

Usage:
    python utils/prepare_dummy_dataset.py
"""

import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from datetime import datetime


def prepare_dummy_dataset(
    dummy_dir='data/dummy',
    output_dir='data/processed'
):
    """
    Prepare dummy dataset by combining IAR and MBAR data

    Args:
        dummy_dir: Directory containing dummy data
        output_dir: Output directory
    """
    print("=" * 80)
    print("Preparing Dummy Dataset for Pipeline")
    print("=" * 80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dummy data
    print("\n[1/5] Loading dummy data...")
    df_mbar = pd.read_parquet(f"{dummy_dir}/mbar_dummy.parquet")
    df_iar = pd.read_parquet(f"{dummy_dir}/iar_dummy.parquet")
    df_fraud = pd.read_parquet(f"{dummy_dir}/fraud_dummy.parquet")

    print(f"  ✓ Loaded {len(df_mbar):,} accounts")
    print(f"  ✓ Loaded {len(df_iar):,} transactions")
    print(f"  ✓ Loaded {len(df_fraud):,} fraud cases")

    # Join IAR with MBAR (sender side)
    print("\n[2/5] Joining transaction data with sender account info...")

    # Rename MBAR columns for sender
    df_mbar_from = df_mbar.copy()
    df_mbar_from = df_mbar_from.add_prefix('from_')
    df_mbar_from = df_mbar_from.rename(columns={'from_a_c_reference': 'ac_from'})

    df_combined = df_iar.merge(
        df_mbar_from,
        on='ac_from',
        how='left'
    )

    print(f"  ✓ Combined with sender info: {len(df_combined):,} rows")

    # Join with MBAR (receiver side)
    print("\n[3/5] Joining with receiver account info...")

    df_mbar_to = df_mbar.copy()
    df_mbar_to = df_mbar_to.add_prefix('to_')
    df_mbar_to = df_mbar_to.rename(columns={'to_a_c_reference': 'ac_to'})

    df_combined = df_combined.merge(
        df_mbar_to,
        on='ac_to',
        how='left'
    )

    print(f"  ✓ Combined with receiver info: {len(df_combined):,} rows")

    # Add fraud labels
    print("\n[4/5] Adding fraud labels...")

    # Create fraud lookup
    fraud_trans_ids = set(df_fraud['trans_id'].tolist())

    df_combined['is_fraud'] = df_combined['trans_id'].isin(fraud_trans_ids).astype(int)

    # Add fraud details
    df_fraud_minimal = df_fraud[['trans_id', 'complaint_num', 'victim_msisdn', 'fraud_msisdn']].copy()

    df_combined = df_combined.merge(
        df_fraud_minimal,
        on='trans_id',
        how='left'
    )

    fraud_count = df_combined['is_fraud'].sum()
    fraud_rate = fraud_count / len(df_combined) * 100

    print(f"  ✓ Fraud transactions: {fraud_count:,} ({fraud_rate:.4f}%)")
    print(f"  ✓ Non-fraud transactions: {(len(df_combined) - fraud_count):,}")

    # Save master dataset
    print(f"\n[5/5] Saving master dataset to {output_dir}...")

    # Save as parquet (partitioned by date and fraud status)
    output_path = f"{output_dir}/master_dataset.parquet"

    df_combined.to_parquet(
        output_path,
        index=False,
        partition_cols=['data_date', 'is_fraud']
    )

    print(f"  ✓ Saved: {output_path}")

    # Save summary statistics
    stats = {
        'preparation_date': datetime.now().isoformat(),
        'total_rows': len(df_combined),
        'total_columns': len(df_combined.columns),
        'fraud_cases': fraud_count,
        'non_fraud_cases': len(df_combined) - fraud_count,
        'fraud_rate': fraud_rate,
        'date_range': f"{df_combined['data_date'].min()} to {df_combined['data_date'].max()}",
        'unique_senders': df_combined['ac_from'].nunique(),
        'unique_receivers': df_combined['ac_to'].nunique(),
        'channels': df_combined['trx_channel'].unique().tolist(),
        'transaction_types': df_combined['trx_type'].unique().tolist()
    }

    with open(f"{output_dir}/dataset_stats.txt", 'w') as f:
        f.write("Master Dataset Statistics\n")
        f.write("=" * 60 + "\n\n")
        for key, value in stats.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"  ✓ Saved statistics: {output_dir}/dataset_stats.txt")

    print("\n" + "=" * 80)
    print("Dataset Preparation Complete!")
    print("=" * 80)
    print(f"\nDataset Summary:")
    print(f"  Total rows: {len(df_combined):,}")
    print(f"  Total columns: {len(df_combined.columns)}")
    print(f"  Fraud cases: {fraud_count:,} ({fraud_rate:.4f}%)")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Unique senders: {stats['unique_senders']:,}")
    print(f"  Unique receivers: {stats['unique_receivers']:,}")

    print(f"\n✓ Master dataset ready at: {output_path}")
    print("\nNext steps:")
    print("  1. Run cleaning pipeline: python pipelines/data_pipeline.py --skip-extraction")
    print("  2. Or run full pipeline: python pipelines/main_pipeline.py --mode dummy")

    return df_combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dummy dataset')
    parser.add_argument(
        '--dummy-dir',
        type=str,
        default='data/dummy',
        help='Directory containing dummy data (default: data/dummy)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory (default: data/processed)'
    )

    args = parser.parse_args()

    prepare_dummy_dataset(
        dummy_dir=args.dummy_dir,
        output_dir=args.output_dir
    )
