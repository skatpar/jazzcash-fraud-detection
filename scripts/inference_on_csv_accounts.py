#!/usr/bin/env python3
"""
Inference script for specific accounts from high_risk_transactions_with_details.csv

This script:
1. Extracts distinct ac_to values from the CSV file
2. Loads complete features for those accounts from ClickHouse
3. Runs inference using the trained model
4. Maps predictions back to the original accounts
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add the scripts directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from fraud_inference import FraudInferenceEngine
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when


def extract_distinct_accounts(csv_path):
    """Extract distinct ac_to values from CSV file"""
    print(f"📁 Reading CSV file: {csv_path}")
    
    # Read CSV with pandas for quick processing
    df = pd.read_csv(csv_path)
    print(f"   • Total rows in CSV: {len(df):,}")
    
    # Extract distinct ac_to values, filtering out NaN/null values
    distinct_accounts = df['ac_to'].dropna().unique()
    
    # Filter out any empty strings or non-string values
    distinct_accounts = [str(acc) for acc in distinct_accounts if pd.notna(acc) and str(acc).strip() != '']
    
    print(f"   • Distinct ac_to accounts: {len(distinct_accounts):,}")
    
    return distinct_accounts


def load_features_for_accounts(engine, account_list, date_range_days=30):
    """Load features from ClickHouse for specific accounts"""
    print(f"🗄️ Loading features for {len(account_list)} accounts from ClickHouse...")
    
    # Use the actual date range available in the ClickHouse table
    start_date_str = '2025-07-01'  # Use July data which is recent and available
    end_date_str = '2025-07-31'    # Latest available data
    
    print(f"   • Date range: {start_date_str} to {end_date_str}")
    
    # Get expected columns from model metadata
    if not engine.feature_metadata:
        engine.load_model_components()
    
    # Get feature columns but ensure cutoff_date is not duplicated
    feature_cols = engine.feature_metadata['all_feature_cols'].copy()
    if 'cutoff_date' in feature_cols:
        feature_cols.remove('cutoff_date')
    
    selected_cols = ['cutoff_date'] + feature_cols
    
    # ClickHouse configuration
    CLICKHOUSE_CONFIG = {
        'host': 'localhost',
        'port': 9000,
        'database': 'public',
        'user': 'default',
        'password': 'DfsTeChB1'
    }
    
    url = f"jdbc:ch://{CLICKHOUSE_CONFIG['host']}:8123/{CLICKHOUSE_CONFIG['database']}"
    user = CLICKHOUSE_CONFIG['user']
    password = CLICKHOUSE_CONFIG['password']
    driver = "com.clickhouse.jdbc.ClickHouseDriver"
    
    # Create a list of account IDs for the SQL IN clause
    # Format accounts for SQL (escape single quotes) - ensure all are strings
    valid_accounts = [str(acc) for acc in account_list if pd.notna(acc) and str(acc).strip() != '']
    formatted_accounts = [f"'{acc.replace(chr(39), chr(39)+chr(39))}'" for acc in valid_accounts]
    accounts_in_clause = ', '.join(formatted_accounts)
    
    if not formatted_accounts:
        print("❌ No valid account IDs found after filtering")
        return None
    
    query = f"""
        SELECT {', '.join(selected_cols)}
        FROM stixor_fraud_features_distributed
        WHERE cutoff_date BETWEEN '{start_date_str}' AND '{end_date_str}'
            AND mbar_account_type_name = 'Customer Account'
            AND ac_to IN ({accounts_in_clause})
    """
    
    subquery = f"({query}) AS account_features"
    
    try:
        df = (engine.spark.read
            .format('jdbc')
            .option('driver', driver)
            .option('url', url)
            .option('user', user)
            .option('password', password)
            .option('dbtable', subquery)
            .load())
        
        # Cache the DataFrame
        df.cache()
        
        # Get row count
        total_rows = df.count()
        print(f"   ✅ Loaded {total_rows:,} feature records from ClickHouse")
        
        if total_rows == 0:
            print("   ⚠️  No features found for the specified accounts in the date range")
            print("   💡 Try expanding the date range or checking if accounts exist in the table")
            return None
        
        # Show account distribution
        account_counts = df.groupBy('ac_to').count().orderBy(col('count').desc())
        print(f"   • Found features for {account_counts.count()} unique accounts")
        print("   • Top accounts by feature count:")
        account_counts.show(5)
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading features from ClickHouse: {str(e)}")
        raise


def run_inference_for_csv_accounts(csv_path, output_path=None, date_range_days=30):
    """Main function to run inference for accounts from CSV"""
    print("🚀 Starting inference for high-risk transaction accounts")
    print("=" * 60)
    
    # Initialize inference engine
    engine = FraudInferenceEngine()
    
    try:
        # Step 1: Extract distinct accounts from CSV
        print("\n📋 Step 1: Extract accounts from CSV")
        distinct_accounts = extract_distinct_accounts(csv_path)
        
        if len(distinct_accounts) == 0:
            print("❌ No accounts found in CSV file")
            return None
        
        # Step 2: Load features for these accounts from ClickHouse
        print(f"\n📋 Step 2: Load features from ClickHouse")
        features_df = load_features_for_accounts(engine, distinct_accounts, date_range_days)
        
        if features_df is None:
            return None
        
        # Step 3: Load model components
        print(f"\n📋 Step 3: Load model components")
        engine.load_model_components()
        
        # Step 4: Preprocess data
        print(f"\n📋 Step 4: Preprocess data")
        preprocessed_df = engine.preprocess_data(features_df)
        
        # Step 5: Make predictions
        print(f"\n📋 Step 5: Make predictions")
        predictions_df = engine.make_predictions(preprocessed_df, features_df)
        
        # Step 6: Process results
        print(f"\n📋 Step 6: Process results")
        
        # Add account mapping and sort by fraud probability
        results_df = predictions_df.orderBy(col('fraud_probability').desc())
        
        # Show summary statistics
        total_predictions = results_df.count()
        fraud_predictions = results_df.filter(col('prediction') == 1.0).count()
        high_risk_count = results_df.filter(col('fraud_probability') > 0.7).count()
        medium_risk_count = results_df.filter((col('fraud_probability') > 0.3) & (col('fraud_probability') <= 0.7)).count()
        
        print(f"\n📊 Prediction Summary:")
        print(f"   • Total predictions: {total_predictions:,}")
        print(f"   • Fraud predictions (>0.5): {fraud_predictions:,}")
        print(f"   • High risk (>0.7): {high_risk_count:,}")
        print(f"   • Medium risk (0.3-0.7): {medium_risk_count:,}")
        print(f"   • Fraud rate: {(fraud_predictions/total_predictions)*100:.2f}%")
        
        # Show top high-risk accounts
        print(f"\n🚨 Top 10 Highest Risk Accounts:")
        top_risk = results_df.select('ac_to', 'cutoff_date', 'prediction', 'fraud_probability') \
                            .orderBy(col('fraud_probability').desc())
        top_risk.show(10, truncate=False)
        
        # Step 7: Save results
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"high_risk_account_predictions_{timestamp}.csv"
        
        print(f"\n📋 Step 7: Save predictions")
        engine.save_predictions(results_df, output_path)
        
        # Create summary report
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("High-Risk Account Predictions Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source CSV: {csv_path}\n")
            f.write(f"Date range: {date_range_days} days\n\n")
            f.write(f"Total accounts analyzed: {len(distinct_accounts):,}\n")
            f.write(f"Total predictions: {total_predictions:,}\n")
            f.write(f"Fraud predictions: {fraud_predictions:,}\n")
            f.write(f"High risk accounts (>0.7): {high_risk_count:,}\n")
            f.write(f"Medium risk accounts (0.3-0.7): {medium_risk_count:,}\n")
            f.write(f"Overall fraud rate: {(fraud_predictions/total_predictions)*100:.2f}%\n")
        
        print(f"✅ Summary saved to: {summary_path}")
        print(f"🎯 Inference completed successfully!")
        print(f"📁 Predictions saved to: {output_path}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in inference pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on high-risk transaction accounts')
    parser.add_argument('--csv_path', type=str, 
                       default='high_risk_transactions_with_details.csv',
                       help='Path to the CSV file with high-risk transactions')
    parser.add_argument('--output_path', type=str, 
                       help='Path to save predictions (default: auto-generated)')
    parser.add_argument('--date_range_days', type=int, default=30,
                       help='Number of days to look back for features (default: 30)')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"❌ CSV file not found: {args.csv_path}")
        return
    
    # Run inference
    predictions = run_inference_for_csv_accounts(
        csv_path=args.csv_path,
        output_path=args.output_path,
        date_range_days=args.date_range_days
    )
    
    if predictions is not None:
        print("\n✅ Process completed successfully!")
    else:
        print("\n❌ Process failed!")


if __name__ == "__main__":
    main()