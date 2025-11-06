#!/usr/bin/env python3
"""
Usage Examples for Fraud Detection Inference

This script demonstrates different ways to use the fraud detection inference engine.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the scripts directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from fraud_inference import FraudInferenceEngine


def example_csv_inference():
    """Example: Run inference on CSV data"""
    print("📁 Example 1: CSV File Inference")
    print("=" * 50)
    
    # Initialize inference engine
    engine = FraudInferenceEngine()
    
    # Example CSV file path (replace with your actual CSV file)
    csv_path = "/root/research-dir/dev/jazzcash-fraud-detection/data/df_snapshot_full.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please provide a valid CSV file path with the required feature columns.")
        return
    
    try:
        # Run inference
        predictions = engine.run_inference(
            input_type='csv',
            input_path=csv_path,
            output_path=f'csv_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            limit=10000  # Limit for testing
        )
        
        print("✅ CSV inference completed successfully!")
        print(f"Predictions shape: {predictions.count()} rows")
        
        # Show sample predictions
        print("\n📊 Sample predictions:")
        predictions.select("prediction", "fraud_probability").show(10)
        
    except Exception as e:
        print(f"❌ Error in CSV inference: {str(e)}")


def example_clickhouse_inference():
    """Example: Run inference on ClickHouse data"""
    print("\n🗄️ Example 2: ClickHouse Database Inference")
    print("=" * 50)
    
    # Initialize inference engine
    engine = FraudInferenceEngine()
    
    # Set date range for inference (adjust as needed)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        # Run inference
        predictions = engine.run_inference(
            input_type='clickhouse',
            start_date=start_date,
            end_date=end_date,
            output_path=f'clickhouse_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            limit=50000,  # Limit for testing
            include_fraud_flag=True  # Include fraud flag for evaluation if available
        )
        
        print("✅ ClickHouse inference completed successfully!")
        print(f"Predictions shape: {predictions.count()} rows")
        
        # Show sample predictions
        print("\n📊 Sample predictions:")
        if 'fraud_flag' in predictions.columns:
            predictions.select("fraud_flag", "prediction", "fraud_probability").show(10)
        else:
            predictions.select("prediction", "fraud_probability").show(10)
            
        # Show prediction statistics
        print("\n📈 Prediction Statistics:")
        fraud_count = predictions.filter(predictions.prediction == 1.0).count()
        total_count = predictions.count()
        print(f"   • Total predictions: {total_count:,}")
        print(f"   • Fraud predictions: {fraud_count:,}")
        print(f"   • Fraud rate: {(fraud_count/total_count)*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error in ClickHouse inference: {str(e)}")


def example_interactive_inference():
    """Example: Interactive inference for notebook/custom use"""
    print("\n🔧 Example 3: Interactive Inference")
    print("=" * 50)
    
    # Initialize inference engine
    engine = FraudInferenceEngine()
    
    try:
        # Load model components
        engine.load_model_components()
        print("✅ Model components loaded successfully!")
        
        # Example: Load sample data from ClickHouse
        print("\n📊 Loading sample data for demonstration...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        raw_data = engine.load_data_from_clickhouse(
            start_date=start_date,
            end_date=end_date,
            limit=1000,
            include_fraud_flag=True
        )
        
        print(f"Sample data loaded: {raw_data.count()} rows")
        
        # Preprocess the data
        print("\n🔧 Preprocessing data...")
        preprocessed_data = engine.preprocess_data(raw_data)
        print("✅ Data preprocessed successfully!")
        
        # Make predictions
        print("\n🔮 Making predictions...")
        predictions = engine.make_predictions(preprocessed_data, raw_data)
        print("✅ Predictions completed!")
        
        # Show results
        print("\n📊 Prediction Results:")
        predictions.select("prediction", "fraud_probability").describe().show()
        
        return engine, predictions
        
    except Exception as e:
        print(f"❌ Error in interactive inference: {str(e)}")
        return None, None


def example_batch_scoring():
    """Example: Batch scoring multiple datasets"""
    print("\n🔄 Example 4: Batch Scoring")
    print("=" * 50)
    
    # Initialize inference engine
    engine = FraudInferenceEngine()
    
    # Define multiple date ranges for batch processing
    base_date = datetime.now()
    date_ranges = [
        (base_date - timedelta(days=7), base_date - timedelta(days=6)),
        (base_date - timedelta(days=6), base_date - timedelta(days=5)),
        (base_date - timedelta(days=5), base_date - timedelta(days=4)),
    ]
    
    all_results = []
    
    try:
        for i, (start_dt, end_dt) in enumerate(date_ranges, 1):
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            
            print(f"\n📅 Processing batch {i}: {start_date} to {end_date}")
            
            predictions = engine.run_inference(
                input_type='clickhouse',
                start_date=start_date,
                end_date=end_date,
                output_path=f'batch_{i}_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                limit=10000
            )
            
            # Collect statistics
            fraud_count = predictions.filter(predictions.prediction == 1.0).count()
            total_count = predictions.count()
            fraud_rate = (fraud_count/total_count)*100 if total_count > 0 else 0
            
            batch_stats = {
                'batch': i,
                'date_range': f"{start_date} to {end_date}",
                'total_predictions': total_count,
                'fraud_predictions': fraud_count,
                'fraud_rate': fraud_rate
            }
            all_results.append(batch_stats)
            
            print(f"   ✅ Batch {i} completed: {fraud_count}/{total_count} fraud ({fraud_rate:.2f}%)")
        
        # Summary report
        print("\n📋 Batch Processing Summary:")
        print("-" * 80)
        print(f"{'Batch':<8} {'Date Range':<25} {'Total':<10} {'Fraud':<8} {'Rate':<8}")
        print("-" * 80)
        for stats in all_results:
            print(f"{stats['batch']:<8} {stats['date_range']:<25} {stats['total_predictions']:<10} {stats['fraud_predictions']:<8} {stats['fraud_rate']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error in batch scoring: {str(e)}")


def main():
    """Run all examples"""
    print("🚀 Fraud Detection Inference Examples")
    print("=" * 60)
    print("This script demonstrates different ways to use the fraud detection inference engine.")
    print("\nExamples included:")
    print("1. CSV file inference")
    print("2. ClickHouse database inference") 
    print("3. Interactive inference")
    print("4. Batch scoring")
    print("\n" + "=" * 60)
    
    # Run examples (comment out any you don't want to run)
    
    # Example 1: CSV inference
    # example_csv_inference()
    
    # Example 2: ClickHouse inference
    example_clickhouse_inference()
    
    # Example 3: Interactive inference
    engine, predictions = example_interactive_inference()
    
    # Example 4: Batch scoring (uncomment to run)
    # example_batch_scoring()
    
    print("\n🎯 All examples completed!")
    print("\nTo run specific examples from command line:")
    print("# CSV inference:")
    print("python fraud_inference.py --input_type csv --input_path /path/to/data.csv --output_path predictions.csv")
    print("\n# ClickHouse inference:")
    print("python fraud_inference.py --input_type clickhouse --start_date 2025-07-01 --end_date 2025-07-31 --output_path predictions.csv")
    print("\n# Interactive mode:")
    print("python fraud_inference.py --interactive")


if __name__ == "__main__":
    main()