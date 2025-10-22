#!/bin/bash

# Quick start script with dummy data
# This script generates dummy data and runs the complete fraud detection pipeline

echo "=============================================================================="
echo "  JazzCash Fraud Detection - Quick Start with Dummy Data"
echo "=============================================================================="
echo ""

# Set up environment
echo "[1/4] Setting up environment..."
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies if needed
if ! python -c "import pyspark" 2>/dev/null; then
    echo "  Installing dependencies..."
    pip install -q -r requirements.txt
fi

echo "  ✓ Environment ready"

# Generate dummy data
echo ""
echo "[2/4] Generating dummy data (100K users, 400K transactions)..."
python utils/generate_dummy_data.py \
    --num-users 100000 \
    --num-transactions 400000 \
    --output-dir data/dummy

if [ $? -ne 0 ]; then
    echo "  ✗ Failed to generate dummy data"
    exit 1
fi

# Prepare dataset
echo ""
echo "[3/4] Preparing master dataset..."
python utils/prepare_dummy_dataset.py \
    --dummy-dir data/dummy \
    --output-dir data/processed

if [ $? -ne 0 ]; then
    echo "  ✗ Failed to prepare dataset"
    exit 1
fi

# Run pipeline (skip data extraction since we have the data)
echo ""
echo "[4/4] Running fraud detection pipeline..."
python pipelines/main_pipeline.py \
    --skip-data \
    --model xgboost

if [ $? -ne 0 ]; then
    echo "  ✗ Pipeline failed"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "  Pipeline Complete!"
echo "=============================================================================="
echo ""
echo "Results:"
echo "  - Model: data/models/xgboost_model.pkl"
echo "  - Metrics: data/models/xgboost_metrics.csv"
echo "  - Plots: plots/model/"
echo ""
echo "View metrics:"
echo "  cat data/models/xgboost_metrics.csv"
echo ""
echo "Explore with Jupyter:"
echo "  jupyter lab notebooks/02_eda.ipynb"
echo ""
