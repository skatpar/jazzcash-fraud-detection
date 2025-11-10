#!/bin/bash
# Quick Start Script for Decision Tree Training
# This script provides a simple way to run the decision tree training

echo "=========================================="
echo "Decision Tree Fraud Detection Training"
echo "=========================================="
echo ""

# Set the script path
SCRIPT_DIR="/root/research-dir/dev/jazzcash-fraud-detection/scripts"
SCRIPT_NAME="train_decision_tree.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ ERROR: Script not found at $SCRIPT_PATH"
    exit 1
fi

echo "📂 Script location: $SCRIPT_PATH"
echo "🕒 Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check Python environment
echo "🐍 Python environment:"
which python
python --version
echo ""

# Run the training script
echo "🚀 Starting Decision Tree training..."
echo "📝 Logs will be saved to: decision_tree_training_*.log"
echo ""
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
python "$SCRIPT_NAME"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📦 Artifacts saved in:"
    echo "   - decision_tree_model_*/"
    echo "   - dt_preprocessing_*/"
    echo "   - dt_rules_*/"
    echo "   - dt_predictions_*/"
    echo "   - training_summary_*.txt"
    echo "   - decision_tree_training_*.log"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the log file for details:"
    echo "   - decision_tree_training_*.log"
fi
echo "=========================================="
echo "🕒 Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

exit $EXIT_CODE
