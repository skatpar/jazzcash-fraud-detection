@echo off
REM Quick start script with dummy data for Windows
REM This script generates dummy data and runs the complete fraud detection pipeline

echo ==============================================================================
echo   JazzCash Fraud Detection - Quick Start with Dummy Data
echo ==============================================================================
echo.

REM Set up environment
echo [1/4] Setting up environment...
if not exist "venv" (
    echo   Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

REM Install dependencies if needed
python -c "import pyspark" 2>nul
if errorlevel 1 (
    echo   Installing dependencies...
    pip install -q -r requirements.txt
)

echo   - Environment ready

REM Generate dummy data
echo.
echo [2/4] Generating dummy data (100K users, 400K transactions)...
python utils/generate_dummy_data.py --num-users 100000 --num-transactions 400000 --output-dir data/dummy

if errorlevel 1 (
    echo   x Failed to generate dummy data
    exit /b 1
)

REM Prepare dataset
echo.
echo [3/4] Preparing master dataset...
python utils/prepare_dummy_dataset.py --dummy-dir data/dummy --output-dir data/processed

if errorlevel 1 (
    echo   x Failed to prepare dataset
    exit /b 1
)

REM Run pipeline
echo.
echo [4/4] Running fraud detection pipeline...
python pipelines/main_pipeline.py --skip-data --model xgboost

if errorlevel 1 (
    echo   x Pipeline failed
    exit /b 1
)

echo.
echo ==============================================================================
echo   Pipeline Complete!
echo ==============================================================================
echo.
echo Results:
echo   - Model: data/models/xgboost_model.pkl
echo   - Metrics: data/models/xgboost_metrics.csv
echo   - Plots: plots/model/
echo.
echo View metrics:
echo   type data\models\xgboost_metrics.csv
echo.
echo Explore with Jupyter:
echo   jupyter lab notebooks/02_eda.ipynb
echo.
pause
