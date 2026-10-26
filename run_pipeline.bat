@echo off
REM End-to-end pipeline execution script for Windows
setlocal enabledelayedexpansion

set SAMPLE_FRAC=1.0
if not "%~1"=="" set SAMPLE_FRAC=%~1

set FORCE_FLAG=
if "%~2"=="--force" set FORCE_FLAG=--force

echo ================================================================================
echo     CICIDS2017 NETWORK INTRUSION DETECTION PIPELINE (PySpark + Streamlit)
echo ================================================================================
echo Configuration: SAMPLE_FRAC=%SAMPLE_FRAC% %FORCE_FLAG%
echo.

cd /d "%~dp0"
set PYTHONPATH=%~dp0src;%PYTHONPATH%
set SAMPLE_FRAC=%SAMPLE_FRAC%

echo [1/7] Ingesting CSVs -^> Partitioned Parquet...
python src\ingest.py %FORCE_FLAG% --sample-frac %SAMPLE_FRAC%
if errorlevel 1 exit /b %errorlevel%

echo.
echo [2/7] Cleaning Data (Inf/Nulls/Duplicates/Zero-Variance/Web Attacks)...
python src\clean.py %FORCE_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo [3/7] Running Exploratory Data Analysis ^& Temporal Breakdowns...
python src\eda.py %FORCE_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo [4/7] Engineering Features (Multicollinearity, RF Top-N, Scaler, Imbalance)...
python src\features.py %FORCE_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo [5/7] Training ^& Evaluating MLlib Classifiers (DT, RF, LR)...
python src\train.py %FORCE_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo [6/7] Running Runtime Benchmarks (Spark vs. pandas + scikit-learn)...
python src\benchmark.py %FORCE_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo [7/7] Precomputing Dashboard Aggregates...
python src\aggregate.py %FORCE_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo ================================================================================
echo Pipeline execution completed successfully!
echo Launch the interactive dashboard with:
echo     streamlit run app\dashboard.py
echo ================================================================================
