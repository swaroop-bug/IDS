#!/usr/bin/env bash
# End-to-end pipeline execution script for CICIDS2017 Intrusion Detection Engine
set -e

# Default sample fraction (1.0 = 100%, 0.05 = 5% smoke test)
SAMPLE_FRAC=${1:-1.0}
FORCE=${2:-""}

FORCE_FLAG=""
if [ "$FORCE" = "--force" ]; then
    FORCE_FLAG="--force"
fi

echo "================================================================================"
echo "    CICIDS2017 NETWORK INTRUSION DETECTION PIPELINE (PySpark + Streamlit)       "
echo "================================================================================"
echo "Configuration: SAMPLE_FRAC=$SAMPLE_FRAC | FORCE=$FORCE_FLAG"
echo ""

# Ensure we are in project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export SAMPLE_FRAC="$SAMPLE_FRAC"

echo "[1/7] Ingesting CSVs -> Partitioned Parquet..."
python src/ingest.py $FORCE_FLAG --sample-frac "$SAMPLE_FRAC"

echo ""
echo "[2/7] Cleaning Data (Inf/Nulls/Duplicates/Zero-Variance/Web Attacks)..."
python src/clean.py $FORCE_FLAG

echo ""
echo "[3/7] Running Exploratory Data Analysis & Temporal Breakdowns..."
python src/eda.py $FORCE_FLAG

echo ""
echo "[4/7] Engineering Features (Multicollinearity, RF Top-N, Scaler, Imbalance)..."
python src/features.py $FORCE_FLAG

echo ""
echo "[5/7] Training & Evaluating MLlib Classifiers (DT, RF, LR)..."
python src/train.py $FORCE_FLAG

echo ""
echo "[6/7] Running Runtime Benchmarks (Spark vs. pandas + scikit-learn)..."
python src/benchmark.py $FORCE_FLAG

echo ""
echo "[7/7] Precomputing Dashboard Aggregates..."
python src/aggregate.py $FORCE_FLAG

echo ""
echo "================================================================================"
echo "Pipeline execution completed successfully!"
echo "Launch the interactive dashboard with:"
echo "    streamlit run app/dashboard.py"
echo "================================================================================"
