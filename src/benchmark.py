"""Runtime Benchmarking Module: PySpark vs. pandas + scikit-learn.

Compares end-to-end execution latency (Data Ingestion + Cleaning + Random Forest Training)
between distributed PySpark and single-node pandas/scikit-learn across multiple data scales.
Handles Out-Of-Memory (OOM) conditions gracefully.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.preprocessing import LabelEncoder

from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F

from config import (
    BENCHMARK_DIR,
    BENCHMARK_FRACTIONS,
    PARQUET_DATA_DIR,
    RANDOM_SEED,
    get_spark_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

CLEANED_PARQUET_PATH = PARQUET_DATA_DIR / "cleaned_flows.parquet"

def benchmark_spark(df_sample, feature_cols: List[str]) -> float:
    """Measure PySpark pipeline execution latency."""
    start_time = time.time()

    indexer = StringIndexer(inputCol="Label", outputCol="label_idx", handleInvalid="skip")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    rf = SparkRF(featuresCol="features", labelCol="label_idx", numTrees=20, maxDepth=8, seed=RANDOM_SEED)

    indexed = indexer.fit(df_sample).transform(df_sample)
    assembled = assembler.transform(indexed)
    model = rf.fit(assembled)
    # Force action
    _ = model.numClasses

    return time.time() - start_time

def benchmark_pandas_sklearn(pdf_sample, feature_cols: List[str]) -> Tuple[float, str]:
    """Measure pandas + scikit-learn execution latency with OOM protection."""
    gc.collect()
    start_time = time.time()
    try:
        # Preprocessing: drop inf / nan
        clean_df = pdf_sample[feature_cols + ["Label"]].replace([np.inf, -np.inf], np.nan).dropna()
        X = clean_df[feature_cols].values
        le = LabelEncoder()
        y = le.fit_transform(clean_df["Label"].values)

        clf = SklearnRF(n_estimators=20, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X, y)
        elapsed = time.time() - start_time
        return elapsed, "Success"
    except (MemoryError, Exception) as e:
        logger.warning(f"scikit-learn encountered failure/OOM: {e}")
        return -1.0, f"OOM/Error: {type(e).__name__}"

def run_benchmarks(
    fractions: List[float] = BENCHMARK_FRACTIONS,
    force: bool = False,
) -> None:
    """Run benchmark suite across specified data scale fractions."""
    output_csv = BENCHMARK_DIR / "runtime_benchmark.csv"
    output_json = BENCHMARK_DIR / "runtime_benchmark.json"

    if output_csv.exists() and not force:
        logger.info(f"Benchmark results already exist at {output_csv}. Skipping. Use --force to rerun.")
        return

    if not CLEANED_PARQUET_PATH.exists():
        logger.error(f"Cleaned dataset missing at {CLEANED_PARQUET_PATH}. Run clean.py first.")
        raise FileNotFoundError("Missing cleaned parquet")

    spark = get_spark_session("CICIDS2017-Benchmark")
    full_df = spark.read.parquet(str(CLEANED_PARQUET_PATH))
    total_records = full_df.count()

    ignore_cols = {"source_file", "day", "hour", "Label", "Timestamp", "parsed_ts", "is_attack"}
    feature_cols = [c for c in full_df.columns if c not in ignore_cols][:20]  # Standardized 20 features for benchmark

    results: List[Dict[str, Any]] = []

    logger.info(f"Starting runtime benchmarks across scales: {fractions}")
    for frac in fractions:
        approx_records = int(total_records * frac)
        logger.info(f"\n--- Benchmarking Scale Fraction: {frac*100:.1f}% (~{approx_records:,} records) ---")

        sample_df = full_df.sample(False, fraction=frac, seed=RANDOM_SEED).cache()
        actual_count = sample_df.count()

        # 1. Benchmark PySpark
        logger.info(f"Running PySpark benchmark ({actual_count:,} records)...")
        spark_time = benchmark_spark(sample_df, feature_cols)
        logger.info(f"  PySpark Runtime: {spark_time:.2f} seconds")

        # 2. Benchmark pandas + scikit-learn
        logger.info(f"Running pandas + scikit-learn benchmark ({actual_count:,} records)...")
        try:
            pdf = sample_df.select(feature_cols + ["Label"]).toPandas()
            sklearn_time, sklearn_status = benchmark_pandas_sklearn(pdf, feature_cols)
            del pdf
            gc.collect()
        except (MemoryError, Exception) as oom_err:
            sklearn_time = -1.0
            sklearn_status = f"OOM: {type(oom_err).__name__}"

        logger.info(f"  scikit-learn Runtime: {sklearn_time:.2f}s ({sklearn_status})")

        speedup = round(sklearn_time / spark_time, 2) if sklearn_time > 0 and spark_time > 0 else "N/A (OOM)"

        results.append({
            "fraction": frac,
            "record_count": actual_count,
            "pyspark_time_sec": round(spark_time, 2),
            "sklearn_time_sec": round(sklearn_time, 2) if sklearn_time > 0 else None,
            "sklearn_status": sklearn_status,
            "speedup_factor": speedup,
        })
        sample_df.unpersist()

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nBenchmark Summary Table:\n{res_df.to_string(index=False)}")
    logger.info(f"Saved benchmark results to {output_csv} and {output_json}")
    logger.info("[SUCCESS] Benchmark stage completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Spark vs. scikit-learn.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing benchmark results.")
    args = parser.parse_args()
    run_benchmarks(force=args.force)
