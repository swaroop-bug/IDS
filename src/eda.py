"""Exploratory Data Analysis (EDA) Module for CICIDS2017.

Computes class distributions, summary statistics, correlation matrices,
and temporal attack traffic breakdowns by day and hour for reporting and dashboards.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, functions as F

from config import (
    EDA_DIR,
    PARQUET_DATA_DIR,
    get_spark_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eda")

INPUT_PARQUET_PATH = PARQUET_DATA_DIR / "cleaned_flows.parquet"

def run_eda(force: bool = False) -> None:
    """Execute EDA computations and save precomputed artifacts."""
    class_dist_path = EDA_DIR / "class_distribution.csv"
    timeline_path = EDA_DIR / "attacks_by_day_hour.csv"
    summary_path = EDA_DIR / "summary_metrics.json"

    if class_dist_path.exists() and timeline_path.exists() and not force:
        logger.info("EDA outputs already exist in data/output/eda/. Skipping. Use --force to rerun.")
        return

    if not INPUT_PARQUET_PATH.exists():
        logger.error(f"Input Parquet dataset not found at {INPUT_PARQUET_PATH}. Run clean.py first.")
        raise FileNotFoundError(f"Missing {INPUT_PARQUET_PATH}")

    spark = get_spark_session("CICIDS2017-EDA")
    logger.info(f"Loading cleaned dataset from {INPUT_PARQUET_PATH}...")
    df = spark.read.parquet(str(INPUT_PARQUET_PATH))

    total_records = df.count()
    logger.info(f"Analyzing {total_records:,} cleaned flow records...")

    # 1. Class Distribution Analysis
    logger.info("Computing class distribution & imbalance metrics...")
    df = df.withColumn("is_attack", F.when(F.col("Label") == "BENIGN", F.lit(0)).otherwise(F.lit(1)))

    class_dist_df = (
        df.groupBy("Label", "is_attack")
        .agg(F.count("*").alias("count"))
        .withColumn("percentage", F.round((F.col("count") / total_records) * 100.0, 4))
        .orderBy(F.desc("count"))
    )

    class_dist_pd = class_dist_df.toPandas()
    class_dist_pd.to_csv(class_dist_path, index=False)
    logger.info(f"Saved class distribution to {class_dist_path}")

    # Summary Statistics
    total_attacks = int(class_dist_pd[class_dist_pd["is_attack"] == 1]["count"].sum())
    total_benign = int(class_dist_pd[class_dist_pd["is_attack"] == 0]["count"].sum())
    attack_ratio = float((total_attacks / total_records) * 100.0) if total_records > 0 else 0.0

    summary_metrics = {
        "total_records": total_records,
        "total_benign": total_benign,
        "total_attacks": total_attacks,
        "attack_percentage": round(attack_ratio, 2),
        "distinct_labels": int(class_dist_pd["Label"].nunique()),
        "distinct_attack_types": int(class_dist_pd[class_dist_pd["is_attack"] == 1]["Label"].nunique()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Saved summary metrics to {summary_path}")

    # 2. Attacks by Day and Hour (Timeline)
    logger.info("Computing attack traffic by day and hour...")
    timeline_df = (
        df.groupBy("day", "hour", "Label", "is_attack")
        .agg(F.count("*").alias("flow_count"))
        .orderBy("day", "hour", F.desc("flow_count"))
    )
    timeline_pd = timeline_df.toPandas()
    timeline_pd.to_csv(timeline_path, index=False)
    logger.info(f"Saved temporal breakdown to {timeline_path}")

    # 3. Basic Stats for Top Representative Features
    sample_features = [
        c for c in [
            "Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets",
            "Total_Length_of_Fwd_Packets", "Flow_Bytes_per_s", "Flow_Packets_per_s",
            "Packet_Length_Mean", "Average_Packet_Size"
        ] if c in df.columns
    ]
    if sample_features:
        logger.info(f"Computing descriptive statistics for key features: {sample_features}...")
        stats_df = df.select(sample_features).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
        stats_pd = stats_df.toPandas()
        stats_path = EDA_DIR / "basic_stats.csv"
        stats_pd.to_csv(stats_path, index=False)
        logger.info(f"Saved basic descriptive statistics to {stats_path}")

    # 4. Correlation Matrix
    logger.info("Calculating Pearson correlation matrix on numerical features...")
    ignore_cols = {"source_file", "day", "hour", "Label", "is_attack", "Timestamp", "parsed_ts"}
    num_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Take a representative stratified or random sample for fast matrix computation if dataset is huge
    corr_sample_size = min(total_records, 100_000)
    sample_pd = df.select(num_cols).sample(withReplacement=False, fraction=float(corr_sample_size / total_records), seed=42).toPandas()
    corr_matrix = sample_pd.corr(method="pearson")
    corr_path = EDA_DIR / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_path)
    logger.info(f"Saved correlation matrix to {corr_path}")

    logger.info("[SUCCESS] EDA stage completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on CICIDS2017.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing EDA outputs.")
    args = parser.parse_args()
    run_eda(force=args.force)
