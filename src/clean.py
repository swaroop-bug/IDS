"""Data Cleaning Module for CICIDS2017.

Handles infinite values, drops nulls and duplicates, removes zero-variance columns,
merges Web Attack variants based on configuration, and logs label distributions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType

from config import (
    MERGE_WEB_ATTACKS,
    PARQUET_DATA_DIR,
    get_spark_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean")

INPUT_PARQUET_PATH = PARQUET_DATA_DIR / "raw_flows.parquet"
OUTPUT_PARQUET_PATH = PARQUET_DATA_DIR / "cleaned_flows.parquet"

METADATA_COLS = {"source_file", "day", "hour", "Label", "Timestamp", "parsed_ts"}

def get_numeric_columns(df: DataFrame) -> List[str]:
    """Retrieve list of numerical feature columns."""
    num_cols = []
    for field in df.schema.fields:
        if field.name not in METADATA_COLS and isinstance(
            field.dataType, (DoubleType, FloatType, IntegerType, LongType)
        ):
            num_cols.append(field.name)
    return num_cols

def print_label_distribution(df: DataFrame, step_name: str) -> None:
    """Log current distribution of traffic labels."""
    logger.info(f"--- Label Distribution: {step_name} ---")
    dist = df.groupBy("Label").count().orderBy(F.desc("count")).collect()
    for row in dist:
        logger.info(f"  {row['Label']:<35} : {row['count']:>10,}")

def clean_data(force: bool = False, merge_web: bool = MERGE_WEB_ATTACKS) -> None:
    """Run data cleaning steps and persist cleaned Parquet."""
    if OUTPUT_PARQUET_PATH.exists() and not force:
        logger.info(f"Cleaned dataset already exists at {OUTPUT_PARQUET_PATH}. Skipping cleaning. Use --force to rerun.")
        return

    if not INPUT_PARQUET_PATH.exists():
        logger.error(f"Input Parquet dataset not found at {INPUT_PARQUET_PATH}. Run ingest.py first.")
        raise FileNotFoundError(f"Missing {INPUT_PARQUET_PATH}")

    spark = get_spark_session("CICIDS2017-Cleaning")
    logger.info(f"Loading raw Parquet from {INPUT_PARQUET_PATH}...")
    df = spark.read.parquet(str(INPUT_PARQUET_PATH))

    initial_count = df.count()
    logger.info(f"Initial row count: {initial_count:,}")
    print_label_distribution(df, "BEFORE Cleaning")

    num_cols = get_numeric_columns(df)
    logger.info(f"Processing {len(num_cols)} numerical features for Infinity/NaN filtering...")

    # Replace Infinity and -Infinity with null
    for col_name in num_cols:
        df = df.withColumn(
            col_name,
            F.when(
                F.col(col_name).isin(float("inf"), float("-inf")) | F.isnan(F.col(col_name)),
                None,
            ).otherwise(F.col(col_name)),
        )

    # Drop rows with nulls in any numeric features
    df = df.dropna(subset=num_cols)
    after_nulls_count = df.count()
    logger.info(f"Rows after dropping nulls & infinities: {after_nulls_count:,} (dropped {initial_count - after_nulls_count:,})")

    # Drop duplicate records
    logger.info("Dropping duplicate rows across network flows...")
    df = df.dropDuplicates()
    after_dedup_count = df.count()
    logger.info(f"Rows after deduplication: {after_dedup_count:,} (dropped {after_nulls_count - after_dedup_count:,})")

    # Identify and drop constant / zero-variance columns
    logger.info("Checking for zero-variance / constant columns...")
    stats_exprs = []
    for col_name in num_cols:
        stats_exprs.append(F.min(col_name).alias(f"{col_name}_min"))
        stats_exprs.append(F.max(col_name).alias(f"{col_name}_max"))

    stats_row = df.select(stats_exprs).first()
    zero_variance_cols = []
    if stats_row:
        for col_name in num_cols:
            c_min = stats_row[f"{col_name}_min"]
            c_max = stats_row[f"{col_name}_max"]
            if c_min is not None and c_max is not None and c_min == c_max:
                zero_variance_cols.append(col_name)

    if zero_variance_cols:
        logger.info(f"Dropping {len(zero_variance_cols)} zero-variance columns: {zero_variance_cols}")
        df = df.drop(*zero_variance_cols)
    else:
        logger.info("No constant columns detected.")

    # Merge Web Attack variants if requested
    if merge_web:
        logger.info("Merging Web Attack variants ('Web Attack - Brute Force', 'Web Attack - XSS', etc.) into 'Web Attack'...")
        df = df.withColumn(
            "Label",
            F.when(F.col("Label").rlike("(?i)web attack"), F.lit("Web Attack")).otherwise(F.col("Label")),
        )

    print_label_distribution(df, "AFTER Cleaning & Merging")

    final_count = df.count()
    logger.info(f"Final cleaned row count: {final_count:,} (Retained {(final_count / initial_count) * 100:.2f}%)")

    logger.info(f"Writing cleaned Parquet partitioned by 'day' to {OUTPUT_PARQUET_PATH}...")
    df.write.partitionBy("day").mode("overwrite").parquet(str(OUTPUT_PARQUET_PATH))
    logger.info("[SUCCESS] Cleaning completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CICIDS2017 dataset.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cleaned parquet dataset.")
    parser.add_argument("--merge-web", action="store_true", default=MERGE_WEB_ATTACKS, help="Merge Web Attack variants.")
    args = parser.parse_args()
    clean_data(force=args.force, merge_web=args.merge_web)
