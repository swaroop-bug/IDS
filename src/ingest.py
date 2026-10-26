"""Data Ingestion Module for CICIDS2017.

Loads raw CSV network flow files using explicit schemas, strips column whitespace,
annotates with source file and day, derives simulated or real timestamps,
and writes partitioned Parquet files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from pyspark.sql import DataFrame, functions as F, Window
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from config import (
    DAY_MAPPING,
    PARQUET_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_FRAC,
    USE_REAL_TIMESTAMPS,
    get_spark_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

OUTPUT_PARQUET_PATH = PARQUET_DATA_DIR / "raw_flows.parquet"

# Standard columns across CICIDS2017 MachineLearningCVE dataset
CANONICAL_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min"
]

def check_raw_files() -> List[Path]:
    """Verify raw CSV files exist, providing download instructions if missing."""
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        error_msg = (
            "\n" + "=" * 80 + "\n"
            "[CRITICAL ERROR] No CSV files found in data/raw/!\n"
            "The CICIDS2017 dataset requires manual download before running this pipeline.\n"
            "Please download the 'MachineLearningCVE' (or 'GeneratedLabelledFlows') dataset from:\n"
            "  1. Official UNB CIC: https://www.unb.ca/cic/datasets/ids-2017.html\n"
            "  2. Kaggle Mirror:    https://www.kaggle.com/datasets/cicdataset/cicids2017\n\n"
            "Expected 8 CSV files inside data/raw/:\n"
            "  - Monday-WorkingHours.pcap_ISCX.csv\n"
            "  - Tuesday-WorkingHours.pcap_ISCX.csv\n"
            "  - Wednesday-workingHours.pcap_ISCX.csv\n"
            "  - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv\n"
            "  - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv\n"
            "  - Friday-WorkingHours-Morning.pcap_ISCX.csv\n"
            "  - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv\n"
            "  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv\n"
            + "=" * 80 + "\n"
        )
        logger.error(error_msg)
        sys.exit(1)
    return sorted(csv_files)

def get_day_from_filename(filename: str) -> str:
    """Derive weekday name from filename."""
    name_lower = filename.lower()
    for key, day in DAY_MAPPING.items():
        if key in name_lower:
            return day
    return "Unknown"

def clean_column_name(col_name: str) -> str:
    """Normalize column names: strip whitespace and replace non-standard characters."""
    c = col_name.strip()
    c = c.replace("/", "_per_").replace(" ", "_").replace(".", "_")
    return c

def ingest_all(force: bool = False, sample_frac: float = SAMPLE_FRAC) -> None:
    """Ingest raw CSVs, apply explicit schema, add metadata, and save Parquet."""
    if OUTPUT_PARQUET_PATH.exists() and not force:
        logger.info(f"Parquet dataset already exists at {OUTPUT_PARQUET_PATH}. Skipping ingestion. Use --force to rerun.")
        return

    csv_files = check_raw_files()
    spark = get_spark_session("CICIDS2017-Ingestion")
    logger.info(f"Found {len(csv_files)} CSV files. Initializing ingestion with sample_frac={sample_frac}...")

    dfs: List[DataFrame] = []

    for file_path in csv_files:
        day_str = get_day_from_filename(file_path.name)
        logger.info(f"Reading {file_path.name} (Assigned Day: {day_str})...")

        # Read header only first to map present columns
        header_df = spark.read.option("header", "true").option("nrows", 5).csv(str(file_path))
        raw_cols = header_df.columns
        cleaned_col_map = {orig: clean_column_name(orig) for orig in raw_cols}

        # Build schema dynamically based on file columns
        fields = []
        for orig in raw_cols:
            c = orig.strip()
            if c.lower() in ("label", "class"):
                fields.append(StructField(orig, StringType(), True))
            elif c.lower() == "timestamp":
                fields.append(StructField(orig, StringType(), True))
            else:
                fields.append(StructField(orig, DoubleType(), True))
        explicit_schema = StructType(fields)

        df = spark.read.option("header", "true").schema(explicit_schema).csv(str(file_path))

        # Rename columns to cleaned versions
        for orig, clean in cleaned_col_map.items():
            df = df.withColumnRenamed(orig, clean)

        # Standardize Label column name
        if "Label" not in df.columns:
            for c in df.columns:
                if c.lower() in ("label", "class"):
                    df = df.withColumnRenamed(c, "Label")
                    break

        # Trim whitespace from Label strings
        if "Label" in df.columns:
            df = df.withColumn("Label", F.trim(F.col("Label")))

        # Add tracking metadata
        df = df.withColumn("source_file", F.lit(file_path.name))
        df = df.withColumn("day", F.lit(day_str))

        # Handle Timestamps: Real vs Simulated
        if USE_REAL_TIMESTAMPS and "Timestamp" in df.columns:
            # Parse real timestamp format (e.g. "dd/MM/yyyy HH:mm:ss" or standard)
            df = df.withColumn("parsed_ts", F.to_timestamp(F.col("Timestamp"), "d/M/yyyy H:m"))
            df = df.withColumn("hour", F.hour(F.coalesce(F.col("parsed_ts"), F.current_timestamp())))
        else:
            # Simulate hour: working hours between 09:00 and 17:00 (hours 9 through 16)
            # Use monotonically_increasing_id within the file to distribute evenly
            w = Window.orderBy(F.monotonically_increasing_id())
            df = df.withColumn("row_idx", F.row_number().over(w))
            # Distribute into 8 working hours (9 to 16)
            df = df.withColumn("hour", F.lit(9) + (F.col("row_idx") % 8)).drop("row_idx")

        # Downsample if requested
        if sample_frac < 1.0:
            df = df.sample(withReplacement=False, fraction=sample_frac, seed=42)

        dfs.append(df)

    # Union all dataframes
    logger.info("Unioning all files across days...")
    full_df = dfs[0]
    for next_df in dfs[1:]:
        full_df = full_df.unionByName(next_df, allowMissingColumns=True)

    total_records = full_df.count()
    logger.info(f"Total ingested records: {total_records:,}")

    # Write out Parquet partitioned by day
    logger.info(f"Writing Parquet partitioned by 'day' to {OUTPUT_PARQUET_PATH}...")
    full_df.write.partitionBy("day").mode("overwrite").parquet(str(OUTPUT_PARQUET_PATH))
    logger.info("[SUCCESS] Ingestion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CICIDS2017 raw CSVs into Parquet.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet dataset.")
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC, help="Fraction of data to sample.")
    args = parser.parse_args()
    ingest_all(force=args.force, sample_frac=args.sample_frac)
