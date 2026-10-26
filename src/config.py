"""Configuration module for CICIDS2017 Network Intrusion Detection Pipeline.

Defines all paths, Spark session configurations, feature selection thresholds,
class balance strategies, and runtime parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARQUET_DATA_DIR = DATA_DIR / "parquet"
OUTPUT_DIR = DATA_DIR / "output"

EDA_DIR = OUTPUT_DIR / "eda"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
AGGREGATES_DIR = OUTPUT_DIR / "aggregates"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"

# Create directories if they do not exist
for d in [RAW_DATA_DIR, PARQUET_DATA_DIR, OUTPUT_DIR, EDA_DIR, MODELS_DIR, METRICS_DIR, AGGREGATES_DIR, BENCHMARK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Sampling & Reproductibility
SAMPLE_FRAC: float = float(os.getenv("SAMPLE_FRAC", "1.0"))
RANDOM_SEED: int = 42
TRAIN_TEST_SPLIT: float = 0.8  # 80/20 train/test split

# Spark Configuration for 8 GB RAM laptop local mode
SPARK_APP_NAME: str = "CICIDS2017-NIDS-Pipeline"
SPARK_DRIVER_MEMORY: str = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_DRIVER_MAX_RESULT: str = "2g"
SPARK_SHUFFLE_PARTITIONS: str = os.getenv("SPARK_SHUFFLE_PARTITIONS", "8")
SPARK_DEFAULT_PARALLELISM: str = os.getenv("SPARK_DEFAULT_PARALLELISM", "8")

# Cleaning & Preprocessing Settings
MERGE_WEB_ATTACKS: bool = True
# If True, looks for existing 'Timestamp' column; if False, simulates day/hour from file name & row order
USE_REAL_TIMESTAMPS: bool = os.getenv("USE_REAL_TIMESTAMPS", "false").lower() in ("true", "1", "yes")

# Feature Selection Settings
CORRELATION_THRESHOLD: float = 0.95
TOP_N_FEATURES: int = 30

# Class Imbalance Handling Settings: "class_weights", "downsample_benign", "both", or "none"
BALANCE_STRATEGY: str = os.getenv("BALANCE_STRATEGY", "class_weights")
BENIGN_DOWNSAMPLE_RATIO: float = float(os.getenv("BENIGN_DOWNSAMPLE_RATIO", "0.20"))

# Benchmark Settings
BENCHMARK_FRACTIONS = [0.05, 0.15, 0.30]  # Safe scales for laptop comparison

# Expected raw CSV filenames and their associated weekday
DAY_MAPPING: Dict[str, str] = {
    "monday": "Monday",
    "tuesday": "Tuesday",
    "wednesday": "Wednesday",
    "thursday": "Thursday",
    "friday": "Friday",
}

def configure_environment() -> None:
    """Ensure JAVA_HOME and HADOOP_HOME are configured for PySpark in cross-platform setups."""
    # Java detection
    if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
        candidate_paths = [
            Path(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"),
            Path(r"C:\Program Files\Eclipse Adoptium"),
            Path(r"C:\Program Files\Java"),
            Path(r"C:\Users\swaro\.jdks\openjdk-27"),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                if (candidate / "bin" / "java.exe").exists():
                    os.environ["JAVA_HOME"] = str(candidate)
                    break
                for sub in candidate.iterdir():
                    if sub.is_dir() and (sub / "bin" / "java.exe").exists():
                        os.environ["JAVA_HOME"] = str(sub)
                        break
            if "JAVA_HOME" in os.environ:
                break

    # Hadoop / winutils detection for Windows
    local_hadoop = PROJECT_ROOT / "hadoop"
    if local_hadoop.exists() and (local_hadoop / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(local_hadoop)
        hadoop_bin = str(local_hadoop / "bin")
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

def get_spark_session(app_name: str = SPARK_APP_NAME, driver_memory: str = SPARK_DRIVER_MEMORY) -> SparkSession:
    """Creates and tunes a local PySpark session tailored for 8GB RAM laptops."""
    configure_environment()
    
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", SPARK_DRIVER_MAX_RESULT)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.network.timeout", "600s")
    )
    return builder.getOrCreate()
