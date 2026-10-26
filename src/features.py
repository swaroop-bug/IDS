"""Feature Engineering and Selection Module for CICIDS2017.

Performs:
1. Pruning of highly correlated features (|r| > 0.95).
2. Supervised feature selection using Random Forest feature importance (top-N).
3. Spark ML Pipeline assembly (StringIndexer, VectorAssembler, StandardScaler).
4. Class imbalance mitigation (class weights, downsampling of BENIGN without test leakage).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, functions as F

from config import (
    BALANCE_STRATEGY,
    BENIGN_DOWNSAMPLE_RATIO,
    CORRELATION_THRESHOLD,
    EDA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PARQUET_DATA_DIR,
    RANDOM_SEED,
    TOP_N_FEATURES,
    TRAIN_TEST_SPLIT,
    get_spark_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("features")

INPUT_PARQUET_PATH = PARQUET_DATA_DIR / "cleaned_flows.parquet"
TRAIN_PARQUET_PATH = PARQUET_DATA_DIR / "train_features.parquet"
TEST_PARQUET_PATH = PARQUET_DATA_DIR / "test_features.parquet"
SELECTED_FEATURES_JSON = EDA_DIR / "selected_features.json"

METADATA_COLS = {"source_file", "day", "hour", "Label", "Timestamp", "parsed_ts", "is_attack"}

def remove_collinear_features(df: DataFrame, threshold: float = CORRELATION_THRESHOLD) -> List[str]:
    """Identify and drop redundant features with Pearson correlation exceeding threshold."""
    candidate_cols = [c for c in df.columns if c not in METADATA_COLS]
    corr_path = EDA_DIR / "correlation_matrix.csv"

    if corr_path.exists():
        logger.info(f"Loading cached correlation matrix from {corr_path}...")
        corr_matrix = pd.read_csv(corr_path, index_col=0)
    else:
        logger.info("Computing Pearson correlation matrix for collinearity reduction...")
        sample_size = min(df.count(), 50_000)
        pdf = df.select(candidate_cols).sample(False, float(sample_size / df.count()), seed=RANDOM_SEED).toPandas()
        corr_matrix = pdf.corr().abs()

    dropped = set()
    cols = [c for c in corr_matrix.columns if c in candidate_cols]

    for i in range(len(cols)):
        col_i = cols[i]
        if col_i in dropped:
            continue
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            if col_j in dropped:
                continue
            val = abs(corr_matrix.loc[col_i, col_j])
            if val > threshold:
                dropped.add(col_j)

    retained = [c for c in candidate_cols if c not in dropped]
    logger.info(f"Collinearity filtering: Evaluated {len(candidate_cols)} features. Dropped {len(dropped)} (|r| > {threshold}). Retained {len(retained)}.")
    return retained

def select_top_features_rf(
    df: DataFrame,
    feature_cols: List[str],
    top_n: int = TOP_N_FEATURES,
) -> Tuple[List[str], Dict[str, float]]:
    """Rank and select top-N features based on Random Forest feature importances."""
    logger.info(f"Training Random Forest on {len(feature_cols)} candidates to extract feature importances...")
    
    # Fast RF on sample for ranking
    sample_df = df.sample(False, fraction=min(1.0, 100_000 / max(1, df.count())), seed=RANDOM_SEED)

    indexer = StringIndexer(inputCol="Label", outputCol="label_idx", handleInvalid="skip")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_vec", handleInvalid="skip")
    rf = RandomForestClassifier(featuresCol="raw_vec", labelCol="label_idx", numTrees=25, maxDepth=7, seed=RANDOM_SEED)

    pipeline = Pipeline(stages=[indexer, assembler, rf])
    model = pipeline.fit(sample_df)
    rf_model = model.stages[-1]

    importances = rf_model.featureImportances.toArray()
    feat_imp = {col_name: float(imp) for col_name, imp in zip(feature_cols, importances)}
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)

    top_features = [f[0] for f in sorted_feats[:top_n]]
    logger.info(f"Top {len(top_features)} selected features:")
    for rank, (feat, score) in enumerate(sorted_feats[:10], start=1):
        logger.info(f"  {rank:2d}. {feat:<35} : {score:.5f}")

    # Persist ranking
    with open(SELECTED_FEATURES_JSON, "w") as f:
        json.dump({"top_features": top_features, "all_importances": dict(sorted_feats)}, f, indent=2)

    pd.DataFrame(sorted_feats, columns=["feature", "importance"]).to_csv(EDA_DIR / "feature_importance.csv", index=False)
    return top_features, dict(sorted_feats)

def compute_class_weights(train_df: DataFrame) -> Dict[str, float]:
    """Calculate inverse-frequency class weights strictly on the training partition."""
    counts = train_df.groupBy("Label").count().collect()
    total = sum(row["count"] for row in counts)
    num_classes = len(counts)
    weights = {}
    for row in counts:
        cls_count = row["count"]
        # Balanced heuristic: N / (k * N_c)
        w = float(total / (num_classes * cls_count))
        weights[row["Label"]] = w
    return weights

def build_features_pipeline(force: bool = False) -> None:
    """Run full feature engineering, split train/test, apply scaling, and persist."""
    if TRAIN_PARQUET_PATH.exists() and TEST_PARQUET_PATH.exists() and not force:
        logger.info(f"Train/Test feature sets already exist. Skipping. Use --force to rerun.")
        return

    if not INPUT_PARQUET_PATH.exists():
        logger.error(f"Input Parquet dataset not found at {INPUT_PARQUET_PATH}. Run clean.py first.")
        raise FileNotFoundError(f"Missing {INPUT_PARQUET_PATH}")

    spark = get_spark_session("CICIDS2017-Features")
    df = spark.read.parquet(str(INPUT_PARQUET_PATH))

    # 1. Stratified Train / Test Split (80/20)
    logger.info("Executing 80/20 Stratified Train/Test split...")
    fractions = {row["Label"]: TRAIN_TEST_SPLIT for row in df.select("Label").distinct().collect()}
    train_df = df.sampleBy("Label", fractions=fractions, seed=RANDOM_SEED).cache()
    test_df = df.subtract(train_df).cache()

    train_count = train_df.count()
    test_count = test_df.count()
    logger.info(f"Partitioned dataset: Train={train_count:,} ({train_count/(train_count+test_count)*100:.1f}%), Test={test_count:,}")

    # 2. Multicollinearity Filtering (on train partition)
    non_collinear = remove_collinear_features(train_df, threshold=CORRELATION_THRESHOLD)

    # 3. Supervised Feature Selection (top N)
    selected_features, _ = select_top_features_rf(train_df, non_collinear, top_n=TOP_N_FEATURES)

    # 4. Handle Class Imbalance
    if "downsample_benign" in BALANCE_STRATEGY:
        logger.info(f"Applying downsampling of BENIGN class on train set (Ratio: {BENIGN_DOWNSAMPLE_RATIO})...")
        benign_train = train_df.filter(F.col("Label") == "BENIGN").sample(False, fraction=BENIGN_DOWNSAMPLE_RATIO, seed=RANDOM_SEED)
        attack_train = train_df.filter(F.col("Label") != "BENIGN")
        train_df = benign_train.unionByName(attack_train)
        logger.info(f"Train count after BENIGN downsampling: {train_df.count():,}")

    class_weights = compute_class_weights(train_df)
    logger.info(f"Computed class weights on train partition: {class_weights}")

    # Map weight column
    mapping_expr = F.create_map([F.lit(x) for x in sum([[k, v] for k, v in class_weights.items()], [])])
    train_df = train_df.withColumn("weight", mapping_expr.getItem(F.col("Label")))
    test_df = test_df.withColumn("weight", F.lit(1.0))  # Standard weights on evaluation test

    # 5. Build Spark ML Pipeline
    logger.info("Assembling Spark ML pipeline stages (StringIndexer, VectorAssembler, StandardScaler)...")
    indexer = StringIndexer(inputCol="Label", outputCol="indexed_label", handleInvalid="skip")
    assembler = VectorAssembler(inputCols=selected_features, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

    feature_pipeline = Pipeline(stages=[indexer, assembler, scaler])
    pipeline_model = feature_pipeline.fit(train_df)

    # Transform both splits
    train_featured = pipeline_model.transform(train_df)
    test_featured = pipeline_model.transform(test_df)

    # Save Pipeline Model
    pipeline_path = MODELS_DIR / "feature_pipeline_model"
    pipeline_model.write().overwrite().save(str(pipeline_path))
    logger.info(f"Saved feature pipeline model to {pipeline_path}")

    # Save transformed datasets
    keep_cols = ["features", "indexed_label", "Label", "weight", "day", "hour", "source_file"]
    logger.info(f"Writing transformed train features to {TRAIN_PARQUET_PATH}...")
    train_featured.select(keep_cols).write.mode("overwrite").parquet(str(TRAIN_PARQUET_PATH))
    
    logger.info(f"Writing transformed test features to {TEST_PARQUET_PATH}...")
    test_featured.select(keep_cols).write.mode("overwrite").parquet(str(TEST_PARQUET_PATH))

    train_df.unpersist()
    test_df.unpersist()
    logger.info("[SUCCESS] Feature engineering stage completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering and pipeline preparation.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing feature datasets.")
    args = parser.parse_args()
    build_features_pipeline(force=args.force)
