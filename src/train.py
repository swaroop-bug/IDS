"""Model Training and Evaluation Module for CICIDS2017.

Trains and evaluates:
1. Decision Tree (Baseline)
2. Random Forest
3. Multinomial Logistic Regression
4. Gradient Boosted Trees (One-vs-Rest, configurable)

Computes accuracy, macro/weighted precision, recall, and F1, per-class metrics,
confusion matrices, and selects the best model by macro-F1.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    OneVsRest,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import DataFrame, functions as F

from config import (
    METRICS_DIR,
    MODELS_DIR,
    PARQUET_DATA_DIR,
    RANDOM_SEED,
    get_spark_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

TRAIN_PARQUET_PATH = PARQUET_DATA_DIR / "train_features.parquet"
TEST_PARQUET_PATH = PARQUET_DATA_DIR / "test_features.parquet"

def get_label_index_mapping(test_df: DataFrame) -> Dict[int, str]:
    """Extract mapping from indexed numerical label to original string Label."""
    mapping = {}
    label_pairs = test_df.select("indexed_label", "Label").distinct().collect()
    for row in label_pairs:
        mapping[int(row["indexed_label"])] = str(row["Label"])
    return mapping

def evaluate_predictions(
    predictions: DataFrame,
    label_map: Dict[int, str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[List[int]], List[str]]:
    """Compute detailed evaluation metrics: Macro/Weighted P/R/F1, Per-class, and Confusion Matrix."""
    # Convert predictions to RDD of (prediction, label)
    pred_label_rdd = predictions.select(
        F.col("prediction").cast("double"),
        F.col("indexed_label").cast("double")
    ).rdd.map(lambda r: (float(r[0]), float(r[1])))

    metrics = MulticlassMetrics(pred_label_rdd)

    labels = sorted(list(label_map.keys()))
    label_names = [label_map.get(i, f"Class_{i}") for i in labels]

    # Global Metrics
    accuracy = float(metrics.accuracy)
    weighted_p = float(metrics.weightedPrecision)
    weighted_r = float(metrics.weightedRecall)
    weighted_f1 = float(metrics.weightedFMeasure())

    # Per-class metrics
    per_class = {}
    precisions, recalls, f1s = [], [], []

    for idx, name in zip(labels, label_names):
        p = float(metrics.precision(idx))
        r = float(metrics.recall(idx))
        f1 = float(metrics.fMeasure(idx))
        per_class[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
        }
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    global_metrics = {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
    }

    # Confusion Matrix as 2D array
    cm_matrix = metrics.confusionMatrix().toArray()
    cm_list = [[int(val) for val in row] for row in cm_matrix]

    return global_metrics, per_class, cm_list, label_names

def train_and_evaluate_all(force: bool = False, include_gbt: bool = False) -> None:
    """Train all classifiers, record runtime, evaluate metrics, and save best model."""
    summary_path = METRICS_DIR / "model_comparison.json"
    if summary_path.exists() and not force:
        logger.info(f"Model metrics already exist at {summary_path}. Skipping. Use --force to rerun.")
        return

    if not TRAIN_PARQUET_PATH.exists() or not TEST_PARQUET_PATH.exists():
        logger.error("Train/Test feature sets missing. Run features.py first.")
        raise FileNotFoundError("Missing feature datasets")

    spark = get_spark_session("CICIDS2017-Training")
    logger.info("Loading preprocessed training and test partitions...")
    train_df = spark.read.parquet(str(TRAIN_PARQUET_PATH)).cache()
    test_df = spark.read.parquet(str(TEST_PARQUET_PATH)).cache()

    logger.info(f"Train records: {train_df.count():,}, Test records: {test_df.count():,}")
    label_map = get_label_index_mapping(test_df)
    logger.info(f"Detected {len(label_map)} classes: {label_map}")

    models_config = {
        "Decision_Tree": DecisionTreeClassifier(
            featuresCol="features",
            labelCol="indexed_label",
            maxDepth=10,
            seed=RANDOM_SEED,
        ),
        "Random_Forest": RandomForestClassifier(
            featuresCol="features",
            labelCol="indexed_label",
            weightCol="weight",
            numTrees=35,
            maxDepth=10,
            seed=RANDOM_SEED,
        ),
        "Logistic_Regression": LogisticRegression(
            featuresCol="features",
            labelCol="indexed_label",
            weightCol="weight",
            maxIter=30,
            regParam=0.01,
            elasticNetParam=0.1,
        ),
    }

    if include_gbt:
        logger.info("Including GBTClassifier via OneVsRest wrapper...")
        gbt_base = GBTClassifier(
            featuresCol="features",
            labelCol="indexed_label",
            maxIter=10,
            maxDepth=5,
            seed=RANDOM_SEED,
        )
        models_config["Gradient_Boosted_Trees"] = OneVsRest(
            classifier=gbt_base,
            featuresCol="features",
            labelCol="indexed_label",
        )
    else:
        logger.info("Skipping GBT (multi-class One-vs-Rest requires K separate ensembles; pass --include-gbt to run).")

    all_results = []
    all_per_class = {}
    all_cms = {}
    trained_models = {}

    for model_name, clf in models_config.items():
        logger.info(f"\n{'='*30} Training {model_name} {'='*30}")
        start_time = time.time()
        fitted_model = clf.fit(train_df)
        train_time = time.time() - start_time
        logger.info(f"{model_name} training finished in {train_time:.2f} seconds.")

        logger.info(f"Evaluating {model_name} on held-out test partition...")
        predictions = fitted_model.transform(test_df)

        global_metrics, per_class, cm_list, label_names = evaluate_predictions(predictions, label_map)
        global_metrics["model_name"] = model_name
        global_metrics["train_time_sec"] = round(train_time, 2)

        logger.info(f"Results for {model_name}:")
        logger.info(f"  Accuracy:         {global_metrics['accuracy']:.4f}")
        logger.info(f"  Macro-F1:         {global_metrics['macro_f1']:.4f} (Selection Criterion)")
        logger.info(f"  Weighted-F1:      {global_metrics['weighted_f1']:.4f}")
        logger.info(f"  Macro-Precision:  {global_metrics['macro_precision']:.4f}")
        logger.info(f"  Macro-Recall:     {global_metrics['macro_recall']:.4f}")

        all_results.append(global_metrics)
        all_per_class[model_name] = per_class
        all_cms[model_name] = {"matrix": cm_list, "labels": label_names}
        trained_models[model_name] = fitted_model

        # Save individual model
        model_save_path = MODELS_DIR / model_name
        fitted_model.write().overwrite().save(str(model_save_path))

    # Identify Best Model by Macro-F1 (robust against severe class imbalance)
    all_results.sort(key=lambda x: x["macro_f1"], reverse=True)
    best_summary = all_results[0]
    best_model_name = best_summary["model_name"]
    logger.info(f"\n>>> Best Model: {best_model_name} with Macro-F1: {best_summary['macro_f1']:.4f}")

    # Persist best model copy
    best_model_path = MODELS_DIR / "best_model"
    trained_models[best_model_name].write().overwrite().save(str(best_model_path))
    logger.info(f"Persisted best model to {best_model_path}")

    # Save metrics JSON & CSV
    with open(METRICS_DIR / "model_comparison.json", "w") as f:
        json.dump({"models": all_results, "best_model": best_summary}, f, indent=2)

    pd.DataFrame(all_results).to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    with open(METRICS_DIR / "per_class_metrics.json", "w") as f:
        json.dump(all_per_class, f, indent=2)

    with open(METRICS_DIR / "confusion_matrices.json", "w") as f:
        json.dump(all_cms, f, indent=2)

    train_df.unpersist()
    test_df.unpersist()
    logger.info("[SUCCESS] Training & evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Spark MLlib classifiers.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing model outputs.")
    parser.add_argument("--include-gbt", action="store_true", help="Include Gradient Boosted Trees via OneVsRest.")
    args = parser.parse_args()
    train_and_evaluate_all(force=args.force, include_gbt=args.include_gbt)
