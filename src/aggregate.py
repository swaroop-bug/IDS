"""Aggregate Module for Dashboard-Ready Datasets.

Combines metrics, EDA summaries, feature importances, and temporal statistics
into lightweight, optimized CSV and JSON artifacts stored in `data/output/aggregates/`.
Enables instant Streamlit dashboard loading with zero runtime Spark dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import (
    AGGREGATES_DIR,
    BENCHMARK_DIR,
    EDA_DIR,
    METRICS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aggregate")

def build_aggregates(force: bool = False) -> None:
    """Precompute all aggregated summaries for the Streamlit dashboard."""
    kpi_file = AGGREGATES_DIR / "dashboard_kpi.json"
    if kpi_file.exists() and not force:
        logger.info(f"Dashboard aggregates already exist at {AGGREGATES_DIR}. Skipping. Use --force to rerun.")
        return

    logger.info("Building dashboard aggregates...")

    # 1. KPI Summary
    eda_summary_path = EDA_DIR / "summary_metrics.json"
    model_comparison_path = METRICS_DIR / "model_comparison.json"

    eda_summary = {}
    if eda_summary_path.exists():
        with open(eda_summary_path, "r") as f:
            eda_summary = json.load(f)

    model_summary = {}
    if model_comparison_path.exists():
        with open(model_comparison_path, "r") as f:
            model_summary = json.load(f)

    best_model = model_summary.get("best_model", {})

    kpi_data = {
        "total_records": eda_summary.get("total_records", 0),
        "total_benign": eda_summary.get("total_benign", 0),
        "total_attacks": eda_summary.get("total_attacks", 0),
        "attack_percentage": eda_summary.get("attack_percentage", 0.0),
        "distinct_attack_types": eda_summary.get("distinct_attack_types", 0),
        "best_model_name": best_model.get("model_name", "N/A"),
        "best_model_macro_f1": best_model.get("macro_f1", 0.0),
        "best_model_accuracy": best_model.get("accuracy", 0.0),
        "best_model_train_time": best_model.get("train_time_sec", 0.0),
    }

    with open(kpi_file, "w") as f:
        json.dump(kpi_data, f, indent=2)
    logger.info(f"Saved KPI summary to {kpi_file}")

    # 2. Timeline Aggregates
    timeline_path = EDA_DIR / "attacks_by_day_hour.csv"
    if timeline_path.exists():
        timeline_df = pd.read_csv(timeline_path)
        timeline_df.to_csv(AGGREGATES_DIR / "attack_timeline.csv", index=False)
        logger.info(f"Exported attack timeline to {AGGREGATES_DIR / 'attack_timeline.csv'}")

    # 3. Class Breakdown Aggregates
    class_dist_path = EDA_DIR / "class_distribution.csv"
    if class_dist_path.exists():
        class_df = pd.read_csv(class_dist_path)
        class_df.to_csv(AGGREGATES_DIR / "class_distribution.csv", index=False)
        logger.info(f"Exported class distribution to {AGGREGATES_DIR / 'class_distribution.csv'}")

    # 4. Model Comparison Aggregates
    if "models" in model_summary:
        models_df = pd.DataFrame(model_summary["models"])
        models_df.to_csv(AGGREGATES_DIR / "model_comparison.csv", index=False)
        logger.info(f"Exported model comparison table to {AGGREGATES_DIR / 'model_comparison.csv'}")

    # 5. Per-class F1 Aggregates
    per_class_path = METRICS_DIR / "per_class_metrics.json"
    if per_class_path.exists():
        with open(per_class_path, "r") as f:
            per_class_raw = json.load(f)

        flat_records = []
        for model_name, class_dict in per_class_raw.items():
            for label, scores in class_dict.items():
                flat_records.append({
                    "model_name": model_name,
                    "label": label,
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1_score": scores["f1_score"],
                })
        per_class_df = pd.DataFrame(flat_records)
        per_class_df.to_csv(AGGREGATES_DIR / "per_class_metrics.csv", index=False)
        logger.info(f"Exported per-class metrics to {AGGREGATES_DIR / 'per_class_metrics.csv'}")

    # 6. Confusion Matrices Aggregates
    cm_path = METRICS_DIR / "confusion_matrices.json"
    if cm_path.exists():
        with open(cm_path, "r") as f:
            cms = json.load(f)
        with open(AGGREGATES_DIR / "confusion_matrices.json", "w") as f:
            json.dump(cms, f, indent=2)
        logger.info(f"Exported confusion matrices to {AGGREGATES_DIR / 'confusion_matrices.json'}")

    # 7. Feature Importance Aggregates
    feat_imp_path = EDA_DIR / "feature_importance.csv"
    if feat_imp_path.exists():
        feat_df = pd.read_csv(feat_imp_path)
        feat_df.to_csv(AGGREGATES_DIR / "feature_importance.csv", index=False)
        logger.info(f"Exported feature importance to {AGGREGATES_DIR / 'feature_importance.csv'}")

    # 8. Benchmark Aggregates
    bench_path = BENCHMARK_DIR / "runtime_benchmark.csv"
    if bench_path.exists():
        bench_df = pd.read_csv(bench_path)
        bench_df.to_csv(AGGREGATES_DIR / "runtime_benchmark.csv", index=False)
        logger.info(f"Exported benchmark aggregates to {AGGREGATES_DIR / 'runtime_benchmark.csv'}")

    logger.info("[SUCCESS] Aggregation stage complete. Ready for instant dashboard consumption.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dashboard-ready aggregate metrics.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing aggregates.")
    args = parser.parse_args()
    build_aggregates(force=args.force)
