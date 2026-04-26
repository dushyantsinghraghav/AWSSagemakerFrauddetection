"""
evaluate.py — SageMaker Processing script (Evaluation Step)
------------------------------------------------------------
Loads the trained XGBoost model and held-out test set,
computes classification metrics, and writes evaluation.json.

The PropertyFile in sagemaker_pipeline.py reads:
  binary_classification_metrics.accuracy.value

This value drives the ConditionStep (register vs fail).
"""

import json
import logging
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR  = "/opt/ml/processing/model"
TEST_DIR   = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def load_model(model_dir: str) -> xgb.Booster:
    """Extract model.tar.gz and load the XGBoost booster."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    extract_dir = os.path.join(model_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    # Find the model file (xgboost saves as xgboost-model or model.xgb)
    for fname in os.listdir(extract_dir):
        if fname.endswith((".xgb", "-model", ".bin")):
            model_path = os.path.join(extract_dir, fname)
            break
    else:
        raise FileNotFoundError(f"No XGBoost model file found in {extract_dir}")

    booster = xgb.Booster()
    booster.load_model(model_path)
    logger.info("Loaded model from %s", model_path)
    return booster


def load_test_data(test_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load test CSV (no header, first column = label)."""
    csv_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {test_dir}")

    df = pd.concat(
        [pd.read_csv(os.path.join(test_dir, f), header=None) for f in csv_files],
        ignore_index=True,
    )
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    logger.info("Test set: %d samples, %d features", *X.shape)
    return X, y


def evaluate(booster: xgb.Booster, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute accuracy, AUC, and F1, return as SageMaker evaluation report schema."""
    dmat = xgb.DMatrix(X)
    y_prob = booster.predict(dmat)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(accuracy_score(y, y_pred))
    auc      = float(roc_auc_score(y, y_prob))
    f1       = float(f1_score(y, y_pred, average="binary"))

    logger.info("Accuracy: %.4f | AUC: %.4f | F1: %.4f", accuracy, auc, f1)
    logger.info("\n%s", classification_report(y, y_pred))

    # SageMaker Model Registry evaluation report schema
    report = {
        "binary_classification_metrics": {
            "accuracy": {
                "value":         accuracy,
                "standard_deviation": "NaN",
            },
            "auc": {
                "value":         auc,
                "standard_deviation": "NaN",
            },
            "f1": {
                "value":         f1,
                "standard_deviation": "NaN",
            },
        }
    }
    return report


def main():
    booster  = load_model(MODEL_DIR)
    X, y     = load_test_data(TEST_DIR)
    report   = evaluate(booster, X, y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report written to %s", out_path)
    logger.info("Final accuracy: %.4f", report["binary_classification_metrics"]["accuracy"]["value"])


if __name__ == "__main__":
    main()
