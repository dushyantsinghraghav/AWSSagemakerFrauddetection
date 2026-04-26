"""
preprocess.py — SageMaker Processing script
--------------------------------------------
Runs inside the SKLearnProcessor container.
Reads raw CSV from /opt/ml/processing/input,
engineers features, and writes train/validation/test splits
to /opt/ml/processing/output/{train,validation,test}.

This script is uploaded to S3 and referenced in sagemaker_pipeline.py.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR      = "/opt/ml/processing/input"
OUTPUT_TRAIN   = "/opt/ml/processing/output/train"
OUTPUT_VAL     = "/opt/ml/processing/output/validation"
OUTPUT_TEST    = "/opt/ml/processing/output/test"


def load_data(input_dir: str) -> pd.DataFrame:
    """Load all CSV files from the input directory and concatenate."""
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dfs = []
    for fname in files:
        path = os.path.join(input_dir, fname)
        df = pd.read_csv(path)
        dfs.append(df)
        logger.info("Loaded %s: %d rows", fname, len(df))

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Combined dataset: %d rows, %d columns", *combined.shape)
    return combined


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-specific feature engineering for fraud detection.
    Adapt column names to match your actual dataset schema.
    """
    df = df.copy()

    # ── Drop useless columns ──────────────────────────────────────────────────
    drop_cols = ["transaction_id", "customer_name", "raw_timestamp"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── Handle missing values ─────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "label" in cat_cols:
        cat_cols.remove("label")
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")

    # ── Encode categoricals ───────────────────────────────────────────────────
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Time-based features (if timestamp column exists) ──────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour_of_day"]   = df["timestamp"].dt.hour
        df["day_of_week"]   = df["timestamp"].dt.dayofweek
        df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
        df.drop(columns=["timestamp"], inplace=True)

    # ── Normalise numeric features (XGBoost doesn't need it but good practice) ─
    feature_cols = [c for c in df.columns if c != "label"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    logger.info("Feature engineering done. Shape: %s", df.shape)
    return df


def split_and_save(
    df: pd.DataFrame,
    test_size: float,
    validation_size: float,
) -> None:
    """Stratified split preserving class balance, then write CSV."""
    target = "label"
    X = df.drop(columns=[target])
    y = df[target]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    # Then split off validation from the remainder
    val_frac = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=42, stratify=y_temp
    )

    for name, (X_part, y_part), out_dir in [
        ("train",      (X_train, y_train), OUTPUT_TRAIN),
        ("validation", (X_val,   y_val),   OUTPUT_VAL),
        ("test",       (X_test,  y_test),  OUTPUT_TEST),
    ]:
        os.makedirs(out_dir, exist_ok=True)
        part = pd.concat([y_part.reset_index(drop=True), X_part.reset_index(drop=True)], axis=1)
        out_path = os.path.join(out_dir, f"{name}.csv")
        part.to_csv(out_path, index=False, header=False)  # XGBoost expects no header
        logger.info(
            "Saved %s: %d rows  (positive rate: %.2f%%)",
            out_path, len(part), y_part.mean() * 100
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size",       type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    args = parser.parse_args()

    df = load_data(INPUT_DIR)
    df = engineer_features(df)
    split_and_save(df, args.test_size, args.validation_size)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
