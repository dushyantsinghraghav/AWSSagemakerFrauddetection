"""
Lambda Trigger — Production Grade (2026)
-----------------------------------------
Triggered by EventBridge when a new CSV lands in S3.
Integrates all 4 advanced patterns from advanced_pipeline_patterns.py:

  1. Duplicate event handling  — DynamoDB fingerprint dedup
  2. Idempotent pipelines      — checks for already-running executions
  3. Failure + retry logic     — exponential backoff + SNS alerts
  4. Model versioning          — semantic version tags on every model

Deploy:
    python setup_infrastructure.py --deploy
"""

import json
import logging
import os
from datetime import datetime, timezone

# Import all 4 patterns from the companion module
from advanced_pipeline_patterns import (
    is_duplicate_event,
    is_pipeline_already_running,
    start_pipeline_with_retry,
)

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ssm_client = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "ap-southeast-2"))

# ── Environment variables (set in Lambda config by setup_infrastructure.py) ───
PIPELINE_NAME  = os.environ.get("PIPELINE_NAME",        "fraud-detection-retrain")
BUCKET         = os.environ.get("TRAINING_DATA_BUCKET", "sagemaker-ml-pipeline-585097636488")
DATA_PREFIX    = os.environ.get("DATA_PREFIX",           "data/")
MIN_FILE_SIZE  = int(os.environ.get("MIN_FILE_SIZE",     "1024"))
ACC_THRESHOLD  = os.environ.get("ACCURACY_THRESHOLD",   "0.80")


def lambda_handler(event: dict, context) -> dict:
    """
    Main entry point. EventBridge delivers S3 ObjectCreated events here.

    Full flow:
      1. Parse S3 details from EventBridge event
      2. Guard clauses — skip irrelevant files
      3. Duplicate event check — DynamoDB fingerprint
      4. Idempotency check — already running for this file?
      5. Start pipeline with exponential backoff retry
      6. Store execution ARN in SSM for monitoring
    """
    logger.info("Received event: %s", json.dumps(event))

    # ── 1. Parse S3 details ───────────────────────────────────────────────────
    try:
        detail = event["detail"]
        bucket = detail["bucket"]["name"]
        key    = detail["object"]["key"]
        size   = detail["object"].get("size", 0)
        etag   = detail["object"].get("etag", "")
    except (KeyError, TypeError) as exc:
        logger.error("Malformed EventBridge event: %s", exc)
        return _response(400, "Malformed event — check EventBridge rule pattern")

    s3_uri = f"s3://{bucket}/{key}"
    logger.info("S3 object detected: %s (%d bytes)", s3_uri, size)

    # ── 2. Guard clauses ──────────────────────────────────────────────────────
    if bucket != BUCKET:
        logger.info("Wrong bucket: %s — skipping", bucket)
        return _response(200, "Skipped: wrong bucket")

    if not key.startswith(DATA_PREFIX):
        logger.info("Wrong prefix: %s — skipping", key)
        return _response(200, "Skipped: wrong prefix")

    if not key.lower().endswith(".csv"):
        logger.info("Not a CSV: %s — skipping", key)
        return _response(200, "Skipped: not a CSV")

    if size < MIN_FILE_SIZE:
        logger.warning("File too small (%d bytes): %s — skipping", size, key)
        return _response(200, "Skipped: file too small")

    # ── 3. Duplicate event check (DynamoDB) ───────────────────────────────────
    if is_duplicate_event(bucket, key, etag):
        return _response(200, "Skipped: duplicate event — already processed this file recently")

    # ── 4. Idempotency check (SageMaker execution history) ────────────────────
    if is_pipeline_already_running(s3_uri):
        return _response(200, "Skipped: pipeline already running for this exact file")

    # ── 5. Start pipeline with retry + backoff ────────────────────────────────
    timestamp     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_prefix = f"s3://{BUCKET}/pipeline-outputs/{timestamp}"

    execution_arn = start_pipeline_with_retry(
        s3_uri=s3_uri,
        output_prefix=output_prefix,
        accuracy_threshold=ACC_THRESHOLD,
    )

    if not execution_arn:
        return _response(500, "Pipeline failed to start after all retries — check SNS alert")

    # ── 6. Store last execution ARN in SSM ────────────────────────────────────
    try:
        ssm_client.put_parameter(
            Name=f"/ml-pipeline/{PIPELINE_NAME}/last-execution-arn",
            Value=execution_arn,
            Type="String",
            Overwrite=True,
        )
    except Exception as exc:
        logger.warning("SSM update failed (non-fatal): %s", exc)

    logger.info("Pipeline triggered successfully: %s", execution_arn)
    return _response(200, {
        "message":      "Pipeline triggered successfully",
        "executionArn": execution_arn,
        "inputData":    s3_uri,
        "timestamp":    timestamp,
    })


def _response(status_code: int, body) -> dict:
    return {
        "statusCode": status_code,
        "body": json.dumps(body) if isinstance(body, dict) else body,
    }
