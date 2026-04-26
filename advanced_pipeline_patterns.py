"""
Advanced Pipeline Patterns (2026)
===================================
Drop-in additions to your existing event-driven ML pipeline covering:

  1. Duplicate Event Handling   — idempotency key deduplication via DynamoDB
  2. Idempotent Pipelines       — skip re-runs for already-processed S3 files
  3. Failure + Retry Logic      — exponential backoff, dead-letter queue, alerting
  4. Model Versioning           — semantic versioning + lineage tracking in Model Registry

These patterns slot into lambda_trigger.py and sagemaker_pipeline.py.
Each section is self-contained — adopt what you need.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REGION       = os.environ.get("AWS_REGION",           "ap-southeast-2")
BUCKET       = os.environ.get("TRAINING_DATA_BUCKET", "sagemaker-ml-pipeline-585097636488")
PIPELINE_NAME= os.environ.get("PIPELINE_NAME",        "fraud-detection-retrain")
ACCOUNT_ID   = "585097636488"

# ── AWS clients ───────────────────────────────────────────────────────────────
dynamodb      = boto3.resource("dynamodb", region_name=REGION)
sagemaker_cl  = boto3.client("sagemaker",  region_name=REGION)
sns_client    = boto3.client("sns",        region_name=REGION)
ssm_client    = boto3.client("ssm",        region_name=REGION)
s3_client     = boto3.client("s3",         region_name=REGION)

SNS_TOPIC_ARN = f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:ml-pipeline-alerts"


# =============================================================================
# 1. DUPLICATE EVENT HANDLING
# =============================================================================
#
# Problem:
#   EventBridge can deliver the same S3 event more than once (at-least-once
#   delivery). Without deduplication, the same CSV triggers two training jobs
#   wasting compute and money.
#
# Solution:
#   Store a fingerprint (MD5 of bucket+key+etag) in DynamoDB with a TTL.
#   Before starting the pipeline, check if this fingerprint was seen recently.
#   If yes → skip. If no → record it and proceed.
#
# Setup (run once):
#   aws dynamodb create-table \
#     --table-name ml-pipeline-dedup \
#     --attribute-definitions AttributeName=event_id,AttributeType=S \
#     --key-schema AttributeName=event_id,KeyType=HASH \
#     --billing-mode PAY_PER_REQUEST \
#     --region ap-southeast-2
#
#   # Enable TTL on the table
#   aws dynamodb update-time-to-live \
#     --table-name ml-pipeline-dedup \
#     --time-to-live-specification Enabled=true,AttributeName=ttl \
#     --region ap-southeast-2
# =============================================================================

DEDUP_TABLE  = "ml-pipeline-dedup"
DEDUP_TTL_H  = 24  # ignore duplicate events within 24 hours


def _make_event_id(bucket: str, key: str, etag: str = "") -> str:
    """
    Create a stable fingerprint for an S3 object.
    ETag changes when the file content changes — so a re-upload of different
    data correctly produces a new fingerprint and triggers a new training run.
    """
    raw = f"{bucket}:{key}:{etag}"
    return hashlib.md5(raw.encode()).hexdigest()


def is_duplicate_event(bucket: str, key: str, etag: str = "") -> bool:
    """
    Returns True if this exact S3 object was already processed recently.
    Uses DynamoDB conditional write for atomic check-and-set.
    """
    table    = dynamodb.Table(DEDUP_TABLE)
    event_id = _make_event_id(bucket, key, etag)
    ttl      = int((datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_H)).timestamp())

    try:
        table.put_item(
            Item={
                "event_id":   event_id,
                "bucket":     bucket,
                "s3_key":     key,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "ttl":        ttl,
            },
            # Only succeed if this event_id does NOT already exist
            ConditionExpression="attribute_not_exists(event_id)",
        )
        logger.info("Dedup check passed — new event: %s", event_id)
        return False  # not a duplicate, safe to proceed

    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            logger.warning(
                "DUPLICATE EVENT detected for s3://%s/%s (id=%s) — skipping",
                bucket, key, event_id,
            )
            return True  # duplicate — skip
        raise  # unexpected error — re-raise


# =============================================================================
# 2. IDEMPOTENT PIPELINES
# =============================================================================
#
# Problem:
#   Even with dedup, a Lambda retry or operator re-run might fire the pipeline
#   twice for the same file. SageMaker will happily start two executions.
#
# Solution:
#   Before starting a new execution, check if a pipeline execution for this
#   exact S3 file is already Running or Succeeded.
#   Also use a deterministic ClientRequestToken so SageMaker itself deduplicates.
#
# The token is: SHA256(pipeline_name + s3_uri)[:40]
# SageMaker rejects duplicate tokens within 8 hours → free server-side dedup.
# =============================================================================

def is_pipeline_already_running(s3_uri: str) -> bool:
    """
    Check if a pipeline execution for this exact S3 URI is already
    Running, Stopping, or Succeeded within the last 2 hours.
    Prevents duplicate training jobs from operator re-runs.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)

    try:
        paginator = sagemaker_cl.get_paginator("list_pipeline_executions")
        for page in paginator.paginate(
            PipelineName=PIPELINE_NAME,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=10,
        ):
            for execution in page.get("PipelineExecutionSummaries", []):
                created = execution["CreationTime"]
                if created < cutoff:
                    break  # older than window, stop checking

                status = execution["PipelineExecutionStatus"]
                if status in ("Executing", "Stopping", "Succeeded"):
                    # Check if this execution was triggered by the same S3 file
                    exec_arn = execution["PipelineExecutionArn"]
                    try:
                        params = sagemaker_cl.list_pipeline_parameters_for_execution(
                            PipelineExecutionArn=exec_arn
                        )["PipelineParameters"]
                        param_map = {p["Name"]: p["Value"] for p in params}
                        if param_map.get("InputDataUri") == s3_uri:
                            logger.warning(
                                "Pipeline already %s for %s (arn=%s) — skipping",
                                status, s3_uri, exec_arn,
                            )
                            return True
                    except Exception:
                        pass  # can't read params, continue checking

    except Exception as exc:
        logger.warning("Could not check existing executions: %s — proceeding anyway", exc)

    return False


def make_idempotency_token(pipeline_name: str, s3_uri: str) -> str:
    """
    Deterministic token for SageMaker ClientRequestToken.
    Same file → same token → SageMaker rejects the duplicate within 8 hours.
    Max length: 63 chars (SageMaker limit).
    """
    raw = f"{pipeline_name}:{s3_uri}"
    return hashlib.sha256(raw.encode()).hexdigest()[:63]


# =============================================================================
# 3. FAILURE + RETRY LOGIC
# =============================================================================
#
# Problem:
#   Transient AWS errors (throttling, capacity issues) can fail the Lambda
#   before it even starts the pipeline. Without retries, data is silently lost.
#
# Solution:
#   A) Exponential backoff with jitter for the start_pipeline call
#   B) Dead-letter queue (DLQ) — events that exhaust retries go to SQS
#   C) CloudWatch alarm → SNS notification for pipeline failures
#
# DLQ Setup (run once):
#   aws sqs create-queue \
#     --queue-name ml-pipeline-dlq \
#     --region ap-southeast-2
#
#   Then set this SQS queue as the Lambda Dead Letter Queue in AWS Console:
#   Lambda → sagemaker-pipeline-trigger → Configuration → Asynchronous invocation
#   → Dead-letter queue → select ml-pipeline-dlq
# =============================================================================

MAX_RETRIES  = 3
BASE_DELAY_S = 2   # seconds — doubles each retry: 2s, 4s, 8s


def start_pipeline_with_retry(
    s3_uri: str,
    output_prefix: str,
    accuracy_threshold: str = "0.80",
) -> Optional[str]:
    """
    Start a SageMaker Pipeline execution with exponential backoff retry.
    Returns the execution ARN on success, None if all retries exhausted.
    """
    token     = make_idempotency_token(PIPELINE_NAME, s3_uri)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    params = [
        {"Name": "InputDataUri",         "Value": s3_uri},
        {"Name": "OutputS3Prefix",       "Value": output_prefix},
        {"Name": "AccuracyThreshold",    "Value": accuracy_threshold},
        {"Name": "ModelApprovalStatus",  "Value": "PendingManualApproval"},
    ]

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = sagemaker_cl.start_pipeline_execution(
                PipelineName=PIPELINE_NAME,
                PipelineExecutionDisplayName=f"auto-retrain-{timestamp}",
                PipelineParameters=params,
                PipelineExecutionDescription=f"Auto-triggered: {s3_uri}",
                ClientRequestToken=token,  # idempotency
            )
            arn = response["PipelineExecutionArn"]
            logger.info("Pipeline started (attempt %d/%d): %s", attempt, MAX_RETRIES, arn)
            return arn

        except sagemaker_cl.exceptions.ResourceLimitExceeded as exc:
            # Account hit max concurrent pipelines — wait longer
            delay = BASE_DELAY_S * (2 ** attempt) + (time.time() % 1)
            logger.warning(
                "ResourceLimitExceeded (attempt %d/%d) — retrying in %.1fs",
                attempt, MAX_RETRIES, delay,
            )
            last_exc = exc
            time.sleep(delay)

        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("ThrottlingException", "ServiceUnavailable", "RequestExpired"):
                delay = BASE_DELAY_S * (2 ** attempt) + (time.time() % 1)
                logger.warning(
                    "%s (attempt %d/%d) — retrying in %.1fs",
                    code, attempt, MAX_RETRIES, delay,
                )
                last_exc = exc
                time.sleep(delay)
            else:
                # Non-retriable error (ValidationException etc.) — fail fast
                logger.error("Non-retriable error starting pipeline: %s", exc)
                _send_failure_alert(
                    subject="Pipeline trigger failed — non-retriable error",
                    message=f"S3 file: {s3_uri}\nError: {exc}",
                )
                raise

    # All retries exhausted
    logger.error("All %d retries exhausted. Last error: %s", MAX_RETRIES, last_exc)
    _send_failure_alert(
        subject=f"Pipeline trigger failed after {MAX_RETRIES} retries",
        message=f"S3 file: {s3_uri}\nLast error: {last_exc}",
    )
    return None


def _send_failure_alert(subject: str, message: str) -> None:
    """Send an SNS alert email when the pipeline fails to start."""
    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[ML Pipeline Alert] {subject}",
            Message=(
                f"{message}\n\n"
                f"Pipeline:  {PIPELINE_NAME}\n"
                f"Region:    {REGION}\n"
                f"Account:   {ACCOUNT_ID}\n"
                f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n"
                f"Console: https://{REGION}.console.aws.amazon.com/sagemaker/home"
                f"?region={REGION}#/pipelines/{PIPELINE_NAME}/executions"
            ),
        )
        logger.info("Failure alert sent to SNS")
    except Exception as exc:
        logger.warning("Could not send SNS alert: %s", exc)


def monitor_execution_and_alert(execution_arn: str, s3_uri: str) -> None:
    """
    Poll a pipeline execution until it finishes, then send an SNS alert.
    Call this in a Step Functions state machine or a separate Lambda
    triggered on a schedule — NOT in the trigger Lambda (it would time out).

    Alternatively set up a CloudWatch Events rule on pipeline state changes.
    """
    while True:
        resp   = sagemaker_cl.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = resp["PipelineExecutionStatus"]

        if status == "Succeeded":
            _send_failure_alert(  # reusing helper — subject makes it clear it's success
                subject="Pipeline SUCCEEDED ✓",
                message=(
                    f"Training completed successfully.\n"
                    f"Input data: {s3_uri}\n"
                    f"Execution:  {execution_arn}\n"
                    f"Model is pending approval in the Model Registry."
                ),
            )
            break

        elif status in ("Failed", "Stopped"):
            _send_failure_alert(
                subject=f"Pipeline {status} ✗",
                message=(
                    f"Training pipeline did not complete.\n"
                    f"Input data: {s3_uri}\n"
                    f"Execution:  {execution_arn}\n"
                    f"Check CloudWatch logs for details."
                ),
            )
            break

        time.sleep(60)  # poll every minute


# =============================================================================
# 4. MODEL VERSIONING
# =============================================================================
#
# Problem:
#   SageMaker Model Registry auto-increments version numbers (1, 2, 3...)
#   but doesn't give you semantic versioning (1.0.0, 1.1.0, 2.0.0) or
#   easy rollback to a known-good version.
#
# Solution:
#   A) Store semantic version in SSM Parameter Store alongside each model
#   B) Tag every model package with the S3 source file, git commit, accuracy
#   C) Helper functions: get latest approved, promote, rollback
#
# Semantic versioning rules used here:
#   MAJOR bumps when accuracy drops > 5% from previous (potential regression)
#   MINOR bumps for routine retraining with new data
#   PATCH bumps for config-only changes (threshold, hyperparameters)
# =============================================================================

MODEL_PKG_GROUP  = "FraudDetectionModels"
SSM_VERSION_KEY  = f"/ml-pipeline/{MODEL_PKG_GROUP}/current-version"
SSM_APPROVED_KEY = f"/ml-pipeline/{MODEL_PKG_GROUP}/latest-approved-arn"


def get_current_semantic_version() -> tuple[int, int, int]:
    """Read the current semantic version from SSM. Returns (0,0,0) if none exists."""
    try:
        val = ssm_client.get_parameter(Name=SSM_VERSION_KEY)["Parameter"]["Value"]
        major, minor, patch = val.strip("v").split(".")
        return int(major), int(minor), int(patch)
    except ssm_client.exceptions.ParameterNotFound:
        return (0, 0, 0)
    except Exception as exc:
        logger.warning("Could not read version from SSM: %s — defaulting to 0.0.0", exc)
        return (0, 0, 0)


def bump_version(
    current_accuracy: float,
    previous_accuracy: float,
    change_type: str = "minor",
) -> str:
    """
    Compute the next semantic version string.

    change_type:
      "major" — breaking change or accuracy regression > 5%
      "minor" — new training data (default)
      "patch" — config-only change
    """
    major, minor, patch = get_current_semantic_version()

    # Auto-detect regression
    if previous_accuracy > 0 and (previous_accuracy - current_accuracy) > 0.05:
        logger.warning(
            "Accuracy dropped %.1f%% → %.1f%% — bumping MAJOR version (regression warning)",
            previous_accuracy * 100, current_accuracy * 100,
        )
        change_type = "major"

    if change_type == "major":
        major += 1; minor = 0; patch = 0
    elif change_type == "minor":
        minor += 1; patch = 0
    else:
        patch += 1

    version = f"v{major}.{minor}.{patch}"
    logger.info("New model version: %s (accuracy: %.4f)", version, current_accuracy)
    return version


def register_versioned_model(
    execution_arn: str,
    s3_uri: str,
    accuracy: float,
    auc: float,
    f1: float,
) -> str:
    """
    After a successful pipeline run, tag the registered model package
    with semantic version and full lineage metadata.
    Returns the new semantic version string.
    """
    # Get previous accuracy for regression detection
    try:
        prev_acc_str = ssm_client.get_parameter(
            Name=f"/ml-pipeline/{MODEL_PKG_GROUP}/last-accuracy"
        )["Parameter"]["Value"]
        previous_accuracy = float(prev_acc_str)
    except Exception:
        previous_accuracy = 0.0

    # Compute new version
    new_version = bump_version(accuracy, previous_accuracy)

    # Find the model package that was just registered by this execution
    packages = sagemaker_cl.list_model_packages(
        ModelPackageGroupName=MODEL_PKG_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]

    if not packages:
        logger.error("No model packages found in group %s", MODEL_PKG_GROUP)
        return new_version

    package_arn = packages[0]["ModelPackageArn"]

    # Add semantic version + lineage tags
    sagemaker_cl.add_tags(
        ResourceArn=package_arn,
        Tags=[
            {"Key": "SemanticVersion",  "Value": new_version},
            {"Key": "TrainingDataUri",  "Value": s3_uri},
            {"Key": "PipelineExecArn",  "Value": execution_arn},
            {"Key": "Accuracy",         "Value": str(round(accuracy, 6))},
            {"Key": "AUC",              "Value": str(round(auc, 6))},
            {"Key": "F1Score",          "Value": str(round(f1, 6))},
            {"Key": "RegisteredAt",     "Value": datetime.now(timezone.utc).isoformat()},
            {"Key": "Project",          "Value": "EventDrivenML"},
        ],
    )
    logger.info("Tagged model package %s with version %s", package_arn, new_version)

    # Persist version + accuracy to SSM for next run's comparison
    for name, value in [
        (SSM_VERSION_KEY,  new_version),
        (SSM_APPROVED_KEY, package_arn),
        (f"/ml-pipeline/{MODEL_PKG_GROUP}/last-accuracy", str(accuracy)),
    ]:
        ssm_client.put_parameter(Name=name, Value=value, Type="String", Overwrite=True)

    logger.info("SSM updated — current version: %s, accuracy: %.4f", new_version, accuracy)
    return new_version


def get_latest_approved_model() -> Optional[str]:
    """Return the ARN of the latest Approved model package, or None."""
    packages = sagemaker_cl.list_model_packages(
        ModelPackageGroupName=MODEL_PKG_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]

    if packages:
        arn = packages[0]["ModelPackageArn"]
        logger.info("Latest approved model: %s", arn)
        return arn

    logger.warning("No approved models found in group %s", MODEL_PKG_GROUP)
    return None


def rollback_to_version(target_version: str) -> Optional[str]:
    """
    Find a model package by semantic version tag and approve it,
    effectively rolling back production to that version.

    Usage: rollback_to_version("v1.3.0")
    """
    paginator = sagemaker_cl.get_paginator("list_model_packages")
    for page in paginator.paginate(ModelPackageGroupName=MODEL_PKG_GROUP):
        for pkg in page["ModelPackageSummaryList"]:
            tags_resp = sagemaker_cl.list_tags(ResourceArn=pkg["ModelPackageArn"])
            tags      = {t["Key"]: t["Value"] for t in tags_resp["Tags"]}

            if tags.get("SemanticVersion") == target_version:
                arn = pkg["ModelPackageArn"]
                sagemaker_cl.update_model_package(
                    ModelPackageArn=arn,
                    ModelApprovalStatus="Approved",
                )
                logger.info("Rolled back to version %s: %s", target_version, arn)
                ssm_client.put_parameter(
                    Name=SSM_APPROVED_KEY, Value=arn, Type="String", Overwrite=True
                )
                return arn

    logger.error("Version %s not found in model registry", target_version)
    return None


# =============================================================================
# UPDATED LAMBDA HANDLER — integrates all 4 patterns
# =============================================================================
#
# Replace lambda_trigger.py's lambda_handler with this version.
# It wires together dedup, idempotency check, retry logic, and version tracking.
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    """
    Production-grade Lambda handler with all 4 patterns applied.

    Flow:
      1. Parse S3 event
      2. Guard clauses (bucket, prefix, size, extension)
      3. Duplicate event check (DynamoDB)
      4. Idempotency check (already running?)
      5. Start pipeline with retry + backoff
      6. Store execution ARN in SSM
    """
    import json as _json

    logger.info("Event: %s", _json.dumps(event))

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    try:
        detail = event["detail"]
        bucket = detail["bucket"]["name"]
        key    = detail["object"]["key"]
        size   = detail["object"].get("size", 0)
        etag   = detail["object"].get("etag", "")
    except (KeyError, TypeError) as exc:
        logger.error("Malformed event: %s", exc)
        return {"statusCode": 400, "body": "Malformed event"}

    s3_uri = f"s3://{bucket}/{key}"

    # ── 2. Guard clauses ──────────────────────────────────────────────────────
    allowed_bucket = os.environ.get("TRAINING_DATA_BUCKET", BUCKET)
    data_prefix    = os.environ.get("DATA_PREFIX", "data/")
    min_size       = int(os.environ.get("MIN_FILE_SIZE", "1024"))

    if bucket != allowed_bucket:
        return {"statusCode": 200, "body": "Skipped: wrong bucket"}
    if not key.startswith(data_prefix):
        return {"statusCode": 200, "body": "Skipped: wrong prefix"}
    if not key.lower().endswith(".csv"):
        return {"statusCode": 200, "body": "Skipped: not a CSV"}
    if size < min_size:
        return {"statusCode": 200, "body": "Skipped: file too small"}

    # ── 3. Duplicate event check ──────────────────────────────────────────────
    if is_duplicate_event(bucket, key, etag):
        return {"statusCode": 200, "body": "Skipped: duplicate event"}

    # ── 4. Idempotency check ──────────────────────────────────────────────────
    if is_pipeline_already_running(s3_uri):
        return {"statusCode": 200, "body": "Skipped: pipeline already running for this file"}

    # ── 5. Start pipeline with retry ──────────────────────────────────────────
    timestamp     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_prefix = f"s3://{allowed_bucket}/pipeline-outputs/{timestamp}"

    execution_arn = start_pipeline_with_retry(
        s3_uri=s3_uri,
        output_prefix=output_prefix,
        accuracy_threshold=os.environ.get("ACCURACY_THRESHOLD", "0.80"),
    )

    if not execution_arn:
        return {"statusCode": 500, "body": "Failed to start pipeline after retries"}

    # ── 6. Store in SSM ───────────────────────────────────────────────────────
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
    return {
        "statusCode": 200,
        "body": _json.dumps({
            "executionArn": execution_arn,
            "inputData":    s3_uri,
            "timestamp":    timestamp,
        }),
    }


# =============================================================================
# CLI — quick utilities for ops use
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Advanced pipeline ops utilities")
    parser.add_argument("--latest-approved",  action="store_true", help="Print latest approved model ARN")
    parser.add_argument("--rollback",         metavar="VERSION",   help="Rollback to a semantic version e.g. v1.3.0")
    parser.add_argument("--current-version",  action="store_true", help="Print current semantic version")
    parser.add_argument("--check-dup",        metavar="S3_URI",    help="Check if an S3 URI would be a duplicate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.latest_approved:
        arn = get_latest_approved_model()
        print(f"Latest approved: {arn or 'None found'}")

    elif args.rollback:
        arn = rollback_to_version(args.rollback)
        print(f"Rolled back to {args.rollback}: {arn or 'Not found'}")

    elif args.current_version:
        v = get_current_semantic_version()
        print(f"Current version: v{v[0]}.{v[1]}.{v[2]}")

    elif args.check_dup:
        parts  = args.check_dup.replace("s3://", "").split("/", 1)
        result = is_duplicate_event(parts[0], parts[1])
        print(f"Duplicate: {result}")

    else:
        parser.print_help()
