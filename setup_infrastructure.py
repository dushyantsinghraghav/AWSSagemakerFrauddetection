"""
Infrastructure Setup: EventBridge + Lambda + IAM
-------------------------------------------------
Provisions all AWS resources needed for event-driven ML pipeline triggering:

  1. Upload preprocess.py and evaluate.py to S3
  2. IAM roles for Lambda and SageMaker
  3. Lambda function (lambda_trigger.py + advanced_pipeline_patterns.py)
  4. EventBridge rule (S3 ObjectCreated → Lambda)
  5. S3 EventBridge notification enablement
  6. SNS topic for pipeline success/failure alerts

Run once to bootstrap, or re-run to update configs:
    python setup_infrastructure.py --deploy
    python setup_infrastructure.py --teardown
"""

import argparse
import json
import logging
import os
import time
import zipfile
from io import BytesIO
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REGION        = os.environ.get("AWS_REGION",           "ap-southeast-2")
ACCOUNT_ID    = boto3.client("sts").get_caller_identity()["Account"]
BUCKET        = os.environ.get("TRAINING_DATA_BUCKET", f"sagemaker-ml-pipeline-{ACCOUNT_ID}")
DATA_PREFIX   = os.environ.get("DATA_PREFIX",           "data/")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME",         "fraud-detection-retrain")

LAMBDA_FUNCTION_NAME = "sagemaker-pipeline-trigger"
LAMBDA_ROLE_NAME     = "LambdaSageMakerTriggerRole"
SAGEMAKER_ROLE_NAME  = "SageMakerPipelineRole"
EVENTBRIDGE_RULE     = "s3-new-data-trigger"
SNS_TOPIC_NAME       = "ml-pipeline-alerts"
LOG_GROUP            = f"/aws/lambda/{LAMBDA_FUNCTION_NAME}"

# ── Local file paths (all must be in same folder as this script) ──────────────
THIS_DIR                  = Path(__file__).parent
PREPROCESS_PATH           = THIS_DIR / "preprocess.py"
EVALUATE_PATH             = THIS_DIR / "evaluate.py"
LAMBDA_TRIGGER_PATH       = THIS_DIR / "lambda_trigger.py"
ADVANCED_PATTERNS_PATH    = THIS_DIR / "advanced_pipeline_patterns.py"

# ── AWS clients ───────────────────────────────────────────────────────────────
iam    = boto3.client("iam",    region_name=REGION)
lam    = boto3.client("lambda", region_name=REGION)
events = boto3.client("events", region_name=REGION)
s3     = boto3.client("s3",     region_name=REGION)
sns    = boto3.client("sns",    region_name=REGION)


# =============================================================================
# S3 Bucket + Script Upload
# =============================================================================

def create_bucket_if_missing(bucket: str) -> None:
    """Create the S3 bucket if it does not already exist."""
    try:
        s3.head_bucket(Bucket=bucket)
        logger.info("Bucket already exists: %s", bucket)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket"):
            logger.info("Creating bucket: %s in %s", bucket, REGION)
            if REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": REGION},
                )
            logger.info("Bucket created ✓")
        else:
            raise


def upload_pipeline_scripts() -> None:
    """
    Upload preprocess.py and evaluate.py to S3.
    These run inside SageMaker Processing containers.
    """
    scripts = [
        (PREPROCESS_PATH, "pipeline-scripts/preprocess.py"),
        (EVALUATE_PATH,   "pipeline-scripts/evaluate.py"),
    ]
    for local_path, s3_key in scripts:
        if not local_path.exists():
            raise FileNotFoundError(
                f"\nScript not found: {local_path}\n"
                f"Expected folder layout:\n"
                f"  your-project/\n"
                f"  ├── setup_infrastructure.py\n"
                f"  ├── preprocess.py                   ← must be here\n"
                f"  ├── evaluate.py                     ← must be here\n"
                f"  ├── lambda_trigger.py               ← must be here\n"
                f"  └── advanced_pipeline_patterns.py   ← must be here\n"
            )
        s3.upload_file(str(local_path), BUCKET, s3_key)
        logger.info("Uploaded %-30s → s3://%s/%s", local_path.name, BUCKET, s3_key)
    logger.info("Pipeline scripts uploaded ✓")


# =============================================================================
# IAM
# =============================================================================

LAMBDA_TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }]
})

# Updated: includes DynamoDB (dedup) and SNS (alerts) permissions
# needed by advanced_pipeline_patterns.py
LAMBDA_INLINE_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "StartPipeline",
            "Effect": "Allow",
            "Action": [
                "sagemaker:StartPipelineExecution",
                "sagemaker:DescribePipeline",
                "sagemaker:ListPipelineExecutions",
                "sagemaker:ListPipelineParameters",
                "sagemaker:ListPipelineParametersForExecution",
            ],
            "Resource": f"arn:aws:sagemaker:{REGION}:{ACCOUNT_ID}:pipeline/{PIPELINE_NAME}",
        },
        {
            "Sid": "ReadS3",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:HeadObject"],
            "Resource": f"arn:aws:s3:::{BUCKET}/{DATA_PREFIX}*",
        },
        {
            "Sid": "DynamoDBDedup",
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:GetItem",
                "dynamodb:UpdateItem",
            ],
            "Resource": f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/ml-pipeline-dedup",
        },
        {
            "Sid": "SSMParams",
            "Effect": "Allow",
            "Action": ["ssm:PutParameter", "ssm:GetParameter"],
            "Resource": f"arn:aws:ssm:{REGION}:{ACCOUNT_ID}:parameter/ml-pipeline/*",
        },
        {
            "Sid": "SNSAlerts",
            "Effect": "Allow",
            "Action": ["sns:Publish"],
            "Resource": f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:{SNS_TOPIC_NAME}",
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:{LOG_GROUP}:*",
        },
    ]
})

SAGEMAKER_TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }]
})


def create_lambda_role() -> str:
    try:
        role    = iam.create_role(
            RoleName=LAMBDA_ROLE_NAME,
            AssumeRolePolicyDocument=LAMBDA_TRUST_POLICY,
            Description="Allows Lambda to trigger SageMaker Pipelines from EventBridge",
            Tags=[{"Key": "Project", "Value": "EventDrivenML"}],
        )
        role_arn = role["Role"]["Arn"]
        logger.info("Created Lambda IAM role: %s", role_arn)
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=LAMBDA_ROLE_NAME)["Role"]["Arn"]
        logger.info("Lambda IAM role already exists: %s", role_arn)

    iam.attach_role_policy(
        RoleName=LAMBDA_ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    iam.put_role_policy(
        RoleName=LAMBDA_ROLE_NAME,
        PolicyName="SageMakerTriggerPolicy",
        PolicyDocument=LAMBDA_INLINE_POLICY,
    )
    time.sleep(10)
    return role_arn


def create_sagemaker_role() -> str:
    try:
        role    = iam.create_role(
            RoleName=SAGEMAKER_ROLE_NAME,
            AssumeRolePolicyDocument=SAGEMAKER_TRUST_POLICY,
            Description="SageMaker execution role for ML pipeline steps",
            Tags=[{"Key": "Project", "Value": "EventDrivenML"}],
        )
        role_arn = role["Role"]["Arn"]
        logger.info("Created SageMaker IAM role: %s", role_arn)
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=SAGEMAKER_ROLE_NAME)["Role"]["Arn"]
        logger.info("SageMaker IAM role already exists: %s", role_arn)

    for policy_arn in [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]:
        iam.attach_role_policy(RoleName=SAGEMAKER_ROLE_NAME, PolicyArn=policy_arn)

    time.sleep(10)
    return role_arn


# =============================================================================
# Lambda — bundles both lambda_trigger.py + advanced_pipeline_patterns.py
# =============================================================================

def _zip_lambda() -> bytes:
    """
    Package lambda_trigger.py AND advanced_pipeline_patterns.py into one ZIP.
    Both files must be present — lambda_trigger.py imports from advanced_pipeline_patterns.
    """
    for path in [LAMBDA_TRIGGER_PATH, ADVANCED_PATTERNS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"\n{path.name} not found in {THIS_DIR}\n"
                f"Make sure both lambda_trigger.py and advanced_pipeline_patterns.py\n"
                f"are in the same folder as setup_infrastructure.py"
            )

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # lambda_function.py is the Lambda entry point (handler name in config)
        zf.write(LAMBDA_TRIGGER_PATH,    arcname="lambda_function.py")
        # advanced_pipeline_patterns.py is imported by lambda_function.py
        zf.write(ADVANCED_PATTERNS_PATH, arcname="advanced_pipeline_patterns.py")
    logger.info("Zipped: lambda_function.py + advanced_pipeline_patterns.py")
    return buf.getvalue()


def deploy_lambda(role_arn: str) -> str:
    zip_bytes = _zip_lambda()
    env_vars  = {
        "PIPELINE_NAME":        PIPELINE_NAME,
        "TRAINING_DATA_BUCKET": BUCKET,
        "DATA_PREFIX":          DATA_PREFIX,
        "TRAINING_INSTANCE":    "ml.m5.medium",
        "MIN_FILE_SIZE":        "1024",
        "ACCURACY_THRESHOLD":   "0.80",
    }
    waiter = lam.get_waiter("function_updated")

    try:
        fn = lam.get_function(FunctionName=LAMBDA_FUNCTION_NAME)

        logger.info("Updating Lambda function code...")
        lam.update_function_code(
            FunctionName=LAMBDA_FUNCTION_NAME,
            ZipFile=zip_bytes,
        )
        logger.info("Waiting for code update to complete...")
        waiter.wait(FunctionName=LAMBDA_FUNCTION_NAME)

        logger.info("Updating Lambda function configuration...")
        lam.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Environment={"Variables": env_vars},
            Timeout=60,
            MemorySize=256,
        )
        logger.info("Waiting for config update to complete...")
        waiter.wait(FunctionName=LAMBDA_FUNCTION_NAME)

        fn_arn = fn["Configuration"]["FunctionArn"]
        logger.info("Updated Lambda function ✓  %s", fn_arn)

    except lam.exceptions.ResourceNotFoundException:
        logger.info("Creating Lambda function...")
        fn = lam.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Description="Triggers SageMaker Pipeline when new CSV arrives in S3",
            Timeout=60,
            MemorySize=256,
            Environment={"Variables": env_vars},
            Tags={"Project": "EventDrivenML"},
        )
        logger.info("Waiting for Lambda to become active...")
        waiter.wait(FunctionName=LAMBDA_FUNCTION_NAME)
        fn_arn = fn["FunctionArn"]
        logger.info("Created Lambda function ✓  %s", fn_arn)

    return fn_arn


# =============================================================================
# S3 EventBridge + Rule
# =============================================================================

def enable_s3_eventbridge(bucket: str) -> None:
    try:
        s3.put_bucket_notification_configuration(
            Bucket=bucket,
            NotificationConfiguration={"EventBridgeConfiguration": {}},
        )
        logger.info("Enabled EventBridge notifications for bucket: %s", bucket)
    except ClientError as exc:
        logger.error("Failed to enable S3→EventBridge: %s", exc)
        raise


def create_eventbridge_rule(lambda_arn: str) -> str:
    event_pattern = json.dumps({
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "detail": {
            "bucket": {"name": [BUCKET]},
            "object": {"key": [{"prefix": DATA_PREFIX}]},
        },
    })

    rule_resp = events.put_rule(
        Name=EVENTBRIDGE_RULE,
        EventPattern=event_pattern,
        State="ENABLED",
        Description=f"Triggers SageMaker retraining on new CSV in s3://{BUCKET}/{DATA_PREFIX}",
        Tags=[{"Key": "Project", "Value": "EventDrivenML"}],
    )
    rule_arn = rule_resp["RuleArn"]
    logger.info("EventBridge rule: %s", rule_arn)

    events.put_targets(
        Rule=EVENTBRIDGE_RULE,
        Targets=[{"Id": "SageMakerPipelineTrigger", "Arn": lambda_arn}],
    )

    try:
        lam.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId="EventBridgeInvoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
        logger.info("Granted EventBridge permission to invoke Lambda ✓")
    except lam.exceptions.ResourceConflictException:
        logger.info("Lambda permission for EventBridge already exists")

    return rule_arn


# =============================================================================
# SNS Alerts
# =============================================================================

def create_sns_alerts(email: str | None = None) -> str:
    resp      = sns.create_topic(
        Name=SNS_TOPIC_NAME,
        Tags=[{"Key": "Project", "Value": "EventDrivenML"}],
    )
    topic_arn = resp["TopicArn"]
    logger.info("SNS topic: %s", topic_arn)

    if email:
        sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)
        logger.info("SNS subscription pending email confirmation: %s", email)

    return topic_arn


# =============================================================================
# Teardown
# =============================================================================

def teardown() -> None:
    def _try(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Exception as exc:
            logger.warning("Teardown partial failure: %s", exc)

    _try(events.remove_targets, Rule=EVENTBRIDGE_RULE, Ids=["SageMakerPipelineTrigger"])
    _try(events.delete_rule,    Name=EVENTBRIDGE_RULE)
    _try(lam.delete_function,   FunctionName=LAMBDA_FUNCTION_NAME)

    _try(iam.detach_role_policy, RoleName=LAMBDA_ROLE_NAME,
         PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    _try(iam.delete_role_policy, RoleName=LAMBDA_ROLE_NAME, PolicyName="SageMakerTriggerPolicy")
    _try(iam.delete_role,        RoleName=LAMBDA_ROLE_NAME)

    for p in [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]:
        _try(iam.detach_role_policy, RoleName=SAGEMAKER_ROLE_NAME, PolicyArn=p)
    _try(iam.delete_role, RoleName=SAGEMAKER_ROLE_NAME)

    logger.info("Teardown complete")


# =============================================================================
# Entry point
# =============================================================================

def deploy(alert_email: str | None = None) -> None:
    logger.info("=== Deploying Event-Driven ML Infrastructure ===")
    logger.info("Region: %s | Account: %s | Bucket: %s", REGION, ACCOUNT_ID, BUCKET)

    create_bucket_if_missing(BUCKET)           # 1. S3 bucket
    upload_pipeline_scripts()                  # 2. preprocess.py + evaluate.py → S3
    lambda_role_arn = create_lambda_role()     # 3. Lambda IAM role
    _               = create_sagemaker_role()  # 4. SageMaker IAM role
    lambda_arn      = deploy_lambda(lambda_role_arn)   # 5. Lambda (both files zipped)
    enable_s3_eventbridge(BUCKET)              # 6. Enable S3 → EventBridge
    rule_arn        = create_eventbridge_rule(lambda_arn)  # 7. EventBridge rule
    topic_arn       = create_sns_alerts(alert_email)   # 8. SNS alerts

    logger.info("\n=== Deployment Complete ===")
    logger.info("Bucket:          s3://%s",  BUCKET)
    logger.info("Scripts:         s3://%s/pipeline-scripts/", BUCKET)
    logger.info("Lambda ARN:      %s", lambda_arn)
    logger.info("EventBridge ARN: %s", rule_arn)
    logger.info("SNS Topic ARN:   %s", topic_arn)
    logger.info("\nNext step: python sagemaker_pipeline.py --create")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy",   action="store_true", help="Deploy all infrastructure")
    parser.add_argument("--teardown", action="store_true", help="Delete all resources")
    parser.add_argument("--email",    default=None,        help="Alert email for SNS")
    args = parser.parse_args()

    if args.teardown:
        teardown()
    elif args.deploy:
        deploy(args.email)
    else:
        parser.print_help()