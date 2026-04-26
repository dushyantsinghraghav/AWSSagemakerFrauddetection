# Event-Driven ML Pipelines: SageMaker + EventBridge (2026)

> Automatically retrain your fraud detection model the moment new data lands in S3 — zero manual intervention.

---

## What This Project Does

This project builds a fully automated, event-driven ML retraining system on AWS. When a new CSV file is uploaded to an S3 bucket, Amazon EventBridge intercepts the upload event, fires an AWS Lambda function, which validates and deduplicates the event, then triggers a 5-step SageMaker Pipeline that preprocesses the data, trains an XGBoost model, evaluates it, and registers it in the SageMaker Model Registry — all without any human intervention.

---

## Architecture

```
S3 Upload (data/*.csv)
        ↓
Amazon EventBridge (Object Created rule)
        ↓
AWS Lambda (validate + dedup + retry logic)
        ↓
SageMaker Pipeline
    ├── Step 1: PreprocessData    (SKLearnProcessor · ml.t3.medium)
    ├── Step 2: TrainModel        (XGBoost built-in · ml.m5.large)
    ├── Step 3: EvaluateModel     (ScriptProcessor · ml.t3.medium)
    ├── Step 4: CheckAccuracy     (ConditionStep · accuracy ≥ 0.70)
    ├── Step 5a: RegisterModel    (Model Registry · PendingApproval)
    └── Step 5b: FailPipeline     (accuracy below threshold)
```

---

## Advanced Patterns Implemented

| Pattern | Implementation |
|---|---|
| Duplicate event handling | DynamoDB fingerprint dedup (MD5 of bucket+key+etag) with 24hr TTL |
| Idempotent pipelines | SageMaker execution history check + deterministic `ClientRequestToken` |
| Failure + retry logic | Exponential backoff (2s→4s→8s), SNS alerts, Dead Letter Queue |
| Model versioning | Semantic versioning (v1.0.0) stored in SSM Parameter Store with lineage tags |

---

## Project File Structure

```
event-driven-ml-pipeline/
├── setup_infrastructure.py          # Provisions all AWS resources
├── sagemaker_pipeline.py            # 5-step pipeline definition
├── lambda_trigger.py                # EventBridge → Lambda handler
├── advanced_pipeline_patterns.py    # Dedup, idempotency, retry, versioning
├── preprocess.py                    # Runs inside SKLearnProcessor container
├── evaluate.py                      # Runs inside ScriptProcessor container
├── sample_fraud_data.csv            # 500-row sample dataset for testing
└── README.md                        # This file
```

---

## AWS Resources Created

| Resource | Name |
|---|---|
| S3 Bucket | `sagemaker-ml-pipeline-{account_id}` |
| Lambda Function | `sagemaker-pipeline-trigger` |
| EventBridge Rule | `s3-new-data-trigger` |
| DynamoDB Table | `ml-pipeline-dedup` |
| SageMaker Pipeline | `fraud-detection-retrain` |
| Model Package Group | `FraudDetectionModels` |
| SNS Topic | `ml-pipeline-alerts` |
| IAM Role (Lambda) | `LambdaSageMakerTriggerRole` |
| IAM Role (SageMaker) | `SageMakerPipelineRole` |

---

## Deployment Steps

```bash
# 1. Install dependencies
pip install "sagemaker>=2.200.0,<3.0.0" boto3 scikit-learn xgboost pandas numpy

# 2. Configure AWS credentials
aws configure

# 3. Create DynamoDB dedup table
aws dynamodb create-table \
  --table-name ml-pipeline-dedup \
  --attribute-definitions AttributeName=event_id,AttributeType=S \
  --key-schema AttributeName=event_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region ap-southeast-2

aws dynamodb update-time-to-live \
  --table-name ml-pipeline-dedup \
  --time-to-live-specification Enabled=true,AttributeName=ttl \
  --region ap-southeast-2

# 4. Deploy infrastructure (bucket + scripts + IAM + Lambda + EventBridge)
python setup_infrastructure.py --deploy --email your@email.com

# 5. Register the pipeline
python sagemaker_pipeline.py --create

# 6. Test — upload CSV and watch it auto-trigger
aws s3 cp sample_fraud_data.csv s3://sagemaker-ml-pipeline-{account_id}/data/sample.csv

# 7. Watch the logs
aws logs tail /aws/lambda/sagemaker-pipeline-trigger --follow --region ap-southeast-2
```

---

## ⚠️ Bottlenecks & Gotchas

This section documents every blocker hit during implementation so you don't hit the same ones.

### 1. SageMaker SDK Version Conflict

**Problem:** `ModuleNotFoundError: No module named 'sagemaker.inputs'`

The default `pip install sagemaker` installs v3.x which has breaking changes. The pipeline code requires v2.x.

**Fix:**
```bash
pip uninstall sagemaker sagemaker-core sagemaker-mlops sagemaker-serve sagemaker-train -y
pip install "sagemaker>=2.200.0,<3.0.0"
```

---

### 2. Python Not Found on Mac

**Problem:** `zsh: command not found: python` and `zsh: command not found: pip`

Mac doesn't ship with Python pre-installed. Commands are `python3` and `pip3`.

**Fix:**
```bash
brew install python
python3 --version
pip3 --version
```

---

### 3. Wrong Jupyter Kernel

**Problem:** `AttributeError: module 'sagemaker' has no attribute '__version__'`

The default **Python 3 (ipykernel)** kernel in SageMaker Studio has an old broken SDK. Always use the **SageMaker Distribution** kernel image.

**Fix:** In Studio space settings, select **SageMaker Distribution 4.0.0** as the image before launching JupyterLab.

---

### 4. S3 Bucket Name Placeholder Not Replaced

**Problem:** `AccessDenied` when uploading — bucket `my-ml-data-bucket` doesn't belong to you.

The template code uses `my-ml-data-bucket` as a placeholder throughout multiple files.

**Fix:** Replace across all files using VS Code `Cmd+Shift+H`:
```
Find:    my-ml-data-bucket
Replace: sagemaker-ml-pipeline-{your_account_id}
```

---

### 5. IAM User Has No Permissions

**Problem:** `AccessDenied` on almost every AWS API call.

The `sagemaker-user` IAM user was created with no policies attached.

**Fix:** In AWS Console → IAM → Users → sagemaker-user → Add permissions:
```
AmazonS3FullAccess
AmazonSageMakerFullAccess
AWSLambda_FullAccess
AmazonEventBridgeFullAccess
AmazonSNSFullAccess
AmazonDynamoDBFullAccess
IAMFullAccess
```

---

### 6. Lambda ResourceConflictException

**Problem:** `ResourceConflictException: An update is in progress for resource: sagemaker-pipeline-trigger`

Lambda only allows one update operation at a time. Calling `update_function_code` and `update_function_configuration` back-to-back without waiting causes this.

**Fix:** Added `waiter.wait()` between each Lambda update operation in `setup_infrastructure.py`:
```python
lam.update_function_code(...)
waiter.wait(FunctionName=LAMBDA_FUNCTION_NAME)   # wait before next call
lam.update_function_configuration(...)
waiter.wait(FunctionName=LAMBDA_FUNCTION_NAME)
```

---

### 7. Cross-Account Role ARN

**Problem:** `ValidationException: Cross-account pass role is not allowed`

The `ROLE_ARN` in `sagemaker_pipeline.py` still had the placeholder account ID `123456789012`.

**Fix:** Replace with your real account ID:
```python
ROLE_ARN = "arn:aws:iam::{your_account_id}:role/SageMakerPipelineRole"
```

---

### 8. ml.m5.medium Not Supported for Processing Jobs

**Problem:** `ValidationException: Value 'ml.m5.medium' for 'ProcessingInstanceType' failed to satisfy enum value set`

SageMaker Processing jobs do not support `ml.m5.medium`. The cheapest supported options are `ml.t3.medium` and `ml.t3.large`.

**Fix:** Use `ml.t3.medium` for preprocessing and evaluation steps:
```python
instance_type="ml.t3.medium"
```

---

### 9. advanced_pipeline_patterns.py Not Bundled in Lambda ZIP

**Problem:** `Runtime.ImportModuleError: No module named 'advanced_pipeline_patterns'`

`setup_infrastructure.py` originally only zipped `lambda_trigger.py`. Since `lambda_trigger.py` imports from `advanced_pipeline_patterns.py`, both files must be in the ZIP.

**Fix:** Updated `_zip_lambda()` in `setup_infrastructure.py`:
```python
def _zip_lambda() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(THIS_DIR / "lambda_trigger.py",             arcname="lambda_function.py")
        zf.write(THIS_DIR / "advanced_pipeline_patterns.py", arcname="advanced_pipeline_patterns.py")
    return buf.getvalue()
```

---

### 10. XGBoost Built-in Algorithm Errors

**Problem 1:** `ClientError: You can't override the metric definitions for Amazon SageMaker algorithms`

**Problem 2:** `TypeError: XGBoost.__init__() missing 1 required positional argument: 'entry_point'`

The `XGBoost` SDK class requires an `entry_point` script, and the built-in algorithm doesn't accept custom `metric_definitions`. Two separate errors from the same root cause — using the wrong class for built-in training.

**Fix:** Switch to the generic `Estimator` class with the built-in XGBoost image URI:
```python
from sagemaker.estimator import Estimator

image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost", region=REGION, version="1.7-1", image_scope="training"
)
xgb_estimator = Estimator(
    image_uri=image_uri,
    instance_type=training_instance_type,
    hyperparameters={...},   # no metric_definitions, no entry_point
)
```

---

### 11. Pasting Multi-line Commands in Terminal

**Problem:** `quote>` prompt appears — terminal hangs waiting for input.

Pasting multiple lines with `#` comments or special characters (like email addresses with `.`) confuses the shell parser.

**Fix:** Always run commands **one line at a time**. Never paste blocks with comments into the terminal.

---

### 12. Environment Variable Overriding Config

**Problem:** Script shows `Bucket: my-ml-data-bucket` even after editing the file.

An old `TRAINING_DATA_BUCKET` environment variable set in a previous session was overriding the default in the code.

**Fix:**
```bash
unset TRAINING_DATA_BUCKET
echo $TRAINING_DATA_BUCKET   # should print nothing
```

---

## What Was Achieved

- ✅ Fully automated event-driven ML pipeline on AWS
- ✅ S3 upload automatically triggers retraining with zero manual steps
- ✅ Duplicate event protection via DynamoDB deduplication
- ✅ Idempotent pipeline executions — safe to re-trigger
- ✅ Exponential backoff retry with SNS email alerts on failure
- ✅ Semantic model versioning with full lineage tracking
- ✅ Model registered in SageMaker Model Registry pending human approval
- ✅ End-to-end tested with real fraud detection dataset (500 rows, 10.6% fraud rate)
- ✅ Infrastructure fully reproducible — one command to deploy, one to tear down

---

## Monitoring Commands

```bash
# Watch Lambda logs live
aws logs tail /aws/lambda/sagemaker-pipeline-trigger --follow --region ap-southeast-2

# Check pipeline execution steps
aws sagemaker list-pipeline-execution-steps \
  --pipeline-execution-arn $(aws sagemaker list-pipeline-executions \
    --pipeline-name fraud-detection-retrain \
    --region ap-southeast-2 \
    --query "PipelineExecutionSummaries[0].PipelineExecutionArn" \
    --output text) \
  --region ap-southeast-2 \
  --query "PipelineExecutionSteps[*].{Step:StepName,Status:StepStatus,Failure:FailureReason}" \
  --output table

# Roll back to a previous model version
python advanced_pipeline_patterns.py --rollback v1.3.0

# Check current model version
python advanced_pipeline_patterns.py --current-version
```

---

## Teardown

```bash
python setup_infrastructure.py --teardown
python sagemaker_pipeline.py --delete
aws dynamodb delete-table --table-name ml-pipeline-dedup --region ap-southeast-2
```

---

*Built with Amazon SageMaker Pipelines · EventBridge · Lambda · DynamoDB · SNS · Python 3.12 · 2026*
